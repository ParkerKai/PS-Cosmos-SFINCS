"""
Make low and high uncertainty variants of an existing set of SFINCS runs.

Python port of `create_boundary_synthetic_v08_uncert.m`. The script works on
boundary conditions that are already on disk and does not change the base runs.

Every folder below `runs_root` that holds a `sfincs.inp` counts as a run, at
any depth, except `runs_root` itself, the variant folders and `exclude_dirs`.
Each variant gets a copy of that run at

    <runs_root>/<variant>/<relative path of the base run>

    <runs_root>/1942          ->  <runs_root>/low/1942
    <runs_root>/000/S000      ->  <runs_root>/low/000/S000

A variant shifts three inputs at once, which assumes their errors are fully
dependent. "low" is the best case and "high" the worst case.
  - bed level   : weir crests and subgrid elevations move by a constant. The
                  shifted files are written once, next to the originals
                  (sfincs_higher.weir, sfincs_subgrid_higher.nc, ...). zsini
                  and qinf_zmin in the inp move with the bed, so the same
                  cells start wet and infiltrate as in the base run.
  - water level : sfincs.bzs gets a constant offset that grows linearly from
                  zero at tstart to its full value after `ramp_days`.
  - discharge   : sfincs.dis is multiplied by a factor, from the first step.

A variant run folder holds only sfincs.inp, sfincs.bzs and sfincs.dis. All
other input files stay where they are and the variant inp points to them.
Only ASCII bzs and dis files are perturbed. A run that takes its water levels
from a netCDF boundary file is refused.

Each variant folder gets an `uncertainty_settings.json` with the values used.
After changing the values, set `overwrite = True`, otherwise the script stops
instead of mixing old and new runs.

Run in the conda env `hydromt-sfincs-dev` (uses numpy and netCDF4):
    python make_uncertainty_runs.py
Then make the SLURM files for <runs_root>/low and <runs_root>/high.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
runs_root    = Path(r"p:\CoSMoS_PS\10_Kitsap\20260526_improve_scripts\2026-05-14-Snohomish")

# "low" is the best case (higher bed, lower water level, less discharge) and
# "high" the worst case.
#   dz_bed   : m added to the weir crests and subgrid elevations (+ is a higher bed)
#   dz_wl    : m added to the water-level boundary
#   q_factor : multiplier on the discharge
#   tag      : suffix of the shifted weir and subgrid files. It describes the bed
#              direction, so "low" uses the "higher" bed
# These are the values used so far: bed +-0.50 m, water level +-0.50 m,
# discharge +-25 %. The skill analysis of 2026-10-06
# (P:\CoSMoS_PS\12_uncertainty\RESULTS_uncertainty_values.md) proposes
# bed +-0.50 m, water level +-0.30 m and discharge +-50 %. Andy O'Neill still
# has to decide on those.
variants = {
    "low":  dict(dz_bed=+0.50, dz_wl=-0.50, q_factor=0.75, tag="higher"),
    "high": dict(dz_bed=-0.50, dz_wl=+0.50, q_factor=1.25, tag="lower"),
}

perturb_bed  = True     # shift weirs + subgrid (and zsini, qinf_zmin)
perturb_wl   = True     # offset sfincs.bzs
perturb_q    = True     # scale sfincs.dis

ramp_days    = 7.0      # days after tstart over which the water-level offset grows from 0 to full.
                        # Keep it equal to the spin-up buffer the boundary conditions were built with
exclude_dirs = ["benchmark"]   # folder names below runs_root with a sfincs.inp that are not runs
only_runs    = []       # e.g. ["1996"] to test on a few runs. Empty means all runs
overwrite    = False    # False skips variant runs that already exist. Set True after changing
                        # `variants` or `ramp_days`, so that everything is written again

# Subgrid variables that hold absolute elevations. The others (volumes, depths,
# Manning, wet fractions) are relative to these (SFINCS sfincs_subgrid.F90)
sbg_elev_vars = ["z_zmin", "z_zmax", "z_level", "uv_zmin", "uv_zmax"]
# =============================================================================


def read_inp(fn: Path) -> dict[str, str]:
    """sfincs.inp as an ordered {key: value} dict."""
    inp = {}
    for ln in fn.read_text(encoding="utf-8").splitlines():
        if "=" in ln:
            key, value = ln.split("=", 1)
            # SFINCS uses the first occurrence of a key
            inp.setdefault(key.strip(), value.strip())
    return inp


def write_inp(fn: Path, inp: dict[str, str]) -> None:
    # write_bytes keeps LF line endings when this runs on Windows (the runs go to Linux).
    # Keys are padded to 21 characters, the layout SFINCS inp files normally have
    txt = "".join(f"{k:<21}= {v}\n" for k, v in inp.items())
    fn.write_bytes(txt.encode("utf-8"))


def inp_seconds(inp: dict[str, str], key: str) -> float:
    """Time keyword (yyyymmdd HHMMSS) in seconds after tref."""
    def parse(value: str) -> datetime:
        date, _, clock = value.partition(" ")
        return datetime.strptime(date + clock.strip().ljust(6, "0"), "%Y%m%d%H%M%S")
    return (parse(inp[key]) - parse(inp["tref"])).total_seconds()


def read_table(fn: Path) -> np.ndarray:
    # ndmin=2 keeps a single-row file two-dimensional
    return np.loadtxt(fn, ndmin=2)


def write_table(fn: Path, t: np.ndarray, data: np.ndarray) -> None:
    """ASCII bzs/dis: time in seconds + one column per point, LF line endings."""
    fmt = ["%.1f"] + ["%.3f"] * data.shape[1]
    with fn.open("w", newline="\n") as f:
        np.savetxt(f, np.column_stack([t, data]), fmt=fmt)


def rel_posix(target: Path, start: Path) -> str:
    """Path of `target` as seen from folder `start`, with forward slashes for the inp."""
    return Path(os.path.relpath(target, start)).as_posix()


def shift_weir(src: Path, dst: Path, dz: float) -> None:
    """Copy a sfincs.weir with the crest level (third column) raised by dz. The file
    is a series of blocks: a name line, an 'nrows ncols' line, then nrows data lines."""
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out, i = [], 0
    while i < len(lines):
        nrows = int(lines[i + 1].split()[0])
        rows = lines[i + 2:i + 2 + nrows]
        if len(rows) < nrows:
            raise ValueError(f"{src}: block '{lines[i].strip()}' has fewer than {nrows} rows")
        out += lines[i:i + 2]
        for ln in rows:
            p = ln.split()
            p[2] = f"{float(p[2]) + dz:.3f}"
            out.append(f"{p[0]:>12} {p[1]:>12} " + " ".join(p[2:]))
        i += 2 + nrows
    dst.write_bytes(("\n".join(out) + "\n").encode("utf-8"))


def shift_subgrid(src: Path, dst: Path, dz: float) -> None:
    """Copy the subgrid file and shift its elevation variables. netCDF4 in r+ mode
    edits the copy in place and leaves every other variable and its encoding as
    it is, which a round trip through xarray would not guarantee."""
    shutil.copy2(src, dst)
    with netCDF4.Dataset(dst, "r+") as nc:
        for var in sbg_elev_vars:
            nc[var][:] = nc[var][:] + dz
        nc.setncattr("bed_shift_m", float(dz))


def make_shifted(src: Path, dz: float, tag: str) -> Path:
    """Write the shifted copy of a weir or subgrid file next to the original."""
    dst = src.with_name(f"{src.stem}_{tag}{src.suffix}")
    print(f"  writing {dst.name} (bed {dz:+.2f} m)")
    # Write under a temporary name first, so an interrupted run leaves no half-made file
    tmp = dst.with_name(dst.name + ".tmp")
    if src.suffix == ".nc":
        shift_subgrid(src, tmp, dz)
    else:
        shift_weir(src, tmp, dz)
    tmp.replace(dst)
    return dst


def find_runs(root: Path) -> list[Path]:
    runs = []
    for fn in sorted(root.rglob("sfincs.inp")):
        rel = fn.parent.relative_to(root)
        # Skip runs_root itself (a template inp) and the variant folders this script makes
        if not rel.parts or rel.parts[0] in variants:
            continue
        if set(exclude_dirs) & set(rel.parts):
            continue
        if only_runs and not set(only_runs) & set(rel.parts):
            continue
        runs.append(fn.parent)
    if not runs:
        raise RuntimeError(f"No run folders with sfincs.inp found under {root}")
    return runs


# =============================================================================
# Main
# =============================================================================
runs = find_runs(runs_root)
rels = [run.relative_to(runs_root) for run in runs]
print(f"Found {len(runs)} runs under {runs_root}: {rels[0].as_posix()} .. {rels[-1].as_posix()}")

# --- Settings per variant: stop if existing runs were made with other values ---
settings = {}
for name, cfg in variants.items():
    settings[name] = dict(dz_bed=cfg["dz_bed"] if perturb_bed else 0.0,
                          dz_wl=cfg["dz_wl"] if perturb_wl else 0.0,
                          q_factor=cfg["q_factor"] if perturb_q else 1.0,
                          tag=cfg["tag"], ramp_days=ramp_days)
    variant_dir = runs_root / name
    settings_fn = variant_dir / "uncertainty_settings.json"
    if variant_dir.exists() and not overwrite:
        if not settings_fn.exists():
            raise RuntimeError(f"{variant_dir} exists but has no {settings_fn.name}. "
                               "Set overwrite = True to write it again.")
        if json.loads(settings_fn.read_text(encoding="utf-8")) != settings[name]:
            raise RuntimeError(f"{variant_dir} was made with other values than the configuration "
                               "has now. Set overwrite = True to write it again.")
    variant_dir.mkdir(exist_ok=True)
    settings_fn.write_bytes((json.dumps(settings[name], indent=2) + "\n").encode("utf-8"))

# --- Per run: inp, bzs, dis. Shifted weir and subgrid files are made on first use ---
shifted: dict[tuple[str, Path], Path] = {}   # (variant, original file) -> shifted file
n_written = 0
t_begin = time.perf_counter()
for irun, (run, rel) in enumerate(zip(runs, rels), start=1):
    inp = read_inp(run / "sfincs.inp")
    todo = [name for name in variants
            if overwrite or not (runs_root / name / rel / "sfincs.inp").exists()]
    if not todo:
        print(f"[{irun}/{len(runs)}] {rel.as_posix()}  (exists, skipped)")
        continue
    if "bzsfile" not in inp or "disfile" not in inp:
        raise RuntimeError(f"{run}: needs an ASCII bzsfile and disfile in sfincs.inp")

    bzs = read_table(run / inp["bzsfile"])
    dis = read_table(run / inp["disfile"])
    # 0 at tstart, 1 from ramp_days later on. bzs times are seconds after tref
    ramp = np.clip((bzs[:, 0] - inp_seconds(inp, "tstart")) / (ramp_days * 86400.0), 0.0, 1.0)

    for name in todo:
        cfg = settings[name]
        dst = runs_root / name / rel
        dst.mkdir(parents=True, exist_ok=True)

        # Point every input file back to where it is now. inp paths use forward slashes
        new_inp = dict(inp)
        for key, value in inp.items():
            src = run / value.replace("\\", "/")
            if not key.endswith("file") or not src.is_file():
                continue
            # resolve() gives the real file, so a shifted copy lands next to it
            src = src.resolve()
            if perturb_bed and key in ("weirfile", "sbgfile"):
                if (name, src) not in shifted:
                    shifted[(name, src)] = make_shifted(src, cfg["dz_bed"], cfg["tag"])
                src = shifted[(name, src)]
            new_inp[key] = rel_posix(src, dst.resolve())
        # bzs and dis are the two files written into the variant folder itself
        new_inp["bzsfile"] = "sfincs.bzs"
        new_inp["disfile"] = "sfincs.dis"

        if perturb_bed:
            # Two thresholds in the inp are absolute levels and move with the bed:
            # cells below zsini start wet, cells above qinf_zmin infiltrate
            new_inp["zsini"] = f"{float(inp.get('zsini', 0.0)) + cfg['dz_bed']:g}"
            if float(inp.get("qinf", 0.0)) > 0.0:
                new_inp["qinf_zmin"] = f"{float(inp.get('qinf_zmin', 0.0)) + cfg['dz_bed']:g}"

        # ramp[:, None] applies the same ramp to every boundary point (column)
        write_table(dst / "sfincs.bzs", bzs[:, 0], bzs[:, 1:] + cfg["dz_wl"] * ramp[:, None])
        write_table(dst / "sfincs.dis", dis[:, 0], dis[:, 1:] * cfg["q_factor"])
        # The inp goes last. Its presence marks the variant run as complete
        write_inp(dst / "sfincs.inp", new_inp)
        n_written += 1

    elapsed = time.perf_counter() - t_begin
    eta = elapsed / irun * (len(runs) - irun)
    print(f"[{irun}/{len(runs)}] {rel.as_posix()}  elapsed {elapsed / 60:.1f} min, "
          f"about {eta / 60:.1f} min to go")

print(f"Wrote {n_written} variant runs in {(time.perf_counter() - t_begin) / 60:.1f} min")

# --- Check the first run of each variant against its base run and the configuration ---
print(f"\nCheck of {rels[0].as_posix()} against its base run:")
run, rel = runs[0], rels[0]
inp = read_inp(run / "sfincs.inp")
bzs = read_table(run / inp["bzsfile"])
dis = read_table(run / inp["disfile"])
ramp = np.clip((bzs[:, 0] - inp_seconds(inp, "tstart")) / (ramp_days * 86400.0), 0.0, 1.0)
for name, cfg in settings.items():
    dst = runs_root / name / rel
    new_inp = read_inp(dst / "sfincs.inp")
    problems = [f"{key} = {value} does not exist" for key, value in new_inp.items()
                if key.endswith("file") and (run / inp.get(key, "")).is_file()
                and not (dst / value).is_file()]
    # 1 mm tolerance: both files are written with three decimals
    if not np.allclose(read_table(dst / "sfincs.bzs")[:, 1:],
                       bzs[:, 1:] + cfg["dz_wl"] * ramp[:, None], atol=1.1e-3):
        problems.append("sfincs.bzs is not base + dz_wl * ramp")
    if not np.allclose(read_table(dst / "sfincs.dis")[:, 1:], dis[:, 1:] * cfg["q_factor"], atol=1.1e-3):
        problems.append("sfincs.dis is not base * q_factor")
    if perturb_bed and "sbgfile" in inp:
        with netCDF4.Dataset(run / inp["sbgfile"]) as base, netCDF4.Dataset(dst / new_inp["sbgfile"]) as new:
            for var in sbg_elev_vars:
                if abs(float(np.mean(new[var][:] - base[var][:])) - cfg["dz_bed"]) > 1e-3:
                    problems.append(f"subgrid {var} is not shifted by {cfg['dz_bed']:+.2f} m")
    if problems:
        raise RuntimeError(f"{name}/{rel.as_posix()}: " + "; ".join(problems))
    print(f"  {name:<5} OK: bed {cfg['dz_bed']:+.2f} m ({new_inp.get('sbgfile')}, {new_inp.get('weirfile')}, "
          f"zsini {new_inp.get('zsini')}), water level {cfg['dz_wl']:+.2f} m after {ramp_days:g} days, "
          f"discharge x{cfg['q_factor']:.2f}")
