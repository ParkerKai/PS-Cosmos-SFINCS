"""
Make low and high uncertainty variants of an existing set of SFINCS runs.

Python port of `create_boundary_synthetic_v08_uncert.m`. The script works on
boundary conditions that are already on disk (written by, for example,
`2026-10-09-Snohomish-bc.py`) and does not change the base runs.

Every folder under `runs_root` that holds a `sfincs.inp` counts as a run, at
any depth. Each variant gets a copy of that run at

    <runs_root>/<variant>/<relative path of the base run>

    2026-05-14-Snohomish/1942        ->  2026-05-14-Snohomish/low/1942
    <parent>/000/S000                ->  <parent>/low/000/S000

A variant shifts three inputs at once, which assumes their errors are fully
dependent. "low" is the best case and "high" the worst case.
  - bed level   : weir crests and subgrid elevations move by a constant. The
                  shifted files are written once, next to the originals
                  (sfincs_higher.weir, sfincs_subgrid_higher.nc, ...).
  - water level : sfincs.bzs gets a constant offset that grows linearly from
                  zero over the spin-up buffer, so the run starts from the
                  same state as the base run.
  - discharge   : sfincs.dis is multiplied by a factor.

A variant run folder holds only sfincs.inp, sfincs.bzs and sfincs.dis. Its inp
points one level further up for the shared static inputs.

Run in the conda env `hydromt-sfincs-dev` (uses numpy and netCDF4):
    python make_uncertainty_runs.py
Then point `make_snohomish_slurm.py` at <runs_root>/low and <runs_root>/high.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import netCDF4
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
runs_root    = Path(r"p:\CoSMoS_PS\10_Kitsap\20260526_improve_scripts\2026-05-14-Snohomish")

# "low" is the best case (higher bed, lower water level, less discharge) and
# "high" the worst case. `tag` names the shifted static files.
# These are the values used so far: bed +-0.50 m, water level +-0.50 m,
# discharge +-25 %. The skill analysis of 2026-10-06
# (P:\CoSMoS_PS\12_uncertainty\RESULTS_uncertainty_values.md) proposes
# bed +-0.50 m, water level +-0.30 m and discharge +-50 %. Andy O'Neill still
# has to decide on those.
variants = {
    "low":  dict(dz_bed=+0.50, dz_wl=-0.50, q_factor=0.75, tag="higher"),
    "high": dict(dz_bed=-0.50, dz_wl=+0.50, q_factor=1.25, tag="lower"),
}

perturb_bed  = True     # shift weirs + subgrid
perturb_wl   = True     # offset sfincs.bzs
perturb_q    = True     # scale sfincs.dis

ramp_days    = 7.0      # days over which the water-level offset grows from 0 to full (the spin-up buffer)
exclude_dirs = ["benchmark"]   # folders under runs_root with a sfincs.inp that are not runs
only_runs    = []       # e.g. ["1996"] to test on a few runs. Empty means all runs
overwrite    = False    # False skips variant runs and static files that already exist

# Subgrid variables that hold absolute elevations. The others are relative to these
sbg_elev_vars = ["z_zmin", "z_zmax", "z_level", "uv_zmin", "uv_zmax"]
# =============================================================================


def read_inp(fn: Path) -> dict[str, str]:
    """sfincs.inp as an ordered {key: value} dict."""
    inp = {}
    for ln in fn.read_text().splitlines():
        if "=" in ln:
            key, value = ln.split("=", 1)
            inp[key.strip()] = value.strip()
    return inp


def write_inp(fn: Path, inp: dict[str, str]) -> None:
    # write_bytes keeps LF line endings when this runs on Windows (the runs go to Linux)
    txt = "".join(f"{k:<21}= {v}\n" for k, v in inp.items())
    fn.write_bytes(txt.encode("utf-8"))


def write_table(fn: Path, t: np.ndarray, data: np.ndarray) -> None:
    """ASCII bzs/dis: time in seconds + one column per point, LF line endings."""
    fmt = ["%.1f"] + ["%.3f"] * data.shape[1]
    with open(fn, "w", newline="\n") as f:
        np.savetxt(f, np.column_stack([t, data]), fmt=fmt)


def shift_weir(src: Path, dst: Path, dz: float) -> None:
    """Copy a sfincs.weir (tekal blocks: name, 'nrows ncols', x y z par1) with z + dz."""
    lines = src.read_text().splitlines()
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        out.append(lines[i])                      # block name
        nrows = int(lines[i + 1].split()[0])
        out.append(lines[i + 1])
        for ln in lines[i + 2:i + 2 + nrows]:
            p = ln.split()
            p[2] = f"{float(p[2]) + dz:.3f}"
            out.append(f"{p[0]:>12} {p[1]:>12} " + " ".join(p[2:]))
        i += 2 + nrows
    dst.write_bytes(("\n".join(out) + "\n").encode("utf-8"))


def shift_subgrid(src: Path, dst: Path, dz: float) -> None:
    """Copy the subgrid file and shift its elevation variables in place. A uniform
    shift is exact, because volumes, depths, Manning and wet fractions are all
    stored relative to the elevations."""
    shutil.copy2(src, dst)
    with netCDF4.Dataset(dst, "r+") as nc:
        for v in sbg_elev_vars:
            nc[v][:] = nc[v][:] + dz
        nc.setncattr("bed_shift_m", float(dz))


def find_runs(root: Path) -> list[Path]:
    skip = set(variants) | set(exclude_dirs)
    runs = []
    for fn in sorted(root.rglob("sfincs.inp")):
        rel = fn.parent.relative_to(root)
        if not rel.parts or skip & set(rel.parts):
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
print(f"Found {len(runs)} runs under {runs_root}: {runs[0].name} .. {runs[-1].name}")

# --- Static inputs: shifted weir and subgrid, once per variant and source file ---
static_done: dict[tuple[str, Path], Path] = {}

def shifted_static(src: Path, variant: str) -> Path:
    """Shifted copy of a weir or subgrid file, next to the original. Made on first use."""
    if (variant, src) in static_done:
        return static_done[(variant, src)]
    v = variants[variant]
    dst = src.with_name(f"{src.stem}_{v['tag']}{src.suffix}")
    if dst.exists() and not overwrite:
        print(f"  keeping existing {dst.name}")
    else:
        print(f"  writing {dst.name} (bed {v['dz_bed']:+.2f} m)")
        if src.suffix == ".nc":
            shift_subgrid(src, dst, v["dz_bed"])
        else:
            shift_weir(src, dst, v["dz_bed"])
    static_done[(variant, src)] = dst
    return dst


# --- Per run: inp, bzs, dis ---
n_written = 0
for irun, run in enumerate(runs, start=1):
    rel = run.relative_to(runs_root)
    inp = read_inp(run / "sfincs.inp")
    todo = [name for name in variants
            if overwrite or not (runs_root / name / rel / "sfincs.inp").exists()]
    print(f"[{irun}/{len(runs)}] {rel.as_posix()}" + ("" if todo else "  (exists, skipped)"))
    if not todo:
        continue

    bzs = np.loadtxt(run / inp["bzsfile"])
    dis = np.loadtxt(run / inp["disfile"])
    # 0 at the first time step, 1 after ramp_days
    ramp = np.clip((bzs[:, 0] - bzs[0, 0]) / (ramp_days * 86400.0), 0.0, 1.0)

    for name in todo:
        v = variants[name]
        dst = runs_root / name / rel
        dst.mkdir(parents=True, exist_ok=True)

        new = dict(inp)
        # Variant runs sit one level deeper, so shared inputs need one more "../"
        for key, value in inp.items():
            if key.endswith("file") and value.startswith("../"):
                new[key] = "../" + value
        if perturb_bed:
            for key in ("weirfile", "sbgfile"):
                if key in inp:
                    src = (run / inp[key]).resolve()
                    new[key] = new[key].replace(src.name, shifted_static(src, name).name)

        dz_wl = v["dz_wl"] if perturb_wl else 0.0
        q_fac = v["q_factor"] if perturb_q else 1.0
        write_table(dst / Path(inp["bzsfile"]).name, bzs[:, 0], bzs[:, 1:] + dz_wl * ramp[:, None])
        write_table(dst / Path(inp["disfile"]).name, dis[:, 0], dis[:, 1:] * q_fac)
        # The inp goes last. Its presence marks the variant run as complete
        write_inp(dst / "sfincs.inp", new)
        n_written += 1

print(f"Wrote {n_written} variant runs")

# --- Check: read back the first run of each variant against its base ---
print("\nCheck against base (first run):")
run = runs[0]
rel = run.relative_to(runs_root)
inp = read_inp(run / "sfincs.inp")
bzs0 = np.loadtxt(run / inp["bzsfile"])
dis0 = np.loadtxt(run / inp["disfile"])
for name in variants:
    dst = runs_root / name / rel
    new = read_inp(dst / "sfincs.inp")
    d_bzs = np.loadtxt(dst / new["bzsfile"])[:, 1:] - bzs0[:, 1:]
    r_dis = np.loadtxt(dst / new["disfile"])[:, 1:].sum() / dis0[:, 1:].sum()
    missing = [k for k, val in new.items() if k.endswith("file") and val.startswith("../")
               and not (dst / val).exists()]
    print(f"  {name:<5} {rel.as_posix()}: WL offset first step {d_bzs[0].mean():+.3f} m, "
          f"last step {d_bzs[-1].mean():+.3f} m; discharge x{r_dis:.3f}; "
          f"weir {new.get('weirfile')}, subgrid {new.get('sbgfile')}; "
          f"missing inputs: {missing or 'none'}")
    if perturb_bed and "sbgfile" in new:
        with netCDF4.Dataset(run / inp["sbgfile"]) as a, netCDF4.Dataset(dst / new["sbgfile"]) as b:
            dz = {v: float(np.mean(b[v][:] - a[v][:])) for v in sbg_elev_vars}
        print(f"        subgrid shift: " + ", ".join(f"{k} {val:+.3f}" for k, val in dz.items()))
