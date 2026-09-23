"""
PS-CoSMoS post-processing helpers.

PS-CoSMoS-specific routines used by `downscale_sfincs_results.py`. Generic
downscaling and disconnected-flood removal live in
`hydromt_sfincs.workflows.downscaling`; this module covers:

  - Water-year aggregation of SFINCS NetCDF outputs (per-cell annual maxima
    of zsmax + extras at the peak time).
  - Cell-level extreme value analysis: empirical Weibull plotting position,
    per-cell GEV fit, and Peaks-Over-Threshold (POT) with declustering.
  - Mapping quadtree-grid variables (e.g. qmax, tmax) to a high-resolution
    DEM grid via a pre-computed index COG.
  - Hazard-category binning of depth / velocity rasters and a unified
    Cloud-Optimized GeoTIFF writer.
  - Infastructure to convert COG rasters to vectorized shapefiles with geopandas and rasterio.

Sections 0a-0e inline utilities formerly kept in `POT_Extremes.py` and
`Xarray_NCtools.py` so the whole PS-CoSMoS-specific stack is one import.
"""

from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Mapping, Any
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
import math
import xarray as xr
from scipy.stats import genextreme
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from rasterio.features import shapes, sieve
from rasterio.windows import Window
from pyproj import CRS


# =============================================================================
# SECTION 0: Inlined utilities
# =============================================================================

# --- 0a: time-delta parsing + 1D series validation --------------------------

_UNIT_TO_NS = {
    "ns": 1,
    "us": 1_000,
    "µs": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "h": 3600 * 1_000_000_000,
    "D": 86_400 * 1_000_000_000,
    "d": 86_400 * 1_000_000_000,
}


def rp_tag(rp: float) -> str:
    return f"RP{int(round(rp)):03d}"



def infer_water_year_from_time_array(t_values: np.ndarray) -> int:
    """
    Infer water year (WY) from an array of datetime-like values.
    Convention: WY = year + 1 when month >= 10 (Oct–Dec), else year.
    Uses the **first** valid timestamp in the array.
    """
    pdt = pd.to_datetime(t_values)
    if pdt.size == 0:
        raise ValueError("Cannot infer water year: empty time coordinate.")
    dt0 = pdt[0]
    wy = int(dt0.year + (1 if dt0.month >= 10 else 0))
    return wy


def _parse_timedelta_to_ns(r) -> int:
    """Parse '24h' / '72h' / timedelta64 / pd.Timedelta to nanoseconds."""
    if isinstance(r, np.timedelta64):
        return int(r.astype("timedelta64[ns]").astype(np.int64))
    if isinstance(r, pd.Timedelta):
        return int(r.to_numpy().astype("timedelta64[ns]").astype(np.int64))
    if not isinstance(r, str):
        raise ValueError(f"`r` must be str/timedelta64/Timedelta; got {type(r)}")

    s = r.strip()
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0 or i == len(s):
        raise ValueError(f"Invalid time delta string: {r!r}")
    mag = int(s[:i])
    unit = s[i:].strip()
    if unit not in _UNIT_TO_NS:
        raise ValueError(f"Unsupported unit in {r!r}; use {sorted(_UNIT_TO_NS)}")
    return mag * _UNIT_TO_NS[unit]


def _validate_1d_time_series(da: xr.DataArray, time_dim: str) -> None:
    if time_dim not in da.dims:
        raise ValueError(f"`time_dim` {time_dim!r} not in DataArray dims")
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(f"DataArray must be numeric; got dtype={da.dtype}")
    if da.ndim != 1 or da.dims[0] != time_dim:
        raise ValueError(
            f"Expected 1D DataArray indexed by {time_dim!r}; got dims={da.dims}"
        )
    coord = da[time_dim]
    if not np.issubdtype(coord.dtype, np.datetime64):
        raise TypeError(f"{time_dim!r} must be datetime64; got {coord.dtype}")
    t = coord.values
    if t.size >= 2 and not np.all(
        np.diff(t).astype("timedelta64[ns]") > np.timedelta64(0, "ns")
    ):
        raise ValueError("Time coordinate must be strictly increasing.")


# --- Assumes you already have rp_tag(rp: float) -> str defined elsewhere ---
# Example fallback if needed:
# def rp_tag(rp: float, decimals: int = 0) -> str:
#     """Safe tag for floats: 2.5 -> 'rp2p5'; 100 -> 'rp100'."""
#     s = f"{rp:.{decimals}f}" if decimals > 0 else str(int(rp)) if rp.is_integer() else str(rp)
#     return f"rp{s.replace('.', 'p')}"


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    shapefile_dir: Path
    domain_stem: str
    provenance_tag: str
    raster_ext: str = ".tif"
    vector_ext: str = ".shp"

    # --- Core builders ---
    def raster(self, name: str, rp: float) -> Path:
        return (
            self.output_dir
            / f"{name}_{rp_tag(rp)}_{self.domain_stem}_{self.provenance_tag}{self.raster_ext}"
        )

    def raster_smooth(self, name: str, rp: float) -> Path:
        return (
            self.output_dir
            / f"{name}_smooth_{rp_tag(rp)}_{self.domain_stem}_{self.provenance_tag}{self.raster_ext}"
        )

    def vector(self, name: str, rp: float) -> Path:
        return (
            self.shapefile_dir
            / f"{name}_{rp_tag(rp)}_{self.domain_stem}_{self.provenance_tag}{self.vector_ext}"
        )

    # --- Convenience for masked variants ---
    @staticmethod
    def masked(p: Path) -> Path:
        return p.parent / f"{p.stem}_masked{p.suffix}"

    # --- Specific raster helpers (mirroring your original names) ---
    def hmax(self, rp: float) -> Path:
        return self.raster("hmax", rp)

    def hmax_smooth(self, rp: float) -> Path:
        return self.raster_smooth("hmax", rp)

    def zsmax(self, rp: float) -> Path:
        return self.raster("zsmax", rp)

    def hmax_masked(self, rp: float) -> Path:
        return self.masked(self.hmax(rp))

    def zsmax_masked(self, rp: float) -> Path:
        return self.masked(self.zsmax(rp))

    def connection(self, rp: float) -> Path:
        return self.raster("connection", rp)

    def extra(self, var: str, rp: float) -> Path:
        return self.raster(var, rp)

    def extra_smooth(self, var: str, rp: float) -> Path:
        return self.raster_smooth(var, rp)

    def depth_bins(self, rp: float) -> Path:
        return self.raster("depth_bins", rp)

    def qmax_bins(self, rp: float) -> Path:
        return self.raster("qmax_bins", rp)

    # --- Vector (shapefile) helpers ---
    def depth_shapefile(self, rp: float) -> Path:
        return self.vector("depth_bins", rp)

    def qmax_shapefile(self, rp: float) -> Path:
        return self.vector("qmax_bins", rp)

    def extent_connected_shapefile(self, rp: float) -> Path:
        return self.vector("extent_connected", rp)

    def extent_disconnected_shapefile(self, rp: float) -> Path:
        return self.vector("extent_disconnected", rp)

    def extent_min_shapefile(self, rp: float) -> Path:
        return self.vector("extent_min", rp)

    def extent_max_shapefile(self, rp: float) -> Path:
        return self.vector("extent_max", rp)

    # --- Utilities ---
    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shapefile_dir.mkdir(parents=True, exist_ok=True)


# --- 0b: POT declustering + threshold search --------------------------------


def _cluster_extrema_1d(
    values: np.ndarray,
    times: np.ndarray,
    threshold: float,
    r_ns: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cluster maxima (times, values) for values > threshold with
    independence window `r_ns` (nanoseconds)."""
    mask = (values > float(threshold)) & np.isfinite(values)
    if not np.any(mask):
        return np.array([], dtype=times.dtype), np.array([], dtype=np.float64)

    ex_vals = values[mask]
    ex_times = times[mask]
    if ex_vals.size == 1:
        return ex_times.copy(), ex_vals.astype(np.float64)

    dt_ns = (
        ex_times[1:].astype("datetime64[ns]") - ex_times[:-1].astype("datetime64[ns]")
    ).astype(np.int64)
    gap_idx = np.flatnonzero(dt_ns > r_ns)

    starts = np.r_[0, gap_idx + 1]
    ends = np.r_[gap_idx + 1, ex_vals.size]
    out_t, out_v = [], []
    for s, e in zip(starts, ends):
        cvals = ex_vals[s:e]
        ctimes = ex_times[s:e]
        imax = np.nanargmax(cvals)
        out_v.append(float(cvals[imax]))
        out_t.append(ctimes[imax])
    return np.array(out_t, dtype=times.dtype), np.array(out_v, dtype=np.float64)


def _count_peaks_1d(values, times, threshold, r_ns) -> int:
    _, v = _cluster_extrema_1d(values, times, threshold, r_ns)
    return int(v.size)


def get_extremes_pot_xr(
    da: xr.DataArray,
    threshold: float,
    r: str = "24h",
    time_dim: str = "time",
    num_exce: Optional[int] = None,
) -> xr.DataArray:
    """Decluster Peaks-Over-Threshold from a 1D xr.DataArray."""
    _validate_1d_time_series(da, time_dim)
    r_ns = _parse_timedelta_to_ns(r)
    values = da.values.astype(float)
    times = da[time_dim].values
    evt_t, evt_v = _cluster_extrema_1d(values, times, threshold, r_ns)
    if evt_v.size == 0:
        raise ValueError("Threshold yields zero exceedances.")
    if num_exce is not None:
        if not isinstance(num_exce, int) or num_exce <= 0:
            raise ValueError("`num_exce` must be a positive integer.")
        if evt_v.size > num_exce:
            t_ns = evt_t.astype("datetime64[ns]").astype(np.int64)
            order = np.lexsort((-t_ns, -evt_v))
            sel = order[:num_exce]
            chrono = np.argsort(t_ns[sel])
            evt_t = evt_t[sel][chrono]
            evt_v = evt_v[sel][chrono]
    return xr.DataArray(
        data=evt_v,
        coords={time_dim: evt_t},
        dims=(time_dim,),
        name=da.name or "extreme_values",
        attrs=dict(**da.attrs),
    )


def pot_threshold_set_num_xr(
    da: xr.DataArray,
    r: str,
    num_exce: int,
    time_dim: str = "time",
    strategy: Literal["geq", "leq", "closest"] = "closest",
) -> float:
    """Binary-search a threshold yielding ~num_exce declustered peaks."""
    if not isinstance(num_exce, int) or num_exce < 0:
        raise ValueError("`num_exce` must be a non-negative integer.")
    _validate_1d_time_series(da, time_dim)
    r_ns = _parse_timedelta_to_ns(r)

    values = da.values.astype(float)
    times = da[time_dim].values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Input contains no finite values.")
    uniq = np.sort(finite)

    if np.unique(uniq).size == 1:
        u = float(np.unique(uniq)[0])
        return float(np.nextafter(u, -np.inf))

    def count_at(i: int) -> int:
        return _count_peaks_1d(values, times, uniq[i], r_ns)

    n = uniq.size

    if strategy == "geq":
        lo, hi, ans = 0, n - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_at(mid) >= num_exce:
                ans, hi = mid, mid - 1
            else:
                lo = mid + 1
        return float(uniq[ans] if ans is not None else uniq[-1])

    if strategy == "leq":
        lo, hi, ans = 0, n - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_at(mid) <= num_exce:
                ans, lo = mid, mid + 1
            else:
                hi = mid - 1
        return float(uniq[ans] if ans is not None else uniq[0])

    if strategy == "closest":
        lo, hi, geq_idx = 0, n - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_at(mid) >= num_exce:
                geq_idx, hi = mid, mid - 1
            else:
                lo = mid + 1
        candidates = [geq_idx, geq_idx + 1] if geq_idx is not None else [0]
        candidates = [c for c in candidates if c is not None and 0 <= c < n]
        best_i, best_diff = candidates[0], abs(count_at(candidates[0]) - num_exce)
        for c in candidates[1:]:
            d = abs(count_at(c) - num_exce)
            if d < best_diff:
                best_i, best_diff = c, d
        return float(uniq[best_i])

    raise ValueError("`strategy` must be one of {'geq','leq','closest'}.")


def rp_axis(
    n: int,
    plotting: Literal["weibull", "gringorten"] = "weibull",
    order: Literal["ascending", "descending"] = "descending",
) -> np.ndarray:
    """Plotting-position return-period axis for n ranked samples."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    m = np.arange(1, n + 1, dtype=float)
    if plotting == "weibull":
        T = (n + 1.0) / m
    elif plotting == "gringorten":
        T = (n - 0.12) / (m - 0.44)
    else:
        raise ValueError("plotting must be 'weibull' or 'gringorten'.")
    if order == "ascending":
        T = T[::-1]
    elif order != "descending":
        raise ValueError("order must be 'ascending' or 'descending'.")
    return T


# --- 0d: NetCDF validation + time helpers (from Xarray_NCtools.py) ----------

_TIME_CANDIDATES: List[str] = ["timemax", "time", "Time", "t", "datetime", "date"]


def _detect_time_coord(
    ds: xr.Dataset, preferred: Optional[str] = None
) -> Optional[str]:
    if preferred and preferred in ds.coords:
        return preferred
    for name in _TIME_CANDIDATES:
        if name in ds.coords:
            return name
    return None


def check_nc_file(
    path,
    required_vars=None,
    required_coords=None,
    check_time=True,
    sample_data=False,
):
    """Lightweight validation of a single SFINCS NetCDF file."""
    issues: List[str] = []
    summary = {"path": str(path)}
    try:
        ds = xr.open_dataset(
            path, decode_cf=True, mask_and_scale=True, engine="netcdf4"
        )
    except Exception as e:
        issues.append(f"OPEN_ERROR: {type(e).__name__}: {e}")
        return False, issues, summary

    summary["dims"] = dict(ds.sizes)
    summary["coords"] = list(ds.coords)
    summary["vars"] = list(ds.data_vars)
    summary["dtypes"] = {v: str(ds[v].dtype) for v in ds.variables}

    for d, n in ds.sizes.items():
        if n <= 0:
            issues.append(f"DIM_EMPTY: {d}={n}")
    for v in required_vars or []:
        if v not in ds.variables:
            issues.append(f"MISSING_VAR: {v}")
    for c in required_coords or []:
        if c not in ds.coords:
            issues.append(f"MISSING_COORD: {c}")

    if check_time:
        tname = _detect_time_coord(ds)
        if tname is not None:
            try:
                t = ds[tname].values
                if np.issubdtype(t.dtype, np.datetime64):
                    if np.any(np.diff(t) < np.timedelta64(0, "ns")):
                        issues.append(f"TIME_NON_MONOTONIC: {tname}")
                else:
                    if np.any(np.diff(t) < 0):
                        issues.append(f"TIME_NON_MONOTONIC: {tname}")
            except Exception as e:
                issues.append(f"TIME_READ_ERROR: {type(e).__name__}: {e}")

    if sample_data:
        try:
            for v in list(ds.data_vars)[:5]:
                slc = {d: 0 for d in ds[v].dims}
                _ = ds[v].isel(**slc).values
        except Exception as e:
            issues.append(f"DATA_SAMPLE_ERROR: {type(e).__name__}: {e}")

    ds.close()
    return len(issues) == 0, issues, summary


def batch_check_nc_files(
    files,
    required_vars=None,
    required_coords=None,
    check_time=True,
    sample_data=False,
):
    """Run check_nc_file across many files; return (good_files, report)."""
    report = []
    good_files = []
    for p in files:
        ok, issues, _ = check_nc_file(
            p,
            required_vars=required_vars,
            required_coords=required_coords,
            check_time=check_time,
            sample_data=sample_data,
        )
        report.append({"path": str(p), "ok": ok, "issues": issues})
        if ok:
            good_files.append(p)
    return good_files, report


def ensure_unique_sorted_time(
    ds: xr.Dataset, time_name: Optional[str] = None, keep: str = "first"
) -> xr.Dataset:
    """Drop duplicate timestamps and sort along the time coord."""
    tname = _detect_time_coord(ds, preferred=time_name)
    if tname is None:
        return ds
    pdt = pd.to_datetime(ds[tname].values)
    dup = pd.Series(pdt).duplicated(keep=keep).to_numpy()
    if dup.any():
        ds = ds.isel({tname: ~dup})
    return ds.sortby(tname)


# --- 0e: Water-year mode trimming (from part1.preprocess_) ------------------


def trim_to_mode_year(
    ds: xr.Dataset,
    time_name: Optional[str] = None,
    tie_break: Literal["earliest", "latest"] = "earliest",
) -> xr.Dataset:
    """Keep only timestamps falling in the most common calendar year."""
    tname = _detect_time_coord(ds, preferred=time_name)
    if tname is None:
        return ds
    if tname not in ds.coords:
        try:
            ds = ds.set_coords(tname)
        except Exception:
            return ds

    years = ds[tname].dt.year
    yvals = years.values
    if yvals.size == 0:
        return ds
    uniq, occ = np.unique(yvals, return_counts=True)
    candidates = uniq[occ == occ.max()]
    mode_year = int(candidates.max() if tie_break == "latest" else candidates.min())
    return ds.where(years == mode_year, drop=True)


# =============================================================================
# SECTION 1: Water-year aggregation
# =============================================================================

# def _open_sfincs_map(
#     sfincs_dir: Path,
#     vars_keep: Sequence[str],
#     time_dim: str,
# ) -> xr.Dataset:
#     """Open one SFINCS run's sfincs_map.nc, trim to its mode year, keep vars."""
#     nc_path = Path(sfincs_dir) / "sfincs_map.nc"
#     ds = xr.open_dataset(nc_path, decode_cf=True, mask_and_scale=True, engine="netcdf4")
#     keep = [v for v in vars_keep if v in ds]
#     if keep:
#         ds = ds[keep]
#     ds = trim_to_mode_year(ds, time_name=time_dim)
#     ds = ensure_unique_sorted_time(ds, time_name=time_dim)
#     return ds


def _open_sfincs_map(
    sfincs_dir: Path,
    vars_keep: Sequence[str],
    time_dim: str,
) -> xr.Dataset:
    """
    Open one SFINCS run's sfincs_map.nc, sanitize/convert the time coordinate,
    trim to its mode year, and keep the requested variables.
    """

    nc_path = Path(sfincs_dir) / "sfincs_map.nc"

    # 1) Open WITHOUT time decoding to avoid Overflow/OutOfBounds during CF decoding
    ds = xr.open_dataset(
        nc_path,
        decode_cf=True,  # still decode CF for other vars
        mask_and_scale=True,  # respect scales for data vars
        engine="netcdf4",
        decode_times=False,  # IMPORTANT: prevent immediate time decoding
    )

    # 2) Sanitize and decode the time coordinate named by `time_dim`
    if time_dim in ds:
        t = ds[time_dim]

        # --- 2a) Build a mask for bad/fill values ---------------------------------
        # Common NetCDF float32 fill value used by several tools:
        COMMON_FILL = np.float32(9.96921e36)

        # Read declared fill/missing markers if present
        declared_fill = t.attrs.get("_FillValue", t.attrs.get("missing_value", None))

        # Work with float for safety; ints can overflow on comparators
        t_float = t.astype("float64")

        # Define a generous threshold: 1e12 seconds ≈ 31,688 years
        # Adjust if you want tighter bounds
        max_seconds = 1e12

        mask = ~np.isfinite(t_float)
        mask |= t_float == COMMON_FILL
        if declared_fill is not None:
            # If declared_fill is an array, compare elementwise; if scalar, broadcast
            mask |= t_float == np.asarray(declared_fill, dtype="float64")

        # Mask extreme magnitudes (likely corrupted/fill)
        mask |= (t_float > max_seconds) | (t_float < -max_seconds)

        # Apply mask; set bad entries to NaN so we can decode safely
        t_clean = xr.where(~mask, t_float, np.nan)

        # --- 2b) Decode to datetime -----------------------------------------------
        units = t.attrs.get("units", None)
        calendar = t.attrs.get("calendar", "standard")  # default

        def _parse_cf_units(units_str: str) -> tuple[str, np.timedelta64]:
            """
            Return (origin_iso, step_dt64) where step_dt64 is one of
            'timedelta64[s]', '[m]', '[h]', '[D]', etc.
            """
            if not units_str or "since" not in units_str:
                return "1970-01-01T00:00:00", np.timedelta64(1, 's')
            base, ref = units_str.split("since", 1)
            base = base.strip().lower()
            ref = ref.strip()
            if " " in ref and "T" not in ref:
                ref = ref.replace(" ", "T")
            # Map CF time units to numpy timedelta64 units
            if   "nanosecond" in base or base == "ns":
                step = np.timedelta64(1, 'ns')
            elif "microsecond" in base or base in ("us", "µs"):
                step = np.timedelta64(1, 'us')
            elif "millisecond" in base or base == "ms":
                step = np.timedelta64(1, 'ms')
            elif "second" in base or base == "s" or base.startswith("sec"):
                step = np.timedelta64(1, 's')
            elif "minute" in base or base == "m" or base.startswith("min"):
                step = np.timedelta64(1, 'm')
            elif "hour"   in base or base == "h":
                step = np.timedelta64(1, 'h')
            elif "day"    in base or base in ("d", "D"):
                step = np.timedelta64(1, 'D')
            else:
                # Conservative fallback
                step = np.timedelta64(1, 's')
            return ref, step

        origin_iso, step = _parse_cf_units(units)

        try:
            origin = np.datetime64(origin_iso)
        except Exception:
            origin = np.datetime64("1970-01-01T00:00:00")

        # Convert cleaned numeric values to timedeltas in the correct unit
        if calendar in ("standard", "gregorian", "proleptic_gregorian", None):
            # numpy datetime64 path
            # NOTE: if step is 'D' and values are float, cast to 'timedelta64[ns]' via seconds
            #       to avoid loss for sub-day units. We scale explicitly:
            # Determine multiplier to nanoseconds
            unit_scale = {
                'ns': 1, 'us': 1_000, 'ms': 1_000_000, 's': 1_000_000_000,
                'm': 60 * 1_000_000_000, 'h': 3600 * 1_000_000_000, 'D': 86_400 * 1_000_000_000
            }
            step_unit = str(step).split('[')[-1].rstrip(']')
            scale_ns = unit_scale.get(step_unit, 1_000_000_000)  # default seconds
            # t_clean is float seconds in *step units*, so multiply to ns
            td_ns = (t_clean * scale_ns).astype("timedelta64[ns]")
            t_dt = origin + td_ns
        else:
            # non-standard calendar via cftime
            import cftime
            t_dt = xr.apply_ufunc(
                lambda v: np.array(cftime.num2date(v.astype("float64"), units, calendar,
                                                only_use_cftime_datetimes=True)),
                t_clean,
                vectorize=True,
                dask="parallelized",
                output_dtypes=[object],
            )


        # --- 2c) Attach decoded coord, tidy attrs ---------------------------------
        ds = ds.assign_coords({time_dim: t_dt})
        for k in ("units", "calendar"):
            ds[time_dim].attrs.pop(k, None)
        ds[time_dim].attrs["standard_name"] = "time"
        ds[time_dim].attrs["long_name"] = "time"

        # --- 2d) Drop rows where time is invalid (NaT/null) -----------------------
        # For numpy datetime64, np.isnat works; for cftime, use pandas null check
        try:
            if np.issubdtype(ds[time_dim].dtype, np.datetime64):
                valid = ~np.isnat(ds[time_dim].values)
            else:
                # cftime/object dtype
                import pandas as pd

                valid = ~pd.isnull(ds[time_dim].values)
            if valid.ndim == 1 and valid.size == ds.sizes[time_dim]:
                ds = ds.sel({time_dim: valid})
        except Exception:
            # If anything goes wrong, keep the dataset; downstream functions may handle
            pass

    # 3) Keep requested variables (xarray will retain needed coords)
    keep = [v for v in vars_keep if v in ds]
    if keep:
        ds = ds[keep]

    # 4) Your existing pipeline: trim to mode year & ensure unique/sorted time
    ds = trim_to_mode_year(ds, time_name=time_dim)
    ds = ensure_unique_sorted_time(ds, time_name=time_dim)

    return ds


def aggregate_water_year_maxima(
    sfincs_dirs: Sequence[Path],
    extra_vars: Sequence[str] = (),
    time_dim: str = "timemax",
    face_dim: str = "nmesh2d_face",
    output_fn: Optional[Path] = None,
    keep_timeseries: bool = False,
    aggregation_mode: Literal["annual_max", "all_maxima"] = "annual_max",
) -> xr.Dataset:
    """Aggregate SFINCS NetCDF outputs across N water years.

    Two modes:

    - ``"annual_max"`` (default): for each year and each cell, pick the
      single argmax along ``time_dim`` and emit `zsmax_ann` + `<var>_at_peak`
      with dims ``(year, face_dim)``. Cells that have zero valid timesteps
      become NaN (never crash).
    - ``"all_maxima"``: skip the argmax step entirely. Each year's
      ``sfincs_map.nc`` is trimmed to its mode year, then concatenated along
      ``time_dim``. Output dims are ``(time_dim, face_dim)`` with the
      original variable names (``zsmax``, ``qmax``, ...). Suitable for POT
      EVA; incompatible with Weibull / GEV.

    Parameters
    ----------
    sfincs_dirs
        One directory per water year; each must contain ``sfincs_map.nc``.
    extra_vars
        Additional variables (e.g. ``qmax``, ``tmax``) to extract.
    keep_timeseries
        Only meaningful in ``"annual_max"`` mode. If True, persist the full
        concatenated series to a sidecar NC (`<output_fn>.timeseries.nc`),
        needed when POT EVA is run on annual-max output. Ignored in
        ``"all_maxima"`` mode (the main file already is the timeseries).
    output_fn
        Optional NetCDF persistence path. If the file exists, it is opened
        and returned without recomputing.

    Returns
    -------
    xr.Dataset with `attrs["aggregation_mode"]` recording the mode and
    `attrs["n_years"]` recording the input count (used by POT EVA).
    """
    if output_fn is not None and Path(output_fn).exists():
        return xr.open_dataset(output_fn)

    vars_keep = ["zsmax", *extra_vars]
    yearly_datasets: List[xr.Dataset] = []
    full_series: List[xr.Dataset] = []

    SENTINEL = np.float32(-1.0e30)

    for sd in sfincs_dirs:
        sd = Path(sd)
        print(f"  aggregating: {sd}")
        ds = _open_sfincs_map(sd, vars_keep, time_dim)
        if "zsmax" not in ds:
            raise RuntimeError(f"`zsmax` missing in {sd / 'sfincs_map.nc'}")

        if aggregation_mode == "all_maxima":
            # Keep all sub-yearly timemax slices; no per-cell argmax.
            full_series.append(ds.load())
            ds.close()
            continue

        if keep_timeseries:
            full_series.append(ds.copy())

        zs = ds["zsmax"]
        zs_valid = zs.where(np.isfinite(zs) & (zs > -100))
        all_nan = zs_valid.isnull().all(dim=time_dim)
        # Sentinel-fill keeps argmax safe for all-NaN cells (permanently
        # dry quadtree faces); the result is masked back to NaN below.
        i_peak = zs_valid.fillna(SENTINEL).argmax(dim=time_dim, skipna=False)
        zsmax_ann = zs_valid.isel({time_dim: i_peak}).where(~all_nan)
        t_peak = ds[time_dim].isel({time_dim: i_peak})

        year_label = infer_water_year_from_time_array(ds[time_dim].values)

        out = xr.Dataset(
            {"zsmax_ann": zsmax_ann.drop_vars(time_dim, errors="ignore")},
            coords={"t_peak": t_peak.drop_vars(time_dim, errors="ignore")},
        )
        for v in extra_vars:
            if v in ds:
                vv = ds[v].isel({time_dim: i_peak}).where(~all_nan)
                out[f"{v}_at_peak"] = vv.drop_vars(time_dim, errors="ignore")
        out = out.expand_dims(year=[year_label])
        yearly_datasets.append(out)
        ds.close()

    n_years = len(sfincs_dirs)

    if aggregation_mode == "all_maxima":
        if not full_series:
            raise RuntimeError("No SFINCS runs were aggregated.")
        ds_agg = xr.concat(full_series, dim=time_dim).sortby(time_dim)
        ds_agg = ensure_unique_sorted_time(ds_agg, time_name=time_dim)
        ds_agg.attrs["aggregation_mode"] = "all_maxima"
        ds_agg.attrs["n_years"] = n_years
        if output_fn is not None:
            Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
            ds_agg.to_netcdf(output_fn)
        return ds_agg

    ds_agg = xr.concat(yearly_datasets, dim="year").sortby("year")
    ds_agg.attrs["aggregation_mode"] = "annual_max"
    ds_agg.attrs["n_years"] = n_years

    if output_fn is not None:
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        ds_agg.to_netcdf(output_fn)

    if keep_timeseries and output_fn is not None and full_series:
        sidecar = Path(output_fn).with_suffix(".timeseries.nc")
        ds_full = xr.concat(full_series, dim=time_dim).sortby(time_dim)
        ds_full = ensure_unique_sorted_time(ds_full, time_name=time_dim)
        ds_full.to_netcdf(sidecar)
        ds_agg.attrs["_full_timeseries_fn"] = str(sidecar)

    return ds_agg


# =============================================================================
# SECTION 2: Extreme value analysis (Weibull / GEV / POT)
# =============================================================================


def eva_weibull(
    da_annual_max: xr.DataArray,
    return_periods: Sequence[float],
    year_dim: str = "year",
) -> xr.DataArray:
    """Empirical Weibull plotting-position EVA per cell.

    Sort the n annual maxima descending; assign T_k = (n+1)/k; linearly
    interpolate the requested return periods in log-T space. Cells with
    fewer than 2 finite years receive NaN.
    """
    rp_target = np.asarray(return_periods, dtype=float)

    def _per_cell(maxes: np.ndarray) -> np.ndarray:
        finite = maxes[np.isfinite(maxes)]
        n = finite.size
        if n < 2:
            return np.full(rp_target.size, np.nan, dtype=np.float64)
        sorted_desc = np.sort(finite)[::-1]
        T_emp = (n + 1.0) / np.arange(1, n + 1, dtype=float)
        log_T_emp = np.log(T_emp[::-1])
        vals_for_T = sorted_desc[::-1]
        return np.interp(
            np.log(rp_target),
            log_T_emp,
            vals_for_T,
            left=vals_for_T[0],
            right=vals_for_T[-1],
        )

    out = xr.apply_ufunc(
        _per_cell,
        da_annual_max,
        input_core_dims=[[year_dim]],
        output_core_dims=[["rp"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"rp": rp_target.size}},
    )
    out = out.assign_coords(rp=("rp", rp_target))
    out.name = "zsmax_rp"
    return out


def eva_gev(
    da_annual_max: xr.DataArray,
    return_periods: Sequence[float],
    year_dim: str = "year",
    min_years: int = 5,
) -> xr.DataArray:
    """Per-cell GEV fit -> non-exceedance quantiles at the target RPs."""
    rp_target = np.asarray(return_periods, dtype=float)
    p = 1.0 - 1.0 / rp_target

    def _per_cell(maxes: np.ndarray) -> np.ndarray:
        finite = maxes[np.isfinite(maxes)]
        if finite.size < min_years:
            return np.full(rp_target.size, np.nan, dtype=np.float64)
        try:
            shape, loc, scale = genextreme.fit(finite)
            return genextreme.ppf(p, shape, loc=loc, scale=scale)
        except Exception:
            return np.full(rp_target.size, np.nan, dtype=np.float64)

    out = xr.apply_ufunc(
        _per_cell,
        da_annual_max,
        input_core_dims=[[year_dim]],
        output_core_dims=[["rp"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"rp": rp_target.size}},
    )
    out = out.assign_coords(rp=("rp", rp_target))
    out.name = "zsmax_rp"
    return out


def eva_pot(
    da_full_timeseries: xr.DataArray,
    return_periods: Sequence[float],
    n_years: int,
    target_per_year: int = 5,
    decluster_window: str = "72h",
    time_dim: str = "timemax",
    face_dim: str = "nmesh2d_face",
) -> xr.DataArray:
    """Per-cell POT extraction + log-T interpolation to target RPs.

    Selects a threshold yielding ~`target_per_year * n_years` declustered
    peaks, then assigns empirical return periods T_k = n_years*(k_peaks+1)/k
    and interpolates in log-T space to `return_periods`.
    """
    rp_target = np.asarray(return_periods, dtype=float)
    target_total = max(1, int(target_per_year * n_years))

    faces = da_full_timeseries[face_dim].values
    out_arr = np.full((faces.size, rp_target.size), np.nan, dtype=np.float64)

    for i, face in enumerate(faces):
        series = da_full_timeseries.sel({face_dim: face})
        vals = series.values
        finite = vals[np.isfinite(vals) & (vals > -100)]
        if finite.size < 5:
            continue
        try:
            th = pot_threshold_set_num_xr(
                series,
                r=decluster_window,
                num_exce=target_total,
                time_dim=time_dim,
                strategy="closest",
            )
            extremes = get_extremes_pot_xr(
                series,
                th,
                r=decluster_window,
                time_dim=time_dim,
                num_exce=target_total,
            )
        except Exception:
            continue
        v = np.sort(extremes.values)[::-1]
        k = v.size
        if k < 2:
            continue
        T_emp = n_years * (k + 1.0) / np.arange(1, k + 1, dtype=float)
        log_T_emp = np.log(T_emp[::-1])
        vals_for_T = v[::-1]
        out_arr[i] = np.interp(
            np.log(rp_target),
            log_T_emp,
            vals_for_T,
            left=vals_for_T[0],
            right=vals_for_T[-1],
        )

    return xr.DataArray(
        out_arr,
        dims=(face_dim, "rp"),
        coords={face_dim: faces, "rp": rp_target},
        name="zsmax_rp",
    )



def eva_apply(
    ds_annual: xr.Dataset,
    method: Literal["weibull", "gev", "pot"],
    return_periods: Sequence[float],
    ds_full_timeseries: Optional[xr.Dataset] = None,
    face_dim: str = "nmesh2d_face",
    time_dim: str = "timemax",
    pot_target_per_year: int = 5,
    pot_decluster: str = "72h",
    gev_min_years: int = 5,
    *,
    z_floor: Optional[np.ndarray] = None,
) -> xr.Dataset:
    """
    Extended EVA dispatcher with optional dry-floor masking.
    If `z_floor` is provided (shape == n_faces), any RP WSE ≤ z_floor + 1e-4
    is set to NaN.
    """
    method = method.lower()
    agg_mode = ds_annual.attrs.get("aggregation_mode", "annual_max")

    if agg_mode == "all_maxima":
        if method != "pot":
            raise ValueError(
                f"aggregation_mode='all_maxima' is incompatible with "
                f"eva_method={method!r}: no annual maxima are available. "
                "Use 'annual_max' for Weibull/GEV, or switch eva_method to 'pot'."
            )
        if "zsmax" not in ds_annual:
            raise ValueError("`zsmax` missing in all-maxima dataset.")
        n_years = int(ds_annual.attrs.get("n_years", 0))
        if n_years < 2:
            raise ValueError(
                f"POT EVA requires at least 2 water years; got n_years={n_years}."
            )
        zsmax_rp = eva_pot(
            ds_annual["zsmax"],
            return_periods,
            n_years=n_years,
            target_per_year=pot_target_per_year,
            decluster_window=pot_decluster,
            time_dim=time_dim,
            face_dim=face_dim,
        )
    elif method == "weibull":
        zsmax_rp = eva_weibull(ds_annual["zsmax_ann"], return_periods)
    elif method == "gev":
        zsmax_rp = eva_gev(
            ds_annual["zsmax_ann"], return_periods, min_years=gev_min_years
        )
    elif method == "pot":
        if ds_full_timeseries is None or "zsmax" not in ds_full_timeseries:
            raise ValueError("POT EVA requires `ds_full_timeseries` with `zsmax`.")
        n_years = int(ds_annual.attrs.get("n_years", ds_annual.sizes.get("year", 0)))
        if n_years < 2:
            raise ValueError("POT EVA requires at least 2 water years of data.")
        zsmax_rp = eva_pot(
            ds_full_timeseries["zsmax"],
            return_periods,
            n_years=n_years,
            target_per_year=pot_target_per_year,
            decluster_window=pot_decluster,
            time_dim=time_dim,
            face_dim=face_dim,
        )
    else:
        raise ValueError(f"Unknown EVA method {method!r}; use weibull/gev/pot.")

    out = xr.Dataset({"zsmax_rp": zsmax_rp})
    out.attrs["eva_method"] = method
    out.attrs["aggregation_mode"] = agg_mode

    # Optional dry-floor masking (z_zmin)
    if z_floor is not None:
        # Align floor along face_dim
        if z_floor.ndim != 1:
            raise ValueError("z_floor must be a 1D array of length n_faces.")
        floor = xr.DataArray(
            z_floor.astype(np.float64),
            dims=(face_dim,),
            coords={face_dim: out["zsmax_rp"][face_dim]},
        )
        masked = xr.where(out["zsmax_rp"] > floor + 1e-4, out["zsmax_rp"], np.nan)
        out["zsmax_rp"] = masked

    return out



# =============================================================================
# SECTION 3: Event-matched extras + quadtree-to-DEM mapping
# =============================================================================


def extras_at_rp_via_rank(
    ds_annual: xr.Dataset,
    return_periods: Sequence[float],
    extras: Sequence[str],
    year_dim: str = "year",
) -> xr.Dataset:
    """For Weibull/GEV: pick each extra at the year contributing the
    rank-matched annual max. Empirical Weibull rank k = (n+1)/RP, rounded
    to the nearest integer and clamped to [1, n]."""
    if not extras:
        return xr.Dataset()

    rp_target = np.asarray(return_periods, dtype=float)
    n_rp = rp_target.size

    zs = ds_annual["zsmax_ann"]
    zs_t = zs.transpose(..., year_dim)
    n_years = zs_t.sizes[year_dim]
    k_real = np.clip((n_years + 1.0) / rp_target, 1.0, n_years)
    k_int = np.round(k_real).astype(int)

    out = xr.Dataset()
    for v in extras:
        key = f"{v}_at_peak"
        if key not in ds_annual:
            continue
        da_extra = ds_annual[key].transpose(..., year_dim)
        zs_vals = zs_t.values
        ex_vals = da_extra.values
        order = np.argsort(-zs_vals, axis=-1, kind="stable")
        ex_sorted = np.take_along_axis(ex_vals, order, axis=-1)
        rank_cols = (k_int - 1).reshape((1,) * (ex_sorted.ndim - 1) + (n_rp,))
        rank_cols_b = np.broadcast_to(rank_cols, ex_sorted.shape[:-1] + (n_rp,))
        picked = np.take_along_axis(ex_sorted, rank_cols_b, axis=-1)
        non_year_dims = [d for d in da_extra.dims if d != year_dim]
        coords = {d: da_extra[d] for d in non_year_dims}
        coords["rp"] = rp_target
        out[f"{v}_rp"] = xr.DataArray(
            picked,
            dims=tuple(non_year_dims) + ("rp",),
            coords=coords,
            name=f"{v}_rp",
        )
    return out


def map_quadtree_to_dem_nearest(
    da_face: xr.DataArray,
    indices_fn: Path,
    out_fn: Path,
    depth_mask_fn: Optional[Path] = None,
    hmin: float = 0.02,
) -> None:
    """Nearest-neighbour map a per-face DataArray onto the DEM grid via an
    index COG. Optionally mask out pixels where depth <= hmin."""
    indices_fn = Path(indices_fn)
    out_fn = Path(out_fn)
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    values = da_face.values.astype(np.float32)

    with rasterio.open(indices_fn) as idx_src:
        idx_nodata_val = idx_src.nodata
        idx_nodata = int(idx_nodata_val) if idx_nodata_val is not None else -1
        meta = idx_src.meta.copy()

    meta.update(
        count=1,
        dtype="float32",
        nodata=float("nan"),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=2,
        BIGTIFF="YES",
    )

    if depth_mask_fn is not None:
        depth_mask_fn = Path(depth_mask_fn)
    with (
        rasterio.open(out_fn, "w", **meta) as dst,
        rasterio.open(indices_fn) as idx_src,
    ):
        dep_src = rasterio.open(depth_mask_fn) if depth_mask_fn is not None else None
        try:
            for _, window in dst.block_windows(1):
                idx_block = idx_src.read(1, window=window)
                if dep_src is not None:
                    dep_block = dep_src.read(1, window=window)
                    wet = np.isfinite(dep_block) & (dep_block > hmin)
                else:
                    wet = np.ones_like(idx_block, dtype=bool)
                valid = (idx_block != idx_nodata) & wet if idx_nodata_val is not None else wet & np.isfinite(idx_block)
                out_block = np.full(idx_block.shape, np.nan, dtype=np.float32)
                if np.any(valid):
                    out_block[valid] = values[idx_block[valid].astype(np.intp)]
                dst.write(out_block, 1, window=window)
        finally:
            if dep_src is not None:
                dep_src.close()


def load_clip_polygon(shapefile_path):
    """
    Load a shapefile and return it as a GeoDataFrame.

    Parameters
    ----------
    shapefile_path : str or path-like
        Path to the shapefile (.shp) or directory containing it.

    Returns
    -------
    geopandas.GeoDataFrame
        The loaded shapefile as a GeoDataFrame.
    """
    return gpd.read_file(shapefile_path)


def smooth_raster_gaussian_blockwise(
    in_fn: Path,
    out_fn: Path,
    smooth_size: float,
    truncate: Optional[float] = None,
    *,
    preserve_input_nodata: bool = False,
    out_dtype: str = "float32",
    out_blocksize: Optional[
        int
    ] = None,  # e.g., 256 or 512; if None, derive from source or fallback
    mode: str = "reflect",  # scipy.ndimage gaussian_filter boundary mode
) -> None:
    """
    Blockwise NaN/nodata‑preserving Gaussian smoothing with halo padding.

    The filter is applied in a streaming, tiled fashion. Each output tile is read
    with a surrounding halo from the input to avoid seam artifacts. Invalid values
    (NaN and source nodata) are excluded via weighted convolution (V/W).

    Parameters
    ----------
    in_fn : Path
        Path to readable raster (single band float ideally).
    out_fn : Path
        Path to write the smoothed raster (must be different from in_fn).
    smooth_size : float
        Gaussian sigma (pixels).
    truncate : float, optional
        Radius multiplier for the Gaussian kernel. The effective radius is:
        radius = truncate * smooth_size. If None, uses 2.0 (common). SciPy's
        default is 4.0.
    preserve_input_nodata : bool, default False
        If True, output nodata metadata matches the source nodata (if any).
        If False, output nodata metadata is omitted (None) and actual NaNs in
        data represent missing values.
    out_dtype : str, default "float32"
        Output dtype.
    out_blocksize : int, optional
        Output tile size (square). If None, derive from source tiling; else use
        provided (typical: 256 or 512).
    mode : {"reflect","nearest","mirror","constant","wrap"}, default "reflect"
        Boundary handling for gaussian_filter.

    Notes
    -----
    • Halo size = ceil(truncate * sigma)
    • Weighted smoothing ignores NaNs and nodata:
        V = data with invalid set to 0
        W = 1 for valid, 0 for invalid
        out = gaussian(V) / gaussian(W), for W > eps
    • Output is tiled GeoTIFF with DEFLATE compression and floating predictor.

    Raises
    ------
    ValueError
        If smooth_size < 0, or if out_fn == in_fn (in‑place writing removed).
    """
    in_fn = Path(in_fn)
    out_fn = Path(out_fn)

    if in_fn.resolve() == out_fn.resolve():
        raise ValueError(
            "out_fn must be different from in_fn; in‑place writing has been removed."
        )

    if smooth_size < 0:
        raise ValueError("smooth_size (sigma) must be >= 0")
    if truncate is None:
        truncate = 2.0  # common choice; SciPy default is 4.0

    out_np_dtype = np.dtype(out_dtype)

    # Short‑circuit: sigma==0 → copy (with nodata update policy, but no in‑place)
    if smooth_size == 0:
        out_fn.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(in_fn) as src:
            if src.count != 1:
                raise ValueError(
                    f"Expected a single-band raster; found {src.count} bands."
                )
            meta = src.meta.copy()

            # Decide nodata metadata policy
            src_nodata = src.nodata
            out_nodata_tag = src_nodata if preserve_input_nodata else None

            meta.update(dtype=out_dtype, nodata=out_nodata_tag)

            band = src.read(1).astype(out_np_dtype, copy=False)

            # If not preserving nodata and source had a finite nodata, convert those cells to NaN
            if (
                not preserve_input_nodata
                and src_nodata is not None
                and not (isinstance(src_nodata, float) and np.isnan(src_nodata))
            ):
                band = band.copy()
                band[band == src_nodata] = np.nan

            with rasterio.open(out_fn, "w", **meta) as dst:
                dst.write(band, 1)
                dst.update_tags(
                    smoothing="gaussian_blockwise",
                    smoothing_sigma=str(smooth_size),
                    smoothing_truncate=str(truncate),
                    smoothing_halo=str(0),
                    input_nodata=str(src_nodata),
                    output_nodata=(
                        "None" if out_nodata_tag is None else str(out_nodata_tag)
                    ),
                    note="sigma=0 → copy",
                    mode=mode,
                )
        return

    # Regular smoothing path
    halo = max(1, int(math.ceil(truncate * smooth_size)))
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_fn) as src:
        if src.count != 1:
            raise ValueError(f"Expected a single-band raster; found {src.count} bands.")

        height, width = src.height, src.width
        src_profile = src.profile.copy()
        src_nodata = src.nodata

        # Choose output block size
        if out_blocksize is not None:
            block_h = block_w = int(out_blocksize)
        else:
            # Try to inherit source tiling; fall back to profile; else 512
            try:
                block_h, block_w = src.block_shapes[0]  # (height, width)
            except Exception:
                block_w = int(src_profile.get("blockxsize", 512))
                block_h = int(src_profile.get("blockysize", 512))

        # Decide output nodata metadata tag (omit when using NaNs)
        out_nodata_tag = src_nodata if preserve_input_nodata else None

        # Build output profile
        dst_profile = src_profile.copy()
        dst_profile.update(
            driver="GTiff",
            dtype=out_dtype,
            count=1,
            nodata=out_nodata_tag,
            tiled=True,
            blockxsize=block_w,
            blockysize=block_h,
            compress="deflate",
            predictor=3,  # better for float data
            zlevel=6,  # reasonable compression level
            BIGTIFF="IF_NEEDED",
        )

        with rasterio.open(out_fn, "w", **dst_profile) as dst:
            # Iterate over the output's block windows for aligned writes
            for _, win in dst.block_windows(1):
                # Pad window coordinates in source space
                row_off = max(win.row_off - halo, 0)
                col_off = max(win.col_off - halo, 0)
                row_end = min(win.row_off + win.height + halo, height)
                col_end = min(win.col_off + win.width + halo, width)

                pad_win = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=col_end - col_off,
                    height=row_end - row_off,
                )

                pad_block = src.read(1, window=pad_win).astype(out_np_dtype, copy=False)

                # Build invalid mask (NaN or source nodata)
                if src_nodata is not None and not (
                    isinstance(src_nodata, float) and np.isnan(src_nodata)
                ):
                    ind_nan = np.isnan(pad_block) | (pad_block == src_nodata)
                else:
                    ind_nan = np.isnan(pad_block)

                # Weighted convolution: ignore invalids
                V = pad_block.copy()
                V[ind_nan] = 0.0

                W = np.ones_like(V, dtype=out_np_dtype)
                W[ind_nan] = 0.0

                VV = gaussian_filter(V, sigma=smooth_size, truncate=truncate, mode=mode)
                WW = gaussian_filter(W, sigma=smooth_size, truncate=truncate, mode=mode)

                out_pad = np.empty_like(VV, dtype=out_np_dtype)
                out_pad.fill(np.nan)

                # Numerical threshold: small but nonzero
                eps = np.finfo(WW.dtype).eps
                mask_valid_weight = WW > eps

                # Compute filtered values where we have any valid neighbor support
                out_pad[mask_valid_weight] = (
                    VV[mask_valid_weight] / WW[mask_valid_weight]
                )

                # Choose data sentinel: use src nodata if present and preserved; else NaN
                if (
                    preserve_input_nodata
                    and src_nodata is not None
                    and not (isinstance(src_nodata, float) and np.isnan(src_nodata))
                ):
                    out_nodata_data = src_nodata
                else:
                    out_nodata_data = np.nan

                # Preserve original invalids AND fill zero-weight cells
                out_pad[ind_nan] = out_nodata_data
                out_pad[~mask_valid_weight] = out_nodata_data

                # Crop back to the exact output tile
                row0 = win.row_off - row_off
                row1 = row0 + win.height
                col0 = win.col_off - col_off
                col1 = col0 + win.width

                out_block = out_pad[row0:row1, col0:col1]
                dst.write(out_block, 1, window=win)

            # Tags / provenance
            dst.update_tags(
                smoothing="gaussian_blockwise",
                smoothing_sigma=str(smooth_size),
                smoothing_truncate=str(truncate),
                smoothing_halo=str(halo),
                input_nodata=str(src_nodata),
                output_nodata=(
                    "None" if out_nodata_tag is None else str(out_nodata_tag)
                ),
                note="Blockwise nodata/NaN‑preserving Gaussian smoothing with halo",
                blockxsize=str(block_w),
                blockysize=str(block_h),
                mode=mode,
            )


# =============================================================================
# SECTION 4: Hazard binning + I/O
# =============================================================================


def stamp_provenance(fn: Path, **tags) -> None:
    """Merge `tags` into the GeoTIFF metadata of `fn` (open in r+ mode).

    Values are coerced to `str`. Existing tags (e.g. those `bin_raster`
    writes) are preserved; only the keys passed here are added/overwritten.
    """
    fn = Path(fn)
    if not fn.exists():
        return
    str_tags = {k: str(v) for k, v in tags.items() if v is not None}
    if not str_tags:
        return
    with rasterio.open(fn, "r+") as dst:
        dst.update_tags(**str_tags)


def bin_raster(
    in_fn: Path,
    bins_dict: Mapping[str, object],
    out_fn: Path,
) -> None:
    """
    Block-wise hazard-category binning of a float raster -> uint8 GeoTIFF.

    Classification uses np.digitize(..., right=False) with edges = VD_Min:
      - 1..N : bin index based on lower-bound edges: [VD_Min[i], VD_Min[i+1]); last bin -> [VD_Min[N-1], +inf)
      - 255  : nodata (non-finite source pixel, explicit src.nodata sentinel if present,
                       or value below the first lower edge)

    Notes
    -----
    - Only VD_Min is used as binning edges; VD_Max is validated and written to tags, but not used to truncate bins.
    - Output is uint8, single-band, tiled, deflate-compressed.
    - Tiling reuses the source block sizes when available; otherwise defaults to 512x512.
    - Bins whose ID == 0 are treated as NoData and written as 255.
    """

    in_fn = Path(in_fn)
    out_fn = Path(out_fn)
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    # --- Extract and validate bins_dict ---
    required = ("ID", "Category", "VD_Label", "VD_Min", "VD_Max")
    missing = [k for k in required if k not in bins_dict]
    if missing:
        raise ValueError(f"bins_dict missing required keys: {missing}")

    ids = np.asarray(bins_dict["ID"], dtype=np.int64)
    cats = list(bins_dict["Category"])
    labels = list(bins_dict["VD_Label"])
    vmin = np.asarray(bins_dict["VD_Min"], dtype=np.float32)
    vmax = np.asarray(bins_dict["VD_Max"], dtype=np.float32)

    n = vmin.size
    if not (len(ids) == len(cats) == len(labels) == vmin.size == vmax.size):
        raise ValueError(
            "All bins_dict arrays/lists must have the same length: "
            f"ID={len(ids)}, Category={len(cats)}, VD_Label={len(labels)}, "
            f"VD_Min={vmin.size}, VD_Max={vmax.size}"
        )

    if n > 1 and not np.all(np.diff(vmin) > 0):
        raise ValueError("bins_dict['VD_Min'] must be strictly increasing.")
    if not np.all(vmin <= vmax):
        raise ValueError("Each bin must satisfy VD_Min <= VD_Max.")

    # Lower bounds as edges for np.digitize
    edges = vmin

    with rasterio.open(in_fn) as src:
        # Enforce single-band input
        if src.count != 1:
            raise ValueError(f"Expected single-band input; got src.count={src.count}")

        src_nodata = src.nodata  # may be None

        # Reuse source tiling if available; else default 512x512
        src_block_x = src.profile.get("blockxsize")
        src_block_y = src.profile.get("blockysize")
        blockx = int(src_block_x) if src_block_x else 512
        blocky = int(src_block_y) if src_block_y else 512

        meta = src.meta.copy()
        meta.update(
            dtype="uint8",
            nodata=255,
            count=1,
            tiled=True,
            blockxsize=blockx,
            blockysize=blocky,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(out_fn, "w", **meta) as dst:
            for _, window in src.block_windows(1):
                block = src.read(1, window=window).astype(np.float32)

                # Classify: 0..N with lower-bound edges
                bins = np.digitize(block, edges, right=False).astype(np.uint8)
                bins = np.minimum(bins, n).astype(np.uint8)

                # Below first edge -> nodata
                bins[bins == 0] = 255

                # Non-finite and explicit finite src.nodata -> nodata
                nonfinite = ~np.isfinite(block)
                if src_nodata is not None:
                    nodata_mask = nonfinite | (block == src_nodata)
                else:
                    nodata_mask = nonfinite
                bins[nodata_mask] = 255

                # NEW: bins whose corresponding ID == 0 -> nodata
                if np.any(ids == 0):
                    zero_bin_indices = (
                        np.nonzero(ids == 0)[0] + 1
                    )  # bin indices are 1..N
                    bins[np.isin(bins, zero_bin_indices)] = 255

                dst.write(bins, 1, window=window)

            # --- Tags reflecting dictionary attributes ---
            def _fmt(v):
                return f"{v:g}" if np.isfinite(v) else ("-inf" if v < 0 else "inf")

            tags = {
                "bin_ids": ",".join(str(int(i)) for i in ids),
                "bin_category": ",".join(str(s) for s in cats),
                "bin_label": ",".join(str(s) for s in labels),
                "bin_min": ",".join(_fmt(float(v)) for v in vmin),
                "bin_max": ",".join(_fmt(float(v)) for v in vmax),
                "nodata_label": "nodata",
                "nodata_value": "255",
                "below_first_edge_is_nodata": "True",
                "binning_edges": "VD_Min",
                "binning_right": "False",
                "id_zero_is_nodata": "True",
            }
            for i in range(n):
                idx = i + 1
                tags[f"bin_{idx}_id"] = str(int(ids[i]))
                tags[f"bin_{idx}_category"] = str(cats[i])
                tags[f"bin_{idx}_label"] = str(labels[i])
                tags[f"bin_{idx}_min"] = _fmt(float(vmin[i]))
                tags[f"bin_{idx}_max"] = _fmt(float(vmax[i]))


def bin_depth_with_overlays(
    hmax_masked_fn: Path,
    connection_fn: Path,
    dem_fn: Path,
    depth_bins: Mapping[str, object],
    out_fn: Path,
    floodprone_label: str = "Flood-prone Low-Lying",
) -> None:
    """Categorical depth raster with flood-prone overlay (no MHHW).

    Code mapping (on the DEM grid):
    ===========  =====================================================
    Code         Source / meaning
    ===========  =====================================================
    1..N         depth bins (N = len(depth_bins["D_Min"]); code i
                 corresponds to bin i in depth_bins arrays)
    N+1          Flood-prone Low-Lying (`connection == 2`)
    255          dry / no flooding **and** nodata (non-finite DEM)
    ===========  =====================================================

    Notes:
    - 255 is both a valid category ("dry") and set as GeoTIFF nodata.
    - There is no 0-category. Any value below the first bin edge
      remains 255 (dry).
    """
    hmax_masked_fn = Path(hmax_masked_fn)
    connection_fn = Path(connection_fn)
    dem_fn = Path(dem_fn)
    out_fn = Path(out_fn)
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    # --- Extract and validate depth_bins ---
    required = ("ID", "Category", "Label_ft", "Label_m", "D_Min", "D_Max")
    missing = [k for k in required if k not in depth_bins]
    if missing:
        raise ValueError(f"depth_bins missing required keys: {missing}")

    ids = np.asarray(depth_bins["ID"])
    cats = list(depth_bins["Category"])
    lbl_ft = list(depth_bins["Label_ft"])
    lbl_m = list(depth_bins["Label_m"])
    dmin = np.asarray(depth_bins["D_Min"], dtype=np.float32)
    dmax = np.asarray(depth_bins["D_Max"], dtype=np.float32)

    n_depth_bins = dmin.size
    if not (
        len(ids) == len(cats) == len(lbl_ft) == len(lbl_m) == dmin.size == dmax.size
    ):
        raise ValueError(
            "All depth_bins arrays/lists must have the same length: "
            f"ID={len(ids)}, Category={len(cats)}, Label_ft={len(lbl_ft)}, "
            f"Label_m={len(lbl_m)}, D_Min={dmin.size}, D_Max={dmax.size}"
        )

    if n_depth_bins > 1 and not np.all(np.diff(dmin) > 0):
        raise ValueError("depth_bins['D_Min'] must be strictly increasing.")
    if not np.all(dmin <= dmax):
        raise ValueError("Each bin must satisfy D_Min <= D_Max.")

    # Codes: bins 1..N; flood-prone N+1; dry/nodata 255
    depth_code_offset = 1
    floodprone_code = n_depth_bins + depth_code_offset  # N+1
    if floodprone_code >= 255:
        raise ValueError(
            f"Too many depth bins ({n_depth_bins}); floodprone_code "
            f"would be {floodprone_code} >= 255 (reserved for dry/nodata)."
        )

    # Open sources and verify grid alignment
    with (
        rasterio.open(hmax_masked_fn) as hsrc,
        rasterio.open(connection_fn) as csrc,
        rasterio.open(dem_fn) as dsrc,
    ):

        def _same_grid(a, b) -> bool:
            return (
                a.width == b.width
                and a.height == b.height
                and a.transform == b.transform
                and a.crs == b.crs
            )

        if not _same_grid(hsrc, csrc):
            raise ValueError(
                "Grid mismatch: 'connection_fn' does not match 'hmax_masked_fn' (width/height/transform/crs)."
            )
        if not _same_grid(hsrc, dsrc):
            raise ValueError(
                "Grid mismatch: 'dem_fn' does not match 'hmax_masked_fn' (width/height/transform/crs)."
            )

        # Prepare output metadata from the hmax source
        meta = hsrc.meta.copy()
        meta.update(
            count=1,
            dtype="uint8",
            nodata=255,  # <-- nodata is 255 (same as 'dry')
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            BIGTIFF="YES",
        )

        with rasterio.open(out_fn, "w", **meta) as dst:
            for _, window in dst.block_windows(1):
                h = hsrc.read(1, window=window).astype(np.float32)
                c = csrc.read(1, window=window)
                d = dsrc.read(1, window=window).astype(np.float32)

                # Default is DRY (255), which is also nodata
                out = np.full(h.shape, 255, dtype=np.uint8)

                # non-finite DEM -> also 255 (nodata/dry)
                dem_nodata = ~np.isfinite(d)
                out[dem_nodata] = 255

                # depth bins where DEM is finite and depth > 0
                wet = (~dem_nodata) & np.isfinite(h) & (h > 0.0)
                if np.any(wet):
                    # Digitize by lower edges; 0..N where 0 = below first bin (stays 255 dry)
                    bins = np.digitize(h, dmin, right=False).astype(np.int16)
                    bins[~wet] = 0
                    bins = np.minimum(bins, n_depth_bins)  # clamp deepest to N

                    # ✅ Shift to 1..N correctly (or just use bins directly)
                    shifted = np.where(
                        bins >= 1, bins + (depth_code_offset - 1), 255
                    ).astype(np.uint8)
                    out[wet] = shifted[wet]

                # flood-prone low-lying: connection == 2, only where still dry (255)
                disconnected = (~dem_nodata) & (c == 2) & (out == 255)
                out[disconnected] = floodprone_code

                dst.write(out, 1, window=window)

            # --- Tags for downstream legend rendering (dictionary-based) ---
            def _fmt(v):
                return f"{v:g}" if np.isfinite(v) else ("-inf" if v < 0 else "inf")

            tags = {
                # High-level bin descriptors
                "bin_ids": ",".join(str(int(i)) for i in ids),
                "bin_category": ",".join(str(s) for s in cats),
                "bin_label_ft": ",".join(str(s) for s in lbl_ft),
                "bin_label_m": ",".join(str(s) for s in lbl_m),
                "bin_min_m": ",".join(_fmt(v) for v in dmin),
                "bin_max_m": ",".join(_fmt(v) for v in dmax),
                # Overlay codes
                "floodprone_code": str(floodprone_code),
                f"code_{floodprone_code}_label": floodprone_label,
                # Dry / nodata
                "code_255_label": "dry",  # presentation label (but masked as nodata by many viewers)
                "nodata_label": "dry",  # explicitly state nodata is also 255/dry
            }

            # Per-code details for 1..N
            for i in range(n_depth_bins):
                code = i + depth_code_offset  # 1..N
                tags[f"code_{code}_label"] = str(cats[i])  # generic label = Category
                tags[f"code_{code}_category"] = str(cats[i])
                tags[f"code_{code}_label_ft"] = str(lbl_ft[i])
                tags[f"code_{code}_label_m"] = str(lbl_m[i])
                tags[f"code_{code}_id"] = str(int(ids[i]))
                tags[f"code_{code}_min_m"] = _fmt(float(dmin[i]))
                tags[f"code_{code}_max_m"] = _fmt(float(dmax[i]))

            # Write tags
            dst.update_tags(**tags)


# =============================================================================
# SECTION 5: Shapefile
# =============================================================================


def raster_to_polygons(
    raster_file: str,
    vector_file: str,
    connectivity: int = 8,
    min_pixels: Optional[int] = None,
    dissolve: bool = False,
    driver: Optional[str] = None,
    labels: Optional[pd.DataFrame | Mapping[str, Any]] = None,
    label_key: str = "ID",
    # --- NEW: simplification knobs ---
    simplify_tolerance: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Polygonize a categorical (integer) raster into a vector dataset, with optional
    attribute labeling from a user-provided table, and optional geometry simplification.

    Parameters
    ----------
    raster_file : str
        Path to input raster. The first band is used. **Must be integer dtype.**
    vector_file : str
        Path to output vector file (e.g., .gpkg, .geojson, .shp).
    connectivity : int, default=8
        Pixel connectivity: 4 (rook) or 8 (queen).
    min_pixels : int | None, default=None
        If set (>0), removes patches smaller than this (salt-and-pepper cleanup).
    dissolve : bool, default=False
        If True, dissolve polygons by 'ID' after polygonization.
    driver : str | None, default=None
        Fiona/GDAL driver ('GPKG', 'GeoJSON', 'ESRI Shapefile', etc.). If None, inferred
        from the output file extension.
    labels : pandas.DataFrame | dict | None, default=None
        A label table keyed by `label_key` (default 'ID') that is joined to polygons.
        Example columns: ['ID', 'Category', 'VD_Label', 'VD_Min', 'VD_Max'].
        If a dict is provided, it is converted to a DataFrame.
    label_key : str, default='ID'
        Name of the key column in `labels` used to join to polygon 'ID'.
    simplify_tolerance : float | None, default=None
        If provided, applies `geometry.simplify(simplify_tolerance, preserve_topology=...)`
        AFTER optional dissolve and BEFORE label join. Units are in the GeoDataFrame CRS.

    Returns
    -------
    gpd.GeoDataFrame
        Polygonized GeoDataFrame with an integer 'ID' column and any joined label fields.

    Notes
    -----
    - Input band **must be integer dtype** (e.g., int16, int32). Floats are rejected.
    - NoData handling:
        * If nodata is defined (not None), pixels equal to nodata are masked out.
        * If nodata is None, all values are included.
    - `sieve` operates on integer arrays; we do not cast — the function will error if
      the band is not integer.
    - Simplification:
        * `simplify_tolerance` is in CRS units. In geographic CRS (degrees), use small values.
        * Applied after dissolve, so shared boundaries within the same ID are simplified together.
    - Label table:
        * Duplicated keys in `labels[label_key]` are dropped (first occurrence kept).
        * Infinite values in numeric label columns are converted to NaN/None so drivers
          like Shapefile/GeoJSON can write them.
        * For Shapefile ('ESRI Shapefile'), remember 10-char field name limits and
          stricter type constraints; prefer GeoPackage ('GPKG') for richer schemas.
    """

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if min_pixels is not None and min_pixels <= 0:
        raise ValueError(f"min_pixels must be a positive integer, got {min_pixels}")

    # --- Open raster and enforce integer band ---
    with rasterio.open(raster_file) as src:
        band = src.read(1)  # first band
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # Enforce integer dtype for categorical polygonization
        if not np.issubdtype(band.dtype, np.integer):
            raise TypeError(
                f"Input raster band must be integer dtype; got {band.dtype}. "
                "Please supply a categorical integer raster."
            )

        # Prefer GDAL's valid data mask if available; fallback to != nodata
        # Rasterio masks: non-zero == valid pixels
        mask = None
        if nodata is not None:
            try:
                m = src.read_masks(1)
                mask = m.astype(bool)
            except Exception:
                mask = band != nodata

        # Optional cleanup via sieve (integer arrays only)
        if min_pixels is not None:
            band = sieve(
                band.astype(np.int32),
                size=int(min_pixels),
                connectivity=connectivity,
                mask=mask,
            )

        # Polygonize
        feats = []
        for geom, value in shapes(
            band,
            mask=mask,
            transform=transform,
            connectivity=connectivity,
        ):
            if value is None:
                continue
            v = value.item() if hasattr(value, "item") else value
            feats.append({"geometry": shape(geom), "properties": {"ID": int(v)}})

    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)

    # Drop empties & reset index
    if not gdf.empty:
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)

    # --- (1) Feature-level sieve: drop tiny polygons by CRS-area equivalent of min_pixels ---
    # Pixel area from affine transform: |a*e - b*d| (robust to rotation/shear)
    # min_area = min_pixels * pixel_area
    if not gdf.empty and min_pixels is not None:
        a = transform.a
        b = transform.b
        d = transform.d
        e = transform.e
        pixel_area = abs(a * e - b * d)
        min_area = float(min_pixels) * pixel_area
        gdf = gdf.loc[gdf.geometry.area >= min_area].reset_index(drop=True)

    # --- (2) Fill (remove) small holes inside polygons ---
    if min_pixels is not None and not gdf.empty:

        def _fill_small_holes(geom, min_area):
            if geom.is_empty:
                return geom
            if geom.geom_type == "Polygon":
                kept = []
                for ring in geom.interiors:
                    hole_area = Polygon(ring).area
                    if hole_area >= min_area:
                        kept.append(ring.coords[:])
                return Polygon(geom.exterior.coords[:], kept)
            elif geom.geom_type == "MultiPolygon":
                parts = [_fill_small_holes(p, min_area) for p in geom.geoms]
                return MultiPolygon(parts)
            else:
                return geom

        gdf["geometry"] = gdf.geometry.apply(lambda g: _fill_small_holes(g, min_area))

    # --- (3) Optional dissolve by ID ----
    if dissolve and not gdf.empty:
        gdf = gdf.dissolve(by="ID", as_index=False)

    # --- (4) Optional coverage simplification ---
    if simplify_tolerance is not None and not gdf.empty:
        # Tolerance is in CRS units; disallow geographic (degrees) by default
        if getattr(gdf.crs, "is_projected", False) is not True:
            raise ValueError(
                "Simplification tolerance is interpreted in CRS units, but layer CRS is geographic. "
                "Reproject to a projected CRS (e.g., EPSG:3857 or a local UTM/State Plane) before simplifying."
            )
        # Use simplify_coverage when available; otherwise fallback to per-geometry simplify
        if hasattr(gdf.geometry, "simplify_coverage"):
            gdf["geometry"] = gdf.geometry.simplify_coverage(
                float(simplify_tolerance), simplify_boundary=True
            )
        else:
            gdf["geometry"] = gdf.geometry.simplify(
                float(simplify_tolerance), preserve_topology=True
            )

    # --- (5) Optional label join ---
    if labels is not None and not gdf.empty:
        if isinstance(labels, Mapping):
            labels_df = pd.DataFrame(labels)
        elif isinstance(labels, pd.DataFrame):
            labels_df = labels.copy()
        else:
            raise TypeError("labels must be a pandas.DataFrame or a dict-like mapping.")

        if label_key not in labels_df.columns:
            raise KeyError(f"labels is missing the key column '{label_key}'.")

        labels_df = labels_df.drop_duplicates(subset=[label_key], keep="first").copy()
        # Use nullable Int64 to avoid overflow from downcasting
        labels_df[label_key] = pd.to_numeric(labels_df[label_key]).astype("Int64")
        gdf["ID"] = pd.to_numeric(gdf["ID"]).astype("Int64")

        # Replace +/-inf with NaN in numeric label columns
        num_cols = labels_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            labels_df[num_cols] = labels_df[num_cols].replace([np.inf, -np.inf], np.nan)

        gdf = gdf.merge(labels_df, left_on="ID", right_on=label_key, how="left")
        if label_key != "ID":
            gdf = gdf.drop(columns=[label_key])

    # Write to disk
    if vector_file:
        if driver is None:
            gdf.to_file(vector_file)
        else:
            gdf.to_file(vector_file, driver=driver)

    return gdf


def export_connectivity_regions(
    raster_file: str,
    connected_shp: str,
    disconnected_shp: str,
    connectivity: int = 8,
    min_pixels: int | None = None,
    fix_invalid: bool = True,
    simplify_tolerance: float | None = None,
    driver: str = "ESRI Shapefile",
):
    """
    Polygonize a binary raster (1 = connected, 2 = disconnected) and write two shapefiles
    containing multiple polygons (NO dissolve). Each feature represents one contiguous region.

    Parameters
    ----------
    raster_file : str
        Path to input raster with values 1 (connected) and 2 (disconnected). May include NoData.
    connected_shp : str
        Output shapefile path for connected regions (ID=1).
    disconnected_shp : str
        Output shapefile path for disconnected regions (ID=2).
    connectivity : int, default=8
        Pixel connectivity used during polygonization (4 = rook, 8 = queen).
    min_pixels : int | None, default=None
        If set, removes patches smaller than this number of pixels before polygonization.
    fix_invalid : bool, default=True
        If True, repairs invalid geometries with a zero-width buffer.
    simplify_tolerance : float | None, default=None
        If set (in CRS units), simplifies geometries (preserve_topology=True) to reduce vertices.
        Use cautiously in geographic CRS; best in projected CRS.
    driver : str, default="ESRI Shapefile"
        GDAL driver for outputs. Alternatives: "GPKG", "GeoJSON", etc.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        (connected_gdf, disconnected_gdf), each with multiple polygons and attributes:
        - ID: 1 or 2 (class)
        - region_id: sequential identifier per shapefile
        - area_m2: optional, if CRS is projected
    """
    # --- Read raster ---
    with rasterio.open(raster_file) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # Build mask to exclude NoData from polygonization
        if nodata is None:
            if np.issubdtype(band.dtype, np.floating):
                mask = ~np.isnan(band)
            else:
                mask = None  # polygonize all values
        else:
            mask = band != nodata

        # Optional cleanup: remove small patches prior to polygonization
        if min_pixels:
            band = sieve(band, size=min_pixels, connectivity=connectivity)

        # Polygonize once; split features into two lists
        feats_conn, feats_disc = [], []
        for geom, val in shapes(
            band, mask=mask, transform=transform, connectivity=connectivity
        ):
            if val == 1:
                feats_conn.append({"geometry": shape(geom), "properties": {"ID": 1}})
            elif val == 2:
                feats_disc.append({"geometry": shape(geom), "properties": {"ID": 2}})
            # ignore any other values

    # Build GeoDataFrames
    gdf_conn = gpd.GeoDataFrame.from_features(feats_conn, crs=crs)
    gdf_disc = gpd.GeoDataFrame.from_features(feats_disc, crs=crs)

    # --- Prep helper ---
    def _prep(gdf: gpd.GeoDataFrame, id_value: int) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf
        gdf = gdf[gdf.geometry.notna()]
        if fix_invalid:
            gdf = gdf.set_geometry(gdf.geometry.buffer(0))  # repair self-intersections
        # Split MultiPolygons into single-part features (each part is treated as a region)
        gdf = gdf.explode(index_parts=False, ignore_index=True)
        # Optional simplify to reduce vertex count
        if simplify_tolerance is not None and simplify_tolerance > 0:
            if hasattr(gdf.geometry, "simplify_coverage"):
                gdf = gdf.set_geometry(gdf.geometry.simplify_coverage(simplify_tolerance, simplify_boundary=True))
            else:
                gdf = gdf.set_geometry(gdf.geometry.simplify(float(simplify_tolerance), preserve_topology=True))

        # Assign sequential region IDs per shapefile
        gdf["region_id"] = np.arange(1, len(gdf) + 1, dtype=int)
        gdf["ID"] = id_value

        # Optional area in square meters if CRS is projected
        try:
            if gdf.crs and getattr(gdf.crs, "is_projected", False):
                gdf["area_m2"] = gdf.geometry.area
        except Exception:
            pass

        return gdf

    gdf_conn = _prep(gdf_conn, 1)
    gdf_disc = _prep(gdf_disc, 2)

    # --- Write outputs ---
    for out_path, gdf in [(connected_shp, gdf_conn), (disconnected_shp, gdf_disc)]:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        if not gdf.empty:
            gdf.to_file(out_path, driver=driver)
        else:
            print(
                f"Info: No polygons written for {out_path} (no regions of class {gdf['ID'].iloc[0] if not gdf.empty else 'N/A'})."
            )

    return gdf_conn, gdf_disc


def _crs_kind(crs) -> str:
    """
    Classify CRS as 'projected', 'geographic', or 'unknown'.
    Works with rasterio.crs.CRS, pyproj.CRS, EPSG strings/ints, WKT, etc.
    """
    if crs is None:
        return "unknown"
    try:
        # Normalize to pyproj.CRS
        if hasattr(crs, "to_wkt"):
            crs_py = CRS.from_wkt(crs.to_wkt())
        else:
            crs_py = CRS.from_user_input(crs)
    except Exception:
        return "unknown"

    if crs_py.is_projected:
        return "projected"
    if crs_py.is_geographic:
        return "geographic"
    return "unknown"
