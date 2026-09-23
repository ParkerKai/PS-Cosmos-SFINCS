# downscale_sfincs_results.py
"""
Unified SFINCS post-processing driver for PS-CoSMoS (corrected RP construction).

Pipeline:
  1. Aggregate SFINCS water-year outputs -> per-cell annual maxima of zsmax (+ extras).
  2. EVA: empirical Weibull plotting positions & log-linear interpolation to target RPs;
     optionally mask WSE below subgrid dry floor (z_zmin).
  3. Build / reuse DEM-pixel -> SFINCS-cell index COG.
  4. Downscale per-RP zsmax -> hmax + zsmax GeoTIFFs (face-space dilation then resample).
  5. Remove disconnected flooding via sfincs.bnd.
  6. Map extras (qmax, tmax, ...) to DEM using index COG.
  7. Bin depth & VD (qmax) into hazards.
  8. (Optional) First RP that wets each pixel map.

Run:
    conda run -n hydromt-sfincs-dev python downscale_sfincs_results.py
"""

import sys
import time
from pathlib import Path
import os
import numpy as np
import xarray as xr
import xugrid as xu
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from hydromt_sfincs import SfincsModel
from hydromt_sfincs.workflows import downscaling

# If you want live prints in VS Code, uncomment:
# sys.stdout.reconfigure(line_buffering=True)

# -----------------------------------------------------------------------------
# Helper import (relative project structure)
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
WORKFLOW_DIR = THIS_FILE.parents[1]  # ./workflow
TOOLS_DIR = WORKFLOW_DIR / "01_tools"  # ./workflow/01_tools
sys.path.insert(0, str(TOOLS_DIR))
import ps_cosmos_postprocess_helper as helper  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
# --- Inputs ---
sfincs_root = Path(r"D:\Kai\SFINCS\Snohomish_slr000")
bnd_file = os.path.join(sfincs_root, "sfincs.bnd")

# DEM + index COG
dem_res = 1  # DEM resolution [m]
dem_file = Path(r"D:\Kai\DataDownloads\Snohomish_MosaicDEM_modded.tif")
indices_fn = Path(
    os.path.join(sfincs_root, "downscaled_results_1m", f"indices_{dem_res}m.tif")
)

# Optional subgrid dry-floor (z_zmin)
subgrid_fn = Path(os.path.join(sfincs_root, "sfincs_subgrid.nc"))

# Output
output_dir = Path(os.path.join(sfincs_root, "downscaled_results_1m"))
shapefile_dir = Path(os.path.join(sfincs_root, "downscaled_results_1m", "shapefiles"))
domain_stem = f"_{dem_res}m"

# --- Aggregation + EVA ---
aggregation_mode = "annual_max"  # "annual_max" | "all_maxima"
eva_method = "weibull"  # "weibull" | "gev" | "pot"
return_periods = [1, 2, 5, 10, 20, 50]
extra_vars = ["qmax"]  # e.g., ["qmax", "tmax", "tmax_zs"]

# POT-only
pot_target_per_yr = 5
pot_decluster = "72h"

# GEV-only
gev_min_years = 5

# --- Downscaling ---
downscale_method = "bilinear"  # "raw" | "constant" | "bilinear"
dilation = 0.5  # face-space dilation factor before downscaling
hmin = 0.02  # wet threshold [m] for hmax

# --- Clipping ---
clip_extent = True
clip_polygon = Path(r"D:\Kai\SFINCS\GIS\Snohomish_ClippingPolygon.shp")

# --- Smoothing ---
smoothing = True
smooth_sigma_h = 4  # sigma [pixels] for hmax
smooth_sigma_extras = 22  # sigma [pixels] for extras (e.g., qmax)
smooth_truncate = 4

# --- Hazard binning ---
depth_bins = {
    "ID": np.array([1, 2, 3, 4, 5], dtype="int16"),
    "Category": ["Low", "Medium", "High", "VeryHigh", "Extreme"],
    "Label_ft": ["<0.5", "0.5-1.0", "1.0-3.0", "3.0-5.0", ">5.0"],
    "Label_m": ["<0.15", "0.15-0.3", "0.3-0.9", "0.9-1.5", ">1.5"],
    "D_Min": np.array([-np.inf, 0.1524, 0.3048, 0.9144, 1.524], dtype="float32"),
    "D_Max": np.array([0.1524, 0.3048, 0.9144, 1.524, np.inf], dtype="float32"),
}

qmax_bins = {
    "ID": np.array([1, 2, 3, 4, 5], dtype="int16"),
    "Category": ["Low", "Medium", "High", "VeryHigh", "Extreme"],
    "VD_Label": ["<0.2", "0.2-0.5", "0.5-1.5", "1.5-2.5", ">2.5"],
    "VD_Min": np.array([0.0, 0.2, 0.5, 1.5, 2.5], dtype="float32"),
    "VD_Max": np.array([0.2, 0.5, 1.5, 2.5, np.inf], dtype="float32"),
}

# --- Step toggles ---
run_aggregation = True
run_eva = True
run_downscale = True
run_disconnect = True
run_extras = True
run_binning = True
run_shapefiles = True
run_first_rp_map = True

# Index COG block size
NRMAX = 2000

# =============================================================================
# Discover water-year run directories
# =============================================================================
sfincs_dirs = sorted(
    p
    for p in sfincs_root.iterdir()
    if p.is_dir() and len(p.name) == 4 and (p / "sfincs_map.nc").exists()
)
if not sfincs_dirs:
    raise RuntimeError(f"No sfincs_map.nc found under {sfincs_root}")
print(
    f"Found {len(sfincs_dirs)} water-year runs: {sfincs_dirs[0].name}..{sfincs_dirs[-1].name}"
)

# =============================================================================
# Derived paths + provenance tag
# =============================================================================
if aggregation_mode == "annual_max":
    aggregated_fn = Path(os.path.join(output_dir, "aggregated_annual.nc"))
    timeseries_fn = Path(os.path.join(output_dir, "aggregated_annual.timeseries.nc"))
    keep_timeseries = eva_method == "pot"
elif aggregation_mode == "all_maxima":
    aggregated_fn = Path(os.path.join(output_dir, "aggregated_all.nc"))
    timeseries_fn = aggregated_fn
    keep_timeseries = False
else:
    raise ValueError("aggregation_mode must be 'annual_max' or 'all_maxima'.")

AGG_TOKEN = {"annual_max": "annual", "all_maxima": "all"}[aggregation_mode]
provenance_tag = f"{eva_method}_{AGG_TOKEN}"

paths = helper.OutputPaths(
    output_dir=output_dir,
    shapefile_dir=shapefile_dir,
    domain_stem=domain_stem,
    provenance_tag=provenance_tag,
)
paths.ensure_dirs()

# =============================================================================
# Step 1: Water-year aggregation
# =============================================================================
print(f"\n{'=' * 60}\nStep 1: water-year aggregation ({aggregation_mode})\n{'=' * 60}")
if run_aggregation:
    t0 = time.time()
    needs_sidecar = (
        aggregation_mode == "annual_max"
        and keep_timeseries
        and not timeseries_fn.exists()
    )
    if aggregated_fn.exists() and not needs_sidecar:
        print(f"  skip: {aggregated_fn.name} exists")
    else:
        helper.aggregate_water_year_maxima(
            sfincs_dirs=sfincs_dirs,
            extra_vars=extra_vars,
            output_fn=aggregated_fn,
            keep_timeseries=keep_timeseries,
            aggregation_mode=aggregation_mode,
        )
        print(f"  wrote: {aggregated_fn.name} ({time.time() - t0:.1f}s)")
else:
    print("  skipped by toggle")

ds_annual = xr.open_dataset(aggregated_fn)
ds_full = None
if eva_method == "pot" and aggregation_mode == "annual_max":
    if not timeseries_fn.exists():
        raise FileNotFoundError(
            f"POT EVA requested but missing {timeseries_fn}. "
            "Re-run with run_aggregation=True to persist the full series."
        )
    ds_full = xr.open_dataset(timeseries_fn)

# Optional: subgrid dry-floor
z_zmin = None
try:
    if subgrid_fn.exists():
        with xr.open_dataset(subgrid_fn) as sds:
            z_zmin = np.asarray(sds["z_zmin"].values, dtype=np.float64)
        print(
            f"  subgrid z_zmin range [{np.nanmin(z_zmin):.3f}, {np.nanmax(z_zmin):.3f}] m"
        )
except Exception as exc:
    print(
        f"  note: cannot read subgrid z_zmin ({type(exc).__name__}); proceeding without dry-floor masking"
    )

# Universal provenance tags
n_years_stamped = int(ds_annual.attrs.get("n_years", len(sfincs_dirs)))
base_tags = {
    "aggregation_mode": aggregation_mode,
    "eva_method": eva_method,
    "n_years": n_years_stamped,
    "hmin": hmin,
    "domain_stem": domain_stem,
    "produced_by": "downscale_sfincs_results.py",
}
if sfincs_dirs:
    base_tags["water_year_first"] = sfincs_dirs[0].name
    base_tags["water_year_last"] = sfincs_dirs[-1].name

# =============================================================================
# Step 2: EVA (RP construction)
# =============================================================================
eva_fn = output_dir / f"eva_RP_{eva_method}.nc"
print(f"\n{'=' * 60}\nStep 2: EVA ({eva_method})\n{'=' * 60}")
if run_eva:
    t0 = time.time()
    if eva_fn.exists():
        print(f"  skip: {eva_fn.name} exists")
    else:

        # Replaces non-finite values with the floor value
        # Preserves Unfirom record length cross all cells, even if cell is dry in many years.  
        # Floor entries participate in the ranking, but are masked out after interpolation when the RP value is < floor
        if (z_zmin is not None and "zsmax_ann" in ds_annual and ds_annual.attrs.get("aggregation_mode", "annual_max") == "annual_max"):
            zs = ds_annual["zsmax_ann"]  # dims: (year, nmesh2d_face)
            floor_da = xr.DataArray(
                z_zmin.astype(np.float64),
                dims=("nmesh2d_face",),
                coords={"nmesh2d_face": zs["nmesh2d_face"]},
            )
            # Fill NaNs with floor; broadcast across 'year'
            ds_annual["zsmax_ann"] = zs.fillna(floor_da)

        # Then run EVA (Weibull) as you do:
        ds_eva = helper.eva_apply(
            ds_annual,
            method="weibull",
            return_periods=return_periods,
            z_floor=z_zmin,          # stays the same
        )

        
        ds_eva = helper.eva_apply(
            ds_annual,
            method=eva_method,
            return_periods=return_periods,
            ds_full_timeseries=ds_full,
            pot_target_per_year=pot_target_per_yr,
            pot_decluster=pot_decluster,
            gev_min_years=gev_min_years,
            z_floor=z_zmin,  # optional dry-floor masking at WSE stage
        )
        # Event-matched extras (Weibull/GEV) via empirical rank
        if eva_method in ("weibull", "gev") and extra_vars:
            ds_extras_rp = helper.extras_at_rp_via_rank(
                ds_annual,
                return_periods=return_periods,
                extras=extra_vars,
            )
            for v in ds_extras_rp.data_vars:
                ds_eva[v] = ds_extras_rp[v]
        ds_eva.to_netcdf(eva_fn)
        print(f"  wrote: {eva_fn.name} ({time.time() - t0:.1f}s)")
else:
    print("  skipped by toggle")

ds_eva = xr.open_dataset(eva_fn)

# =============================================================================
# Step 3: SFINCS model + index COG
# =============================================================================
print(f"\n{'=' * 60}\nStep 3: SFINCS model + index COG\n{'=' * 60}")

mod = SfincsModel(str(sfincs_dirs[0]), mode="r")
print(f"  grid type: {mod.grid_type}")

if not indices_fn.exists():
    print(f"  building index COG: {indices_fn}")
    indices_fn.parent.mkdir(parents=True, exist_ok=True)
    downscaling.make_index_cog(mod, indices_fn, dem_file, nrmax=NRMAX)
    print("  index COG created")
else:
    print(f"  reuse: {indices_fn}")

# Build quadtree topology template for copy-on-write of EVA faces
if run_downscale:
    template_src = Path(sfincs_dirs[0]) / "sfincs_map.nc"
    print(f"  building topology template from {template_src}")
    _template_ds = xu.open_dataset(template_src)
    zsmax_template = (
        _template_ds["zsmax"].isel(timemax=0).drop_vars("timemax", errors="ignore")
    )

# =============================================================================
# Step 4: Downscale per RP (dilation -> resample)
# =============================================================================
print(f"\n{'=' * 60}\nStep 4: downscale per RP ({downscale_method})\n{'=' * 60}")
if run_downscale:
    # Guards
    if "zsmax_rp" not in ds_eva:
        raise KeyError("EVA output missing 'zsmax_rp'.")

    eva_faces = ds_eva["zsmax_rp"].sizes.get("nmesh2d_face")
    tmpl_faces = zsmax_template.sizes.get("nmesh2d_face")
    if eva_faces != tmpl_faces:
        raise RuntimeError("face count mismatch between EVA dataset and SFINCS model")

    clip_gdf = helper.load_clip_polygon(clip_polygon) if clip_extent else None

    for rp in return_periods:
        h_fn = paths.hmax(rp)
        z_fn = paths.zsmax(rp)
        if h_fn.exists() and z_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}: outputs exist")
            continue

        t0 = time.time()
        # Re-attach quadtree topology by copying EVA face values into template
        eva_face = ds_eva["zsmax_rp"].sel(rp=rp)
        zsmax_uda = zsmax_template.copy(
            data=eva_face.values.astype(zsmax_template.dtype)
        )
        zsmax_uda.name = "zsmax"

        # Face-space dilation prior to resampling
        if downscale_method == "bilinear" and dilation:
            zsmax_uda = downscaling.adjust_zsmax_dilation(zsmax_uda, factor=dilation)

        # Downscale to hmax (subtract DEM, apply hmin)
        downscaling.downscale_floodmap(
            zsmax=zsmax_uda,
            dep=str(dem_file),
            reproj_method="bilinear",
            subtract_dem=True,
            hmin=hmin,
            indices=str(indices_fn),
            floodmap_fn=str(h_fn),
            nrmax=NRMAX,
            gdf_mask=clip_gdf if clip_extent else None,
        )

        # Downscale to zsmax (WSE map)
        downscaling.downscale_floodmap(
            zsmax=zsmax_uda,
            dep=str(dem_file),
            reproj_method="bilinear",
            subtract_dem=False,
            indices=str(indices_fn),
            zsmap_fn=str(z_fn),
            nrmax=NRMAX,
            gdf_mask=clip_gdf if clip_extent else None,
        )

        step_tags = {
            "return_period": int(round(rp)),
            "downscale_method": downscale_method,
            "dilation": dilation if downscale_method == "bilinear" else "",
        }
        helper.stamp_provenance(h_fn, **base_tags, **step_tags, variable="hmax")
        helper.stamp_provenance(z_fn, **base_tags, **step_tags, variable="zsmax")
        print(f"  {helper.rp_tag(rp)}: wrote hmax + zsmax ({time.time() - t0:.1f}s)")

        # Optional smoothing of hmax
        if smoothing:
            h_smooth_fn = paths.hmax_smooth(rp)
            print(
                f"  {helper.rp_tag(rp)}: smoothing {h_smooth_fn.name} (σ={smooth_sigma_h})"
            )
            helper.smooth_raster_gaussian_blockwise(
                in_fn=h_fn,
                out_fn=h_smooth_fn,
                smooth_size=smooth_sigma_h,
                truncate=smooth_truncate,
            )
            helper.stamp_provenance(
                h_smooth_fn,
                **base_tags,
                smoothing=f"gaussian filter (σ={smooth_sigma_h})",
            )
else:
    print("  skipped by toggle")

# =============================================================================
# Step 5: Remove disconnected flooding
# =============================================================================
print(f"\n{'=' * 60}\nStep 5: disconnected-flooding removal\n{'=' * 60}")
if run_disconnect:
    for rp in return_periods:
        h_in = (
            paths.hmax_smooth(rp)
            if (smoothing and paths.hmax_smooth(rp).exists())
            else paths.hmax(rp)
        )
        z_fn = paths.zsmax(rp)
        h_m = paths.hmax_masked(rp)
        z_m = paths.zsmax_masked(rp)
        c_fn = paths.connection(rp)

        if not h_in.exists():
            print(f"  skip {helper.rp_tag(rp)}: hmax not found")
            continue
        if h_m.exists() and z_m.exists() and c_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}: masked outputs exist")
            continue

        t0 = time.time()
        downscaling.remove_disconnected_flooding(
            depth_fn=str(h_in),
            bnd_fn=str(bnd_file),
            hmin=hmin,
            connection_fn=str(c_fn),
            output_fns={str(h_in): str(h_m), str(z_fn): str(z_m)},
        )
        step5_tags = {
            "return_period": int(round(rp)),
            "produced_by_step": "remove_disconnected_flooding",
            "bnd_file": Path(bnd_file).name,
        }
        helper.stamp_provenance(h_m, **base_tags, **step5_tags, variable="hmax_masked")
        helper.stamp_provenance(z_m, **base_tags, **step5_tags, variable="zsmax_masked")
        helper.stamp_provenance(
            c_fn, **base_tags, **step5_tags, variable="connection_mask"
        )
        print(
            f"  {helper.rp_tag(rp)}: connection + masked rasters ({time.time() - t0:.1f}s)"
        )
else:
    print("  skipped by toggle")

# =============================================================================
# Step 6: Map extras (qmax, tmax, ...) to DEM
# =============================================================================
print(f"\n{'=' * 60}\nStep 6: extras nearest-neighbour mapping\n{'=' * 60}")
if run_extras and extra_vars:
    for rp in return_periods:
        depth_for_mask = (
            paths.hmax_masked(rp) if paths.hmax_masked(rp).exists() else paths.hmax(rp)
        )
        if not depth_for_mask.exists():
            print(f"  skip {helper.rp_tag(rp)}: no hmax raster")
            continue

        for v in extra_vars:
            out_fn = paths.extra(v, rp)
            if out_fn.exists():
                print(f"  skip {helper.rp_tag(rp)}/{v}: exists")
                continue
            key = f"{v}_rp"
            if key not in ds_eva:
                print(f"  skip {helper.rp_tag(rp)}/{v}: {key} not in EVA dataset")
                continue

            t0 = time.time()
            helper.map_quadtree_to_dem_nearest(
                da_face=ds_eva[key].sel(rp=rp),
                indices_fn=indices_fn,
                out_fn=out_fn,
                depth_mask_fn=depth_for_mask,
                hmin=hmin,
            )
            helper.stamp_provenance(
                out_fn,
                **base_tags,
                return_period=int(round(rp)),
                variable=v,
                source="quadtree_nearest_via_index_cog",
            )

            if smoothing:
                out_smooth = paths.extra_smooth(v, rp)
                print(
                    f"  {helper.rp_tag(rp)}/{v}: smoothing {out_smooth.name} (σ={smooth_sigma_extras})"
                )
                helper.smooth_raster_gaussian_blockwise(
                    in_fn=out_fn,
                    out_fn=out_smooth,
                    smooth_size=smooth_sigma_extras,
                    truncate=smooth_truncate,
                )
                helper.stamp_provenance(
                    out_smooth,
                    **base_tags,
                    smoothing=f"gaussian filter (σ={smooth_sigma_extras})",
                )

            print(
                f"  {helper.rp_tag(rp)}/{v}: wrote {out_fn.name} ({time.time() - t0:.1f}s)"
            )
else:
    print("  skipped by toggle / no extras configured")

# =============================================================================
# Step 7: Hazard binning (depth + qmax)
# =============================================================================
print(f"\n{'=' * 60}\nStep 7: hazard binning (depth + qmax)\n{'=' * 60}")
if run_binning:
    for rp in return_periods:
        h_for_bin = (
            paths.hmax_masked(rp) if paths.hmax_masked(rp).exists() else paths.hmax(rp)
        )
        if not h_for_bin.exists():
            print(f"  skip {helper.rp_tag(rp)}: no hmax raster")
            continue

        # Depth bins composite
        d_bins_fn = paths.depth_bins(rp)
        c_fn = paths.connection(rp)
        if d_bins_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}/depth_bins: exists")
        elif not c_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}/depth_bins: connection raster not found")
        else:
            t0 = time.time()
            helper.bin_depth_with_overlays(
                hmax_masked_fn=h_for_bin,
                connection_fn=c_fn,
                dem_fn=dem_file,
                depth_bins=depth_bins,
                out_fn=d_bins_fn,
            )
            helper.stamp_provenance(
                d_bins_fn,
                **base_tags,
                return_period=int(round(rp)),
                variable="depth_bins_composite",
                source_depth=Path(h_for_bin).name,
                source_connection=Path(c_fn).name,
                source_dem=Path(dem_file).name,
                units="meters",
            )
            print(
                f"  {helper.rp_tag(rp)}/depth_bins: {d_bins_fn.name} ({time.time() - t0:.1f}s)"
            )

        # qmax bins (VD product)
        q_fn = (
            paths.extra_smooth("qmax", rp)
            if (smoothing and paths.extra_smooth("qmax", rp).exists())
            else paths.extra("qmax", rp)
        )
        if not q_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}/qmax_bins: qmax raster not available")
            continue

        q_bins_fn = paths.qmax_bins(rp)
        if q_bins_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}/qmax_bins: exists")
        else:
            t0 = time.time()
            helper.bin_raster(q_fn, qmax_bins, q_bins_fn)
            helper.stamp_provenance(
                q_bins_fn,
                **base_tags,
                return_period=int(round(rp)),
                variable="qmax_bins",
                source=Path(q_fn).name,
                units="m2/s",
            )
            print(
                f"  {helper.rp_tag(rp)}/qmax_bins: {q_bins_fn.name} ({time.time() - t0:.1f}s)"
            )
else:
    print("  skipped by toggle")

# =============================================================================
# Step 8: Shapefile conversion
# =============================================================================
print(f"\n{'=' * 60}\nStep 8: shapefile conversion\n{'=' * 60}")
if run_shapefiles:
    for rp in return_periods:
        # Extent (Connected/Disconnected)
        extentConn_shp_fn = paths.extent_connected_shapefile(rp)
        extentDisConn_shp_fn = paths.extent_disconnected_shapefile(rp)
        c_fn = paths.connection(rp)

        if extentConn_shp_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}/extent shapefiles: Connected exists")
        elif extentDisConn_shp_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}/extent shapefiles: Disconnected exists")
        elif not c_fn.exists():
            print(
                f"  skip {helper.rp_tag(rp)}/extent shapefiles: connection raster not found"
            )
        else:
            t0 = time.time()
            helper.export_connectivity_regions(
                raster_file=c_fn,
                connected_shp=extentConn_shp_fn,
                disconnected_shp=extentDisConn_shp_fn,
                connectivity=8,
                min_pixels=30,
                fix_invalid=True,
                simplify_tolerance=2,
                driver="ESRI Shapefile",
            )
            print(
                f"  {helper.rp_tag(rp)}/extent shapefiles: {extentConn_shp_fn.name} & {extentDisConn_shp_fn.name} ({time.time() - t0:.1f}s)"
            )

        # Depth bins polygons (include "LowLying" label for code N+1)
        dbins_shp_fn = paths.depth_shapefile(rp)
        d_bins_fn = paths.depth_bins(rp)
        if not d_bins_fn.exists():
            print(
                f"  skip {helper.rp_tag(rp)}/depth_bins: depth bins raster not available"
            )
            continue

        depth_bins_labels = {
            k: (np.array(v) if isinstance(v, list) else v)
            for k, v in depth_bins.items()
        }
        # Assuming N=5 depth bins -> floodprone code = 6
        depth_bins_labels["ID"] = np.append(depth_bins_labels["ID"], np.int16(6))
        depth_bins_labels["Category"] = list(depth_bins_labels["Category"]) + [
            "LowLying"
        ]
        depth_bins_labels["Label_ft"] = list(depth_bins_labels["Label_ft"]) + ["N/A"]
        depth_bins_labels["Label_m"] = list(depth_bins_labels["Label_m"]) + ["N/A"]
        depth_bins_labels["D_Min"] = np.append(depth_bins_labels["D_Min"], np.nan)
        depth_bins_labels["D_Max"] = np.append(depth_bins_labels["D_Max"], np.nan)

        if dbins_shp_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}/depth_bins: exists")
        else:
            t0 = time.time()
            helper.raster_to_polygons(
                raster_file=d_bins_fn,
                vector_file=dbins_shp_fn,
                connectivity=8,
                min_pixels=30,
                dissolve=True,
                driver="ESRI Shapefile",
                labels=depth_bins_labels,
                label_key="ID",
                simplify_tolerance=2,
            )
            print(
                f"  {helper.rp_tag(rp)}/depth_bins: {d_bins_fn.name} ({time.time() - t0:.1f}s)"
            )

        # qmax bins polygons
        qbins_shp_fn = paths.qmax_shapefile(rp)
        q_bins_fn = paths.qmax_bins(rp)
        if not q_bins_fn.exists():
            print(
                f"  skip {helper.rp_tag(rp)}/qmax_bins: qmax binned raster not available"
            )
            continue
        if qbins_shp_fn.exists():
            print(f"  skip {helper.rp_tag(rp)}/qmax_bins: exists")
        else:
            t0 = time.time()
            helper.raster_to_polygons(
                raster_file=q_bins_fn,
                vector_file=qbins_shp_fn,
                connectivity=8,
                min_pixels=30,
                dissolve=True,
                driver="ESRI Shapefile",
                labels=qmax_bins,
                label_key="ID",
                simplify_tolerance=1,
            )
            print(
                f"  {helper.rp_tag(rp)}/qmax_bins: {qbins_shp_fn.name} ({time.time() - t0:.1f}s)"
            )
else:
    print("  skipped by toggle")
# =============================================================================
# Step 9 (optional): First RP wet map
# =============================================================================
if run_first_rp_map:
    print(f"\n{'=' * 60}\nStep 9: first-RP-wet map\n{'=' * 60}")

    # Prefer smoothed hmax -> masked hmax -> raw hmax, per RP
    rp_hmax_files = []
    for rp in return_periods:
        candidates = [
            paths.hmax_smooth(rp) if smoothing else None,
            paths.hmax_masked(rp),
            paths.hmax(rp),
        ]
        for fn in candidates:
            if fn is not None and fn.exists():
                rp_hmax_files.append((rp, fn))
                break

    if rp_hmax_files:
        first_fn = output_dir / f"rp_first_wet{domain_stem}.tif"

        # Use first raster to get output profile & dimensions
        with rasterio.open(str(rp_hmax_files[0][1])) as src0:
            prof = src0.profile.copy()
            prof.update(
                dtype="uint16",
                nodata=0,
                count=1,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
            )
            H, W = src0.height, src0.width

        # Open all RP rasters (ascending RP)
        srcs = [(rp, rasterio.open(str(fn))) for rp, fn in rp_hmax_files]

        # Build first-RP-wet map in blocks
        B = 2048
        counts = {rp: 0 for rp, _ in rp_hmax_files}
        with rasterio.open(str(first_fn), "w", **prof) as dst:
            for r0 in range(0, H, B):
                for c0 in range(0, W, B):
                    win = Window(c0, r0, min(B, W - c0), min(B, H - r0))

                    # 0 = never wet; store the lowest RP that is wet (>0 depth)
                    out = np.zeros((int(win.height), int(win.width)), dtype=np.uint16)

                    for rp, src in srcs:  # low RP first
                        a = src.read(1, window=win, masked=True)
                        vals = np.ma.filled(a, np.nan)
                        wet = (
                            (~np.ma.getmaskarray(a)) & np.isfinite(vals) & (vals > 0.0)
                        )
                        newly = wet & (out == 0)
                        out[newly] = int(round(rp))
                        counts[rp] += int(newly.sum())

                    dst.write(out, 1, window=win)

        # Close all sources and add overviews
        for _, s in srcs:
            s.close()
        with rasterio.open(str(first_fn), "r+") as dst:
            dst.build_overviews([2, 4, 8, 16], Resampling.nearest)

        # Log summary
        tot = sum(counts.values())
        print(f"  wrote {first_fn.name}: {tot:,} wet pixels total")
        for rp in return_periods:
            if rp in counts:
                pct = 100.0 * counts[rp] / tot if tot else 0.0
                print(f"    first wet at RP{int(rp):>3}: {counts[rp]:>12,} px  ({pct:5.1f}%)")
                
    else:
        print("  skip: no RP hmax files available to build first-RP-wet map")
