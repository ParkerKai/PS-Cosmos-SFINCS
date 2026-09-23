# PS-Cosmos-SFINCS
This repo hold the PS-Cosmos SFINCS-based coastal or flood modeling workflow.

This repository is organized to support the full project lifecycle: preprocessing, model setup, execution, post-processing, and figure generation. Use this README as a baseline and replace the placeholder sections with the specifics of your study area, modeling workflow, and outputs.

## Overview

Describe the purpose of the project here.

- Study area: Puget Sound, Washington, USA
- Modeling framework: SFINCS 
- Main objective: Local scale flood model development and PS-Cosmos data delivery
- Key deliverables: USGS Cosmos products
- Project period: 2024-2027

## Repository Structure

```text
PS-Cosmos-SFINCS/
├── 01_tools/                  # Helper scripts and utilities
│   ├── ps_cosmos_postprocess_helper.py
│   └── SFINCS_QuadtreeTools.py
├── 02_preprocessing/          # Data preparation and preprocessing notebooks/scripts
│   └── ReadMe.txt
├── 03_modelbuilds/            # Individual model domains or study areas
│   ├── 01_King/
│   ├── 02_Pierce/
│   ├── 03_Snohomish/
│   ├── 04_Kitsap/
│   └── 05_Stillaguamish/
├── 04_postprocessing/         # Post-processing and output aggregation scripts
│   └── downscale_sfincs_results.py
├── 05_figures/                # Figure generation and plotting workflows
│   └── ReadMe.txt
├── 06_project/                # Project metadata, documentation, and configs
│   └── ReadMe.txt
├── 07_temp/                   # Temporary files and scratch work
│   └── ReadMe.txt
├── README.md                  # Project overview and instructions
└── .git/                      # Git metadata
```

## Workflow

1. Prepare inputs in the preprocessing directory.
2. Define or update model setups under the relevant model build folder.
3. Run the SFINCS model using the required domain-specific configuration.
4. Process model output in the post-processing scripts.
5. Generate figures and summary products for reporting.
6. Store project notes, metadata, and final documentation in the project directory.

## Getting Started

### Prerequisites

### Recommended Setup


## Data and Inputs

Document the data sources and required files here.

- DEM / topography
- Land cover or roughness layers
- Bathymetry or coastal boundary conditions
- Forcing data (rainfall, wind, tides, surge, river discharge)
- Model boundary and mesh configuration files
- Calibration or validation datasets


## Model Builds

Each model build folder can contain its own configuration, parameter files, and output folders. Keep the organization consistent across domains.

### Example structure

```text
03_modelbuilds/
└── 01_King/
    ├── README.md or ReadMe.txt
    ├── inputs/
    ├── config/
    ├── output/
    └── logs/
```

## Outputs

List the expected outputs here, such as:

- Max flood depth maps
- Water level time series
- Hazard summary rasters
- Comparison plots
- Model validation summaries
- Final report figures

## Notes and Best Practices

- Keep raw inputs separate from processed products.
- Use clear naming conventions for files and directories.
- Document assumptions, parameter choices, and model limitations.
- Archive output products in a reproducible and traceable way.
- Update this README whenever the workflow changes.

## Project Status


## Contact / Ownership

Contact: O'Neill, Andrea <aoneill@usgs.gov>
Contact: Henderson, Cassandra <cshender@uw.edu>
contact: Parker, Kai <kaip461@ecy.wa.gov>
contact: Nederhoff, Kees <kees.nederhoff@deltares-usa.us>

USGS- Cosmos Team
