# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:54:42 2024

Functions to interface with matlab

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"


import xarray as xr
import numpy as np
from typing import Tuple, Optional

# ===============================================================================
# %% Define some functions
# ===============================================================================

def load_SfincsQuadtree(
    file_in: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
    """
    Load SFINCS quadtree grid and compute cell centers and z values.

    Parameters
    ----------
    file_in : str
        Path to the NetCDF file.
    Returns
    -------
    xc : (nFaces,) ndarray
        X-coordinate of cell centers.
    yc : (nFaces,) ndarray
        Y-coordinate of cell centers.
    zc : (nFaces,) ndarray
        Values associated with each face (from z_var).

    Notes
    -----
    - Assumes lower-left corner structure in sfincs.nc grid)
    """

    # Load quadtree grid data
    qtr = xr.open_dataset(file_in)

    # Basic geometry arrays
    bx = qtr.mesh2d_node_x.values
    by = qtr.mesh2d_node_y.values

    # Load the face nodes
    face_da = qtr["mesh2d_face_nodes"]

    # Replace NaNs with -1, then cast to int
    i = xr.where(np.isnan(face_da), -1, face_da).astype("int32").values
    lev = qtr.level.values

    # Get the grid spacing
    qdx = qtr.dx / (2 ** (lev - 1))
    qdy = qtr.dy / (2 ** (lev - 1))

    # Get the corners
    x0, y0 = [], []
    for j in range(len(i)):
        x0.append(bx[i[j][0]])
        y0.append(by[i[j][0]])
    x0 = np.array(x0)
    y0 = np.array(y0)

    # Get the centers
    xc, yc = x0 + qdx / 2, y0 + qdy / 2

    # Add the z values
    zc = qtr['z'].values

    # Close dataset (defensive; xarray uses lazy loading)
    qtr.close()

    return xc, yc, zc

