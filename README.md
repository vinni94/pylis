# PyLIS

**PyLIS** is a Python library to read, process, and visualize LIS (Land Information System) model outputs, observation files, and derived products. It provides easy-to-use tools for handling large LIS datasets, generating cubes, computing statistics, and plotting results on geospatial grids.

---

## Features

- **LISReader**
  - Read LIS model outputs (instantaneous or averaged) into xarray DataArrays.
  - Read LIS DA (Data Assimilation) cubes such as increments, innovations, and spread.
  - Read binary LIS observation files (DAOBS) into structured xarray objects.
  - Extract landcover, soil texture, irrigation fraction, and topographic complexity.
  - Compute root-zone values, interquartile range (IQR), autocorrelations, and observation counts.

- **LISplotter**
  - Geospatial plotting of LIS outputs using `matplotlib` and `cartopy`.
  - Customizable plots with lat/lon grids, colorbars, and map features.
  - Compatible with both 2D and 3D LIS data cubes.

---

## Installation

### Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate pylis-env
pip install -e .
