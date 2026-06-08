import os
import re
import numpy as np
import xarray as xr
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from netCDF4 import Dataset
from tqdm import tqdm
from functools import partial
import pandas as pd
from scipy.interpolate import griddata

class LISReader:
    def __init__(self, lis_dir, lis_input_file, start, end, n_workers=8, parallel=True, debug = False):
        self.lis_dir = lis_dir
        self.lis_input_file = lis_input_file
        self.start = datetime(int(start[6:10]), int(start[3:5]), int(start[0:2]))
        self.end = datetime(int(end[6:10]), int(end[3:5]), int(end[0:2]))
        print(f"Processing for years between {self.start} to {self.end}")
        self.n_workers = n_workers
        self.parallel = parallel
        self.debug = debug

        # Read lat/lon once
        with Dataset(lis_input_file, "r") as f:
            self.lats = f.variables["lat"][:, :].data
            self.lons = f.variables["lon"][:, :].data
        self.n_lat, self.n_lon = self.lats.shape

    # ------------------------------
    # Generic helpers
    # ------------------------------
    def _get_files(self, pattern_str, subfolder=""):
        pattern = re.compile(pattern_str)
        files = []
        for subdir, _, filenames in os.walk(os.path.join(self.lis_dir, subfolder)):
            if self.debug:
                print("Subdirs are :", subdir)
                print("Filenames are :", filenames)
            for fn in filenames:
                m = pattern.search(fn)
                if m:
                    try:
                        file_dt = datetime.strptime(m.group(1), "%Y%m%d%H%M")
                        if self.start <= file_dt <= self.end:
                            files.append(os.path.join(subdir, fn))
                    except Exception:
                        pass
        return sorted(files)

    def _process_files(self, files, worker_func, desc="Processing files"):
        results = []
        if self.parallel:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(worker_func, f): f for f in files}
                for fut in tqdm(as_completed(futures), total=len(files), desc=desc):
                    res = fut.result()
                    if res is not None:
                        results.append((futures[fut], res))
        else:
            for f in tqdm(files, desc=desc):
                res = worker_func(f)
                if res is not None:
                    results.append((f, res))
        return results

    @staticmethod
    def _read_nc_variable(file_path, varname=None, skip_vars=("lat", "lon", "time"), layers=None):
        """Top-level static method for parallel-safe execution."""
        try:
            with Dataset(file_path, "r") as f:
                if varname:
                    data = f.variables[varname][:].data
                    # Apply layer selection if layers is provided
                    if layers is not None:
                        # Convert layers to zero-based index
                        layers_idx = [l-1 if l>0 else l for l in layers]
                        data = data[layers_idx, ...]  # assuming first axis is layers
                else:
                    data = {v: f.variables[v][:].data for v in f.variables if v not in skip_vars}
            return data
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return None
        
    @staticmethod
    def _extract_datetime_from_filename(file_path, pattern_str):
        fn = os.path.basename(file_path)
        m = re.search(pattern_str, fn)
        if m:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M")
        return None

    @staticmethod
    def _get_spread_varname(file_path, varname_prefix, a="01"):
        """
        Dynamically detect the actual variable name in the spread file.
        Searches for variables matching 'ensspread_<varname_prefix>*_{a}'.
        Returns (varname_found, is_layered) tuple.
        varname_found is returned WITHOUT the 'ensspread_' prefix.
        """
        try:
            with Dataset(file_path, "r") as f:
                for var in f.variables.keys():
                    # Look for ensspread_* variables matching the instance
                    if var.startswith("ensspread_") and var.endswith(f"_{a}"):
                        # Check if varname_prefix is in this variable name
                        if varname_prefix in var:
                            # Check if it's a layered variable (contains "Layer")
                            is_layered = "Layer" in var
                            # Extract the base variable name (without ensspread_ prefix and without Layer X and _a suffix)
                            var_no_prefix = var.replace("ensspread_", "")
                            if is_layered:
                                base_var = var_no_prefix.replace(f"_{a}", "").rsplit(" Layer ", 1)[0]
                            else:
                                base_var = var_no_prefix.replace(f"_{a}", "")
                            return base_var, is_layered
        except Exception as e:
            print(f"Error detecting variable in {file_path}: {e}")
        return None, False

    @staticmethod
    def _read_nc_incr(file_path, varname, layers, a="01"):
        """
        Read LIS EnKF increment files.
        Handles layered variables: anlys_incr_<varname> Layer {l}_{a}
        and non-layered: anlys_incr_<varname>_{a}
        Returns array with shape (n_layers, y, x).
        """
        try:
            with Dataset(file_path, "r") as f:
                # Check if layered
                if layers is not None:
                    test_vname = f"anlys_incr_{varname} Layer {layers[0]}_{a}"
                    is_layered = test_vname in f.variables
                else:
                    is_layered = False

                if is_layered:
                    arrs = []
                    for l in layers:
                        vname = f"anlys_incr_{varname} Layer {l}_{a}"
                        if vname not in f.variables:
                            raise KeyError(f"{vname} not found in {file_path}")
                        arrs.append(f.variables[vname][:].data)
                    data = np.stack(arrs, axis=0)  # (n_layers, y, x)
                else:
                    vname = f"anlys_incr_{varname}_{a}"
                    if vname not in f.variables:
                        raise KeyError(f"{vname} not found in {file_path}")
                    data = f.variables[vname][:].data
                    data = np.expand_dims(data, axis=0)  # (1, y, x)

            return data
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return None

    @staticmethod
    def _read_nc_spread(file_path, varname, layers, a="01"):
        """
        Read LIS EnKF spread / incr / innov files.
        Handles both layered variables (Soil Moisture) and single variables (LAI, etc.).
        varname should be passed WITHOUT the 'ensspread_' prefix (e.g., "Soil Moisture" or "LAI").
        Returns array with shape (layer, y, x) for layered, or (1, y, x) for non-layered.
        """
        try:
            with Dataset(file_path, "r") as f:
                # Check if this is a layered variable (contains "Layer" in variable names)
                is_layered = False
                if layers is not None and len(layers) > 0:
                    test_vname = f"ensspread_{varname} Layer {layers[0]}_{a}"
                    is_layered = test_vname in f.variables
                
                if is_layered and layers is not None:
                    # Read layered variables (Soil Moisture, etc.)
                    arrs = []
                    for l in layers:
                        vname = f"ensspread_{varname} Layer {l}_{a}"
                        if vname not in f.variables:
                            raise KeyError(f"{vname} not found in {file_path}")
                        arrs.append(f.variables[vname][:].data)
                    data = np.stack(arrs, axis=0)
                else:
                    # Read non-layered variable (LAI, etc.)
                    vname = f"ensspread_{varname}_{a}"
                    if vname not in f.variables:
                        raise KeyError(f"{vname} not found in {file_path}")
                    # Add a dummy layer dimension for consistency
                    data = f.variables[vname][:].data
                    data = np.expand_dims(data, axis=0)
                
                return data

        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return None

    @staticmethod
    def _read_single_obs_file(file_path, n_lat, n_lon, rescaled=False, a="01", d="01"):
        """
        Read a single binary observation file using regex to filter correct 'a' and 'd'.
        Returns (obs_date, obs_array) or (None, None) if not valid.
        """
        try:
            filename = os.path.basename(file_path)
            # Regex to match LISDAOBS_<YYYYMMDDHHMM>.a<aa>.d<dd>.<suffix>
            pattern = rf"LISDAOBS_(\d{{12}})\.a{a}\.d{d}\..*"
            m = re.match(pattern, filename)
            if not m:
                print(f"Skipping file (pattern mismatch): {filename}")
                return None, None

            # Extract datetime from filename
            file_dt_str = m.group(1)
            obs_date = datetime.strptime(file_dt_str, "%Y%m%d%H%M")

            # Read binary data
            with open(file_path, 'rb') as fid:
                obs = np.fromfile(fid, dtype='>f4')

            n_grid = n_lat * n_lon
            if len(obs) < n_grid + 2:
                print(f"File too short, filling with NaNs: {filename}")
                obs = np.full(n_grid, np.nan)

            n_sources = len(obs) // (n_grid + 2)

            if n_sources == 1:
                obs = obs[1:-1]
            elif n_sources == 2:
                if not rescaled:
                    obs = obs[:len(obs)//2][1:-1]
                else:
                    obs = obs[len(obs)//2:][1:-1]
            else:
                obs = np.full(n_grid, np.nan)

            obs = np.reshape(obs, (n_lat, n_lon))
            obs[obs == -9999] = np.nan
            return obs_date, obs

        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            obs = np.full((n_lat, n_lon), np.nan)
            return None, obs

    # ------------------------------
    # Static Data Readers
    # ------------------------------
    def landflag(self):
        """
        Return flag indicating whether a grid cell is land or not
        """
        with Dataset(self.lis_input_file, mode="r") as f:
            landflag = f.variables["LANDMASK"][:,:].data

        da = xr.DataArray(
            data=np.array(landflag, dtype=bool),
            dims=["x", "y"],
            coords=dict(
                lon=(["x", "y"], self.lons),
                lat=(["x", "y"], self.lats),
            ),
        )
        return da


    def landcover(self, majority=True, classification_system="IGBP"):
        if classification_system == "IGBP":
            sfctypes = [
                'Evergreen Needleleaf Forest', 'Evergreen Broadleaf Forest',
                'Deciduous Needleleaf Forest', 'Deciduous Broadleaf Forest',
                'Mixed Forests', 'Closed Shrublands', 'Open Shrublands',
                'Woody Savannas', 'Savannas', 'Grasslands', 'Permanent wetlands',
                'Croplands', 'Urban and Built-Up', 'cropland/natural vegetation mosaic',
                'Snow and Ice', 'Barren or Sparsely Vegetated', 'Water',
                'Wooded Tundra', 'Mixed Tundra', 'Barren Tundra'
            ]
        else:
            raise Exception("This classification system is not yet implemented.")

        with Dataset(self.lis_input_file, mode="r") as f:
            lc = f.variables["LANDCOVER"][:,:,:].data

        n_types, n_lat, n_lon = lc.shape

        if majority:
            lc = np.argmax(lc, axis=0)
            lc_string = np.empty(lc.shape, dtype="<U100")
            for i in range(n_lat):
                for j in range(n_lon):
                    lc_string[i, j] = sfctypes[lc[i, j]]

            da = xr.DataArray(
                data=lc_string,
                dims=["x", "y"],
                coords=dict(
                    lon=(["x", "y"], self.lons),
                    lat=(["x", "y"], self.lats),
                ),
            )
        else:
            da = xr.DataArray(
                data=lc,
                dims=["sfctype", "x", "y"],
                coords=dict(
                    sfctype=sfctypes,
                    lon=(["x", "y"], self.lons),
                    lat=(["x", "y"], self.lats),
                ),
            )
        return da


    def soiltexture(self, majority=True, classification_system="STATSGO"):
        if classification_system == "STATSGO":
            soiltypes = [
                'SAND','LOAMY SAND','SANDY LOAM','SILT LOAM','SILT','LOAM',
                'SANDY CLAY LOAM','SILTY CLAY LOAM','CLAY LOAM','SANDY CLAY',
                'SILTY CLAY','CLAY','ORGANIC MATERIAL','WATER','BEDROCK',
                'OTHER(land-ice)','PLAYA','LAVA','WHITE SAND'
            ]
        else:
            raise Exception("This classification system is not yet implemented.")

        with Dataset(self.lis_input_file, mode="r") as f:
            tc = f.variables["TEXTURE"][:,:,:].data

        n_types, n_lat, n_lon = tc.shape

        if majority:
            tc = np.argmax(tc, axis=0)
            tc_string = np.empty(tc.shape, dtype="<U100")
            for i in range(n_lat):
                for j in range(n_lon):
                    tc_string[i,j] = soiltypes[tc[i,j]]

            da = xr.DataArray(
                data=tc_string,
                dims=["x", "y"],
                coords=dict(
                    lon=(["x", "y"], self.lons),
                    lat=(["x", "y"], self.lats),
                ),
            )
        else:
            da = xr.DataArray(
                data=tc,
                dims=["soiltype", "x", "y"],
                coords=dict(
                    soiltype=soiltypes,
                    lon=(["x", "y"], self.lons),
                    lat=(["x", "y"], self.lats),
                ),
            )
        return da


    def irrigfrac(self):
        with Dataset(self.lis_input_file, mode="r") as f:
            irrig = f.variables["IRRIGFRAC"][:,:].data

        da = xr.DataArray(
            data=np.array(irrig, dtype='>f4'),
            dims=["x", "y"],
            coords=dict(
                lon=(["x", "y"], self.lons),
                lat=(["x", "y"], self.lats),
            ),
        )
        return da
    
    def topo_complexity(self, topo_complexity_file="/dodrio/scratch/projects/2022_200/project_input/rsda/l_data/obs_satellite/ESA_CCI_SM/ancillary/ESACCI-SOILMOISTURE-TOPOGRAPHIC_COMPLEXITY_V01.1.nc"):
        """
        Read in topographic complexity and interpolate to LIS grid.

        :param str topo_complexity_file: NetCDF file with topographic complexity (tested for 0.25° regular grid).
        :return: xarray.DataArray of topo complexity on LIS grid (dims ["x","y"]).

        Notes:
        - The LIS grid may be projected (Lambert Conformal), so 2D interpolation is required.
        - Missing values (-9999) are converted to NaN.
        """

        # --- Obtain LIS domain coordinates ---
        with Dataset(self.lis_input_file, mode="r") as f:
            lats = f.variables["lat"][:,:].data  # 2D array
            lons = f.variables["lon"][:,:].data  # 2D array

        # --- Read topo complexity file ---
        with Dataset(topo_complexity_file) as f:
            tc = f.variables["topographic_complexity"][:,:].data
            lat_map = f.variables["lat"][:].data
            lon_map = f.variables["lon"][:].data

        # Replace fill value with NaN
        tc[tc == -9999] = np.nan

        # --- Flatten topo grid for interpolation ---
        lon_grid, lat_grid = np.meshgrid(lon_map, lat_map)
        points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        values = tc.ravel()

        # --- Interpolate to 2D LIS grid using nearest neighbor ---
        tc_interp = griddata(
            points=points,
            values=values,
            xi=(lats, lons),
            method='nearest'
        )

        # --- Wrap as xarray DataArray ---
        tc_da = xr.DataArray(
            data=tc_interp,
            dims=["x", "y"],
            coords=dict(
                lat=(["x", "y"], lats),
                lon=(["x", "y"], lons)
            )
        )

        return tc_da

    # ------------------------------
    # Cube readers
    # ------------------------------
    def read_innov_cube(self, varname="innov", subfolder="EnKF", a="01", d="01", freq=None):
        pattern_str = rf"LIS_DA_EnKF_(\d{{12}})_innov\.a{a}\.d{d}\.nc"
        files = self._get_files(pattern_str, subfolder=subfolder)
        var_name = f"{varname}_{a}"

        # Use top-level static method
        worker = partial(LISReader._read_nc_variable, varname=var_name)
        results = self._process_files(files, worker, desc="Reading innovation files")
        results.sort(key=lambda x: LISReader._extract_datetime_from_filename(x[0], pattern_str))

        dates = [LISReader._extract_datetime_from_filename(f, pattern_str) for f, _ in results]
        innov_data = np.stack([np.squeeze(r) for _, r in results])
        innov_data[innov_data == -9999] = np.nan

        dc = xr.DataArray(
            data=innov_data,
            dims=["time", "x", "y"],
            coords=dict(lon=(["x", "y"], self.lons),
                        lat=(["x", "y"], self.lats),
                        time=np.array(dates)),
            attrs=dict(description="LIS innovations", variable=var_name)
        )
        if freq:
            dc = dc.sortby("time").resample(time=freq).mean()
        return dc
    
    def read_incr_cube(self, subfolder="EnKF", varname="Soil Moisture",
                   layers=[1, 2, 3, 4], a="01", d="01", freq=None):
        """
        Read LIS EnKF increment files and return an xarray.DataArray.
        Variable naming: anlys_incr_<varname> Layer {l}_{a}  (layered)
                    or anlys_incr_<varname>_{a}             (non-layered)
        Auto-detects whether the variable is layered from the first file found.
        """
        pattern_str = rf"LIS_DA_EnKF_(\d{{12}})_incr\.a{a}\.d{d}\.nc"
        files = self._get_files(pattern_str, subfolder=subfolder)

        if len(files) == 0:
            raise FileNotFoundError("No increment files found.")

        # --- Auto-detect layering from the first file ---
        with Dataset(files[0], "r") as f:
            test_vname = f"anlys_incr_{varname} Layer {layers[0]}_{a}"
            is_layered = test_vname in f.variables

        if is_layered:
            layers_to_use = layers
            n_layers = len(layers)
        else:
            layers_to_use = None  # signal to _read_nc_incr to skip layer loop
            n_layers = 1

        print(f"Variable '{varname}' (a={a}): layered={is_layered}, n_layers={n_layers}")

        worker = partial(LISReader._read_nc_incr, varname=varname, layers=layers_to_use, a=a)
        results = self._process_files(files, worker, desc="Reading increment files")
        results.sort(key=lambda x: LISReader._extract_datetime_from_filename(x[0], pattern_str))

        times = [LISReader._extract_datetime_from_filename(f, pattern_str) for f, _ in results]
        n_time = len(results)

        data_cube = np.full((n_time, n_layers, self.n_lat, self.n_lon), np.nan)
        for i, (_, arr) in enumerate(results):
            if arr is not None:
                arr = np.array(arr, dtype=float)
                arr[arr == -9999] = np.nan
                data_cube[i] = arr

        da = xr.DataArray(
            data=data_cube,
            dims=["time", "layer", "x", "y"],
            coords=dict(
                lon=(["x", "y"], self.lons),
                lat=(["x", "y"], self.lats),
                layer=layers_to_use if is_layered else [1],
                time=times,
            ),
            attrs=dict(description="LIS analysis increments", variable=varname,
                    is_layered=is_layered),
        )

        # Squeeze out dummy layer dim for non-layered variables
        if not is_layered:
            da = da.sel(layer=1)

        if freq:
            da = da.sortby("time").resample(time=freq).mean()

        return da
        
    def spread_cube(self, subfolder="EnKF", varname="Soil Moisture", layers=[1,2,3,4],
                    a="01", d="01", h=0, freq="1D", start=None, end=None):
        """
        Read LIS ensemble spread files and return an xarray.DataArray.
        Automatically detects variable naming and whether to read layers.
        - For variables containing 'Soil Moisture', reads the specified layers
        - For other variables (LAI, etc.), reads the single variable without layers
        """
        # --- Get LIS input lat/lon from reference file ---
        with Dataset(self.lis_input_file, "r") as f:
            lats = f.variables["lat"][:].data
            lons = f.variables["lon"][:].data

        n_lat, n_lon = lats.shape

        # --- Construct file pattern ---
        pattern_str = rf"LIS_DA_EnKF_(\d{{12}})_spread\.a{a}\.d{d}\.nc"
        files = self._get_files(pattern_str, subfolder=subfolder)
        
        if len(files) == 0:
            raise FileNotFoundError("No spread files found.")

        # --- Sort files by datetime ---
        files.sort(key=lambda f: self._extract_datetime_from_filename(f, pattern_str))

        # --- Detect variable name and whether it has layers from first file ---
        actual_varname, is_layered = self._get_spread_varname(files[0], varname, a=a)
        
        if actual_varname is None:
            raise ValueError(f"Could not find variable matching '{varname}' in {files[0]}")
        
        print(f"Detected variable: {actual_varname}, Layered: {is_layered}")
        
        # Determine number of layers and which layers to use
        if is_layered and layers is not None and len(layers) > 0:
            n_layers = len(layers)
            layers_to_read = layers
        elif is_layered:
            # Default to reading what's available
            n_layers = 1
            layers_to_read = [1]
        else:
            # Non-layered variable
            n_layers = 1
            layers_to_read = None

        # --- Worker function ---
        worker = partial(self._read_nc_spread, varname=actual_varname, layers=layers_to_read, a=a)

        # --- Process files ---
        results = self._process_files(files, worker, desc="Reading spread files")
        results.sort(key=lambda x: self._extract_datetime_from_filename(x[0], pattern_str))
        times = [self._extract_datetime_from_filename(f, pattern_str) for f, _ in results]
        
        # --- Allocate cube ---
        data_cube = np.full((len(files), n_layers, n_lat, n_lon), np.nan)

        for i, (_, arr) in enumerate(results):
            if arr is not None:
                arr = np.array(arr, dtype=float)
                arr[arr == -9999] = np.nan
                data_cube[i] = arr

        # --- Build xarray ---
        if is_layered and layers_to_read is not None:
            layer_coords = layers_to_read
        else:
            layer_coords = [1]
        
        da = xr.DataArray(
            data=data_cube,
            dims=["time", "layer", "x", "y"],
            coords=dict(
                lon=(["x","y"], lons),
                lat=(["x","y"], lats),
                layer=layer_coords,
                time=times,
            ),
            attrs=dict(description="LIS model spread", variable=varname, is_layered=is_layered),
        )

        if n_layers == 1 and len(layer_coords) == 1:
            da = da.sel(layer=layer_coords[0])

        # --- Resample if requested ---
        if freq:
            da = da.resample(time=freq).mean()

        # --- Filter by start/end date ---
        if start is not None and end is not None:
            start_dt = pd.to_datetime(start, dayfirst=True)
            end_dt = pd.to_datetime(end, dayfirst=True)
            da = da.sel(time=slice(start_dt, end_dt))

        return da

    def read_lis_cube(self, subfolder="SURFACEMODEL", varname=None, freq="1D", d="01", date_shift=False):
        """
        Read LIS files and return an xarray.DataArray.
        - Dynamically detects dimensions (time, x/y, depth/layer if present)
        - Behaves like lis_cube: reads lat/lon from LIS input file, stacks files into a time series
        - Handles missing files gracefully
        - Uses self._process_files for parallel execution with progress bar
        """
        # warnings related to the date_shift option
        if ("_tavg" in varname) and (not date_shift):
            print("It is recommended to set date_shift=True for averaged outputs.")
            
        elif ("_inst" in varname) and (date_shift):
            print("It is recommended to set date_shift=False for instantaneous outputs.")

        # --- Construct file list based on patterns YYYYMMDDHHMM ---
        pattern_str = rf"LIS_HIST_(\d{{12}})\.d{d}\..*"
        files = self._get_files(pattern_str, subfolder=subfolder)

        if len(files) == 0:
            raise FileNotFoundError("No LIS history files found in the specified folder.")

        # Sort and extract times
        files.sort(key=lambda f: self._extract_datetime_from_filename(f, pattern_str))
        times = [self._extract_datetime_from_filename(f, pattern_str) for f in files]

        if date_shift:
            times = [t - pd.Timedelta(freq) for t in times]

        # --- Worker function for reading a single file ---
        worker = partial(self._read_nc_variable, varname=varname)

        # --- Process files in parallel/serial using the existing helper ---
        results = self._process_files(files, worker, desc="Reading LIS history files")

        # Sort results by datetime
        results.sort(key=lambda x: self._extract_datetime_from_filename(x[0], pattern_str))
        times = [self._extract_datetime_from_filename(f, pattern_str) for f, _ in results]

        print("Sorted results based on extracted datetime from filenames.")

        # --- Read first file to infer dimensions ---
        sample_data = results[0][1]  # results: list of (filename, data_array)

        n_lat, n_lon = self.lats.shape
        n_time = len(times)

        # Detect number of layers
        if sample_data.ndim == 2:
            n_layers = 1
            data_cube = np.full((n_time, n_lat, n_lon), np.nan)
        elif sample_data.ndim == 3:
            n_layers = sample_data.shape[0]
            data_cube = np.full((n_time, n_layers, n_lat, n_lon), np.nan)
        else:
            raise ValueError(f"Unexpected number of dimensions ({sample_data.ndim}) in variable {varname}")

        # --- Fill data_cube with results ---
        for i, (_, arr) in enumerate(results):
            arr = np.array(arr, dtype=float)
            arr[arr == -9999] = np.nan
            if varname in ["SoilMoist_inst", "SoilMoist_tavg"]:
                arr[arr > 1] = np.nan
            data_cube[i] = arr

        # --- Convert to xarray ---
        if n_layers == 1:
            da = xr.Dataset(data_vars = {varname: (["time", "x", "y"], data_cube)},
                coords=dict(
                    lon=(["x", "y"], self.lons),
                    lat=(["x", "y"], self.lats),
                    time=times,
                ),
                attrs=dict(description="LIS model output", variable=varname),
            )
        else:
            da = xr.Dataset(
                            data_vars={
                            varname: (["time", "layer", "x", "y"], data_cube)
                            },
                            coords={
                            "lon": (["x", "y"], self.lons),
                            "lat": (["x", "y"], self.lats),
                            "layer": [i + 1 for i in range(n_layers)],
                            "time": times
                        },
                        attrs=dict(description="LIS model output", variable=varname)
                    )

        # Optional resampling
        if freq:
            da = da.resample(time=freq).mean()

        return da
        
    def read_irr_cube(self, subfolder="IRRIGATION", varname=None, freq=None, date_shift = False, irrfrac_threshold = None):
        """
        Read LIS irrigation files.
        """
        pattern_str = r"LIS_HIST_(\d{12})"
        files = self._get_files(pattern_str, subfolder=subfolder)

        worker = partial(self._read_nc_variable, varname=varname)
        results = self._process_files(files, worker, desc="Reading irrigation files")
        results.sort(key=lambda x: self._extract_datetime_from_filename(x[0], pattern_str))
        times = [self._extract_datetime_from_filename(f, pattern_str) for f, _ in results]
        
        if date_shift:
            times = [t - pd.Timedelta(days = 1) for t in times]

        # Create mask if threshold is provided
        if irrfrac_threshold is not None:
            irrfrac_da = self.irrigfrac()  # shape: (x, y)
            mask = irrfrac_da > irrfrac_threshold
        else:
            mask = None

        if varname:
            arr = np.stack([r for _, r in results])
            arr[arr == -9999] = np.nan
            if mask is not None:
                arr = np.where(mask, arr, np.nan)  # broadcast mask across time
            da = xr.DataArray(
                data=arr,
                dims=["time", "x", "y"],
                coords=dict(lon=(["x", "y"], self.lons),
                            lat=(["x", "y"], self.lats),
                            time=np.array(times)),
                attrs=dict(description=f"Irrigation variable {varname}")
            )
            if freq:
                da = da.resample(time=freq).mean()
            return da
        else:
            all_vars = list(results[0][1].keys())
            data_dict = {}
            for v in all_vars:
                if ("_tavg" in v) and (not date_shift):
                    print(" It is recommended to keep date_shift as true for tavg vars")
                arr = np.stack([r[v] for _, r in results])
                arr[arr == -9999] = np.nan
                if mask is not None:
                    arr = np.where(mask, arr, np.nan)  # broadcast mask across time
                data_dict[v] = (["time", "x", "y"], arr)
            ds = xr.Dataset(data_vars=data_dict,
                            coords=dict(lon=(["x", "y"], self.lons),
                                        lat=(["x", "y"], self.lats),
                                        time=np.array(times)))
            if freq:
                ds = ds.resample(time=freq).mean()
            return ds

    def read_obs_cube(self, subfolder="DAOBS", a="01", d="01", rescaled=False, freq=None):
        """
        Read LIS observation binary files as xarray DataArray.
        Uses regex for file matching and reuses the _process_files helper.
        """
        # Regex to match filenames like LISDAOBS_YYYYMMDDHHMM_a01_d01
        pattern_str = rf"LISDAOBS_(\d{{12}})\.a{a}\.d{d}\..*"
        # pattern_str = rf"LISDAOBS_(\d{{12}})_.*\.a{a}\.d{d}\..*"

        # Gather all matching files within the date range
        files = self._get_files(pattern_str, subfolder=subfolder)

        # Worker function with pre-filled arguments
        worker = partial(LISReader._read_single_obs_file,
                        n_lat=self.n_lat,
                        n_lon=self.n_lon,
                        rescaled=rescaled,
                        a=a,
                        d=d)

        # Process files in parallel or serial using the common helper
        results = self._process_files(files, worker, desc="Reading observation files")
        results.sort(key=lambda x: self._extract_datetime_from_filename(x[0], pattern_str))
        print("Sorted results based on extracted datetime from filenames.")

        # Extract dates and arrays
        obs_dates = []
        obs_arrays = []
        for f, (date, obs) in results:
            if date is not None and obs is not None:
                obs_dates.append(date)
                obs_arrays.append(obs)

        # Ensure consistent shapes
        shapes = [arr.shape for arr in obs_arrays]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent shapes in obs arrays: {set(shapes)}")

        # Convert to xarray
        obs_cube = xr.DataArray(
            data=np.stack(obs_arrays),
            dims=["time", "x", "y"],
            coords=dict(
                lon=(["x","y"], self.lons),
                lat=(["x","y"], self.lats),
                time=np.array(obs_dates)
            ),
            attrs=dict(description="Observations from binary DAOBS files")
        )

        obs_cube = obs_cube.sortby("time")
        if freq:
            obs_cube = obs_cube.resample(time=freq).mean()

        return obs_cube

