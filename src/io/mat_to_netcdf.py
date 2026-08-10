import sys
from pathlib import Path
import numpy as np
from scipy import sparse
import xarray as xr
import h5py

# link core repo folder
SCRIPT_DIR = Path(__name__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.chapter1 import load_mat_file

# read static basin data
basin_mat = load_mat_file(REPO_ROOT / "data/MATLAB_mat_files/high_elev_basin_static_data.mat")

# unpack static data

# coordinates
northing = basin_mat["northing"]
easting = basin_mat["easting"]
azimuth = basin_mat["discrete_azimuth_values"]
zenith = basin_mat["discrete_zenith_values"]
dx = basin_mat['dx']
dy = basin_mat['dy']
outlet_coordinate = basin_mat['outlet_coordinate']

# 2D data
aspect = basin_mat["aspect"] 
elev = basin_mat["elev"]
flowacc = basin_mat["flowacc"]
mask = basin_mat["mask"]
slope = basin_mat["slope"]
SVF = basin_mat["SVF"]

# extract sparse matrix
flowdir = basin_mat["flowdir"]
flowdir = flowdir.tocoo() # switch to more intuituve Coordinate (COO) system instead of MATLAB Compressed Sparse Column (CSC) system

# 4D interger matrix
shade = basin_mat["shade_lookup_table"]

# outlet coordinate vectors
outline_x = basin_mat['watershed_outline_coords']['x']
outline_y = basin_mat['watershed_outline_coords']['y']

# attribute data
dx = basin_mat['dx']
dy = basin_mat['dy']
outlet_coordinate = basin_mat['outlet_coordinate']
shade_calc_flag = np.int8(basin_mat['shade_calc_flag'])

# Build the xarray Dataset
static_ds = xr.Dataset(
    
    # Define the variables
    data_vars={
        # 2D map data
        "aspect":   (("northing", "easting"), aspect),
        "elev":     (("northing", "easting"), elev),
        "flowacc":  (("northing", "easting"), flowacc),
        "mask":     (("northing", "easting"), mask),
        "slope":    (("northing", "easting"), slope),
        "SVF":      (("northing", "easting"), SVF),
        
        # flowdir sparse matrix
        "flowdir_data": (("nnz",), flowdir.data),
        "flowdir_row":  (("nnz",), flowdir.row),
        "flowdir_col":  (("nnz",), flowdir.col),
        
        # Shade lookup table ("easting_target" is a dummy variable)
        "shade_lookup_table": (("northing", "easting", "zenith", "azimuth"), shade),
        
        # watershed outline coordinatess (Z is a dummy variable)
        "watershed_outline_x": (("Z",), outline_x),
        "watershed_outline_y": (("Z",), outline_y),
    },
    
    # Define shared coordinates
    coords={
        "northing": ("northing", northing, {"units": "m"}),
        "easting": ("easting", easting, {"units": "m"}),
        "zenith": ("zenith", zenith, {"units": "degrees"}),
        "azimuth": ("azimuth", azimuth, {"units": "degrees"}),
    },
    
    # Scalar and Coordinate Attributes
    attrs={
        "outlet_coordinate": outlet_coordinate,
        "shade_calc_flag": shade_calc_flag,
        "dx": dx,
        "dy": dy,
        "flowdir_shape": 2*[len(northing) * len(easting)]
    }
)

# Save directly to a NetCDF4 file
output_path = REPO_ROOT / "data/preprocessed_inputs/high_elev_basin_static_data.nc"
static_ds.to_netcdf(output_path)

# read meteorological forcing data
met_data = load_mat_file(REPO_ROOT / "data/MATLAB_mat_files/high_elev_met_forcing.mat")

# extract data
time = met_data['time'] # day of year - starts at Oct 1st
Ta = met_data['Ta'] # K
qa = met_data['qa'] # kg/kg
Psfc = met_data['Psfc'] # Pa
U = met_data['U'] # m/s
SW = met_data['SW'] # W/m^2
PPT = met_data['PPT'] # mm/h
gage_elev = met_data['gage_elev'] # m

# build to xarray Dataset
met_forcing_ds = xr.Dataset(
    # The tuple format is: ("dimension_name", numpy_array, {attributes})
    data_vars={
        "Ta":   ("time", Ta,    {"units": "K",      "long_name": "air temperature"}),
        "qa":   ("time", qa,    {"units": "kg/kg",  "long_name": "specific humidity"}),
        "Psfc": ("time", Psfc,  {"units": "Pa",     "long_name": "surface air pressure"}),
        "U":    ("time", U,     {"units": "m/s",    "long_name": "windspeed"}),
        "SW":   ("time", SW,    {"units": "W/m^2",  "long_name": "incoming shortwave radiation"}),
        "PPT":  ("time", PPT,   {"units": "mm/h",   "long_name": "precipitation rate"}),
    },
    # Define the shared 1D dimension coordinate
    coords={
        "time": ("time", time, {"units": "days"})
    },
    # Store the single float parameter (x) as a global attribute
    attrs={
        "gage_elev": gage_elev,
    }
)

# Save directly to a NetCDF4 file
output_path = REPO_ROOT / "data/preprocessed_inputs/high_elev_met_forcing.nc"
met_forcing_ds.to_netcdf(output_path)

# reloaded_ds = xr.open_dataset(REPO_ROOT / "data/preprocessed_inputs/high_elev_met_forcing.nc", decode_timedelta=False)
# reloaded_static = xr.open_dataset(REPO_ROOT / "data/preprocessed_inputs/high_elev_basin_static_data.nc")
