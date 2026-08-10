from pathlib import Path
import xarray as xr
import scipy.sparse as sp

def load_static_basin_data(file_path: str | Path, control, spatial, network, shade) -> None:
    """Read basin NetCDF file and populate spatial, and shade attributes directly."""

    # Read Data
    if not file_path.exists():
        raise FileNotFoundError(f"NetCDF file not found at: {file_path.resolve()}")
    ds = xr.open_dataset(file_path)

    # Extract and load data

    # Coordinate Data
    spatial.northing = ds['northing'].values
    spatial.easting = ds['easting'].values
    control.dx = ds.dx
    control.dy = ds.dy
    control.nx = len(ds['northing'].values)
    control.ny = len(ds['easting'].values)
    spatial.outlet_coordinate = ds.outlet_coordinate

    # Map/Grid Data
    spatial.aspect_deg = ds['aspect'].values
    spatial.elev = ds['elev'].values
    spatial.flowacc = ds['flowacc'].values
    spatial.mask = ds['mask'].values
    spatial.slope_deg = ds['slope'].values
    spatial.SVF = ds['SVF'].values

    # load sparse matrix data
    flowdir_data = ds['flowdir_data'].values
    flowdir_row = ds['flowdir_row'].values
    flowdir_col = ds['flowdir_col'].values
    flowdir_shape = ds.flowdir_shape
    # Create COO matrix
    network.flowdir = sp.coo_matrix((flowdir_data, (flowdir_row, flowdir_col)), shape=flowdir_shape)

    # load (or not load) shade lookup table data based on calculation flag
    if ds.shade_calc_flag == 1:
        shade.shade_calc_flag = True
        shade.shade_table = ds['shade_lookup_table'].values
        shade.azimuth = ds['azimuth'].values        
        shade.zenith = ds['zenith'].values
        

def load_met_forcing_data(file_path: str | Path, forcing) -> None:
    """Read meteorologic forcing NetCDF file and populate forcing attributes directly."""

    # Read Data
    if not file_path.exists():
        raise FileNotFoundError(f"NetCDF file not found at: {file_path.resolve()}")
    ds = xr.open_dataset(file_path, decode_timedelta=False)

    # Extract and load data

    # Time index
    forcing.time = ds['time'].values

    # Gage Elevation
    forcing.gage_elev = ds.gage_elev

    # Meteorological Inputs
    forcing.Ta = ds['Ta'].values
    forcing.qa = ds['qa'].values
    forcing.Psfc = ds['Psfc'].values
    forcing.U = ds['U'].values
    forcing.SW = ds['SW'].values
    forcing.PPT = ds['PPT'].values
    
