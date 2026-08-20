import sys

from .terrain import derive_terrain_maps
from .soil import derive_soil_properties
from .channel import derive_channel_properties
from .initialize_model import (
    initialize_model_state,
    record_initial_conditions_time_series,
    record_special_pixel_initial_conditions,
    get_solar_index,
)

def set_static_physical_parameters(model) -> None:
    """Orchestrate all derived static calculations across domain submodules."""
    # Order matters: terrain establishes maskNaN used by downstream modules
    adjust_run_time(model)
    derive_terrain_maps(model)
    derive_soil_properties(model)
    derive_channel_properties(model)

def initialize_dynamic_model_states(model) -> None:
    """Orchestrate state initialization, t=0 output recording, and forcing data pre-processing to get solar index."""
    initialize_model_state(model)
    record_initial_conditions_time_series(model)
    get_solar_index(model)
    # record_special_pixel_initial_conditions(model)

def adjust_run_time(model) -> None:
    # New Feature: Adjust model number of time steps (and time series and map record length)
    # based on length of meteorological record length
    len_forcing_data = model.forcing.time.shape[0]
    
    # check to make sure record has an integer number of days (no partial data for any day)
    if (len_forcing_data * model.control.dt) % 24 != 0:
        sys.ext("Error: meteorological forcing data should not contain incomplete or partial days.")

    # only change control parameters if meteorologcal data record is not 365 days long
    if len_forcing_data != model.control.ntime:
        model.control.ntime = len_forcing_data
        model.control.nt = int(model.control.ntime // model.control.timeseries_frq2store)
        model.control.n_map = int(model.control.ntime // model.control.map_frq2store)
        model.control.n_days = int((len_forcing_data * model.control.dt) // 24)