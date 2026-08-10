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
    derive_terrain_maps(model)
    derive_soil_properties(model)
    derive_channel_properties(model)

def initialize_dynamic_model_states(model) -> None:
    """Orchestrate state initialization, t=0 output recording, and forcing data pre-processing to get solar index."""
    initialize_model_state(model)
    record_initial_conditions_time_series(model)
    get_solar_index(model)
    # record_special_pixel_initial_conditions(model)