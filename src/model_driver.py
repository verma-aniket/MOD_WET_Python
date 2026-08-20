# import base libraries
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# To Do:
# 1. Test stream_pixels functionality
# 2. Add special_pixels functionality script

# import data containers & setup routines
from src.model_setup.data_classes import (
    PhysicalConstants, ControlParameters, ModelParameters, 
    SpatialMaps, NetworkTopology, ShadeLookupTable, MetForcingData,
    ModelState, StepVariables, Accumulators, MapOutputs, TimeSeriesOutputs
)
from src.io.data_loader import load_static_basin_data, load_met_forcing_data
from src.model_setup import set_static_physical_parameters, initialize_dynamic_model_states
from src.io.export_results import save_simulation_results_netcdf

# import physics engine modules
from src.physics_engine.met_forcing import step_met_forcing_distribution
from src.physics_engine.snow import step_snow_model
from src.physics_engine.SEB import step_bare_soil_seb
from src.physics_engine.infiltration import step_infiltration
from src.physics_engine.topmodel import step_topmodel
from src.physics_engine.routing import step_routing
from src.physics_engine.mass_balance import step_mass_balance
from src.physics_engine.update_step import step_update_state_and_maps

class MODWETModel:
    """Modular Distributed Watershed Educational Toolbox (MOD-WET) Python Driver."""

    def __init__(self, basin_data_path: str | Path, met_forcing_path: str | Path, 
                 constants_path: Optional[str | Path] = None, parameters_path: Optional[str | Path] = None):
        """Load model constants, basin static data, and meteorological forcing data.

        Parameters
        ----------
        basin_data_path : str | Path
            Path to the NetCDF file containing static basin data.
        met_forcing_path : str | Path
            Path to the NetCDF file containing meteorological forcing data.
        constants_path : str | Path
            Path to the CSV file containing physical scalar constants. 
            Default is None, if valid path is provided, a CSV file is used to set the physical constants.
        parameters_path : str | Path
            Path to the CSV file containing model parameters. 
            Default is None, if valid path is provided, a CSV file is used to set the physical constants.
        """
        
        # 1. Instantiate model variable containers
        self.constants = PhysicalConstants.load(csv_path=constants_path)
        self.control = ControlParameters()
        self.params = ModelParameters.load(csv_path=parameters_path)
        self.spatial = SpatialMaps()
        self.network = NetworkTopology()
        self.shade = ShadeLookupTable()
        self.forcing = MetForcingData()

        # 2. Load in data from NetCDF files
        self._load_data(basin_data_path, met_forcing_path)

        # 3. Perform derived static calculations
        set_static_physical_parameters(self)

        # 4. Instantiate model states, step variables, accumulators, map outputs, and time series outputs
        self.state = ModelState(nx=self.control.nx, ny=self.control.ny)
        self.step_vars = StepVariables(nx=self.control.nx, ny=self.control.ny)
        self.accumulators = Accumulators(nx=self.control.nx, ny=self.control.ny)
        self.map_outputs = MapOutputs(n_map=self.control.n_map, nx=self.control.nx, ny=self.control.ny)
        self.time_series = TimeSeriesOutputs(nt=self.control.nt, nx=self.control.nx, ny=self.control.ny)

        # 5. Initialize state values
        # Note: need to add record_special_pixel_initial_conditions to model_setup __init__
        initialize_dynamic_model_states(self)

    def _load_data(self, basin_data_path: str | Path, met_forcing_path: str | Path) -> None:
        """Delegate NetCDF loading to the model_setup loader module."""
        load_static_basin_data(basin_data_path, self.control, self.spatial, self.network, self.shade)
        load_met_forcing_data(met_forcing_path, self.forcing)

    def run_simulation(self, output_filepath: str | Path) -> None:
        """Executes the main hydrologic simulation loop and exports NetCDF results.

        Parameters
        ----------
        output_filepath : str | Path
            Destination path for saving the generated NetCDF simulation results.
        """
        # Initialize storage time indices
        t_time = 0
        t_map = 0

        # Main Simulation Loop
        for t in tqdm(range(self.control.ntime), desc="MOD-WET Simulation", mininterval=1.0, maxinterval=5.0):

            # a. Meteorological Forcing Distribution
            PPT0, U0, Ta0, Psfc0, qa0, SW0, LWdown0 = step_met_forcing_distribution(self, step_idx=t, ts_idx=t_time)

            # b. Snow Physics Module
            masksnow, maskSEB = step_snow_model(self, PPT0, U0, Ta0, Psfc0, qa0, SW0, LWdown0)

            # c. Bare Soil Surface Energy Balance
            step_bare_soil_seb(self, maskSEB, SW0, Ta0, qa0, U0, Psfc0, LWdown0, step_idx=t, ts_idx=t_time)

            # d. Surface Soil Infiltration & ET
            f, ETsoil = step_infiltration(self, masksnow, maskSEB, PPT0, step_idx=t, ts_idx=t_time)

            # e. Subsurface Soil Moisture & TOPMODEL
            step_topmodel(self, f=f, ETsoil=ETsoil)

            # f. Overland & Channel Flow Routing
            step_routing(self, step_idx=t, ts_idx=t_time)

            # g. Basin Water Balance Residual Check
            step_mass_balance(self, step_idx=t, PPT0=PPT0)

            # h. Update Dynamic States & Accumulate Map Outputs
            step_update_state_and_maps(self, step_idx=t, map_t_idx=t_map)

            # i. Increment Storage Time Indices
            if (t + 1) % self.control.timeseries_frq2store == 0:
                t_time += 1

            if (t + 1) % self.control.map_frq2store == 0:
                t_map += 1

        # Export Simulation Output Data to NetCDF
        save_simulation_results_netcdf(self, output_filepath)
