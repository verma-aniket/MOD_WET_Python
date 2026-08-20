import numpy as np

from src.chapter2 import specific_humidity_to_vp, Wp_from_near_surface_met_data
from src.chapter3 import TOA_incoming_solar, clear_sky_shortwave_radiation
from src.chapter6 import disaggregate_Tair

def initialize_model_state(model) -> None:
    """Populates initial 2D physical state variables into the ModelState dataclass."""
    # Unpack model containers
    control = model.control
    params = model.params
    spatial = model.spatial
    forcing = model.forcing
    state = model.state
    step_vars = model.step_vars

    nx, ny = control.nx, control.ny
    mask = spatial.maskNaN

    # Days since last major snowfall (-)
    state.NDayLastSnow = np.zeros((nx, ny), dtype=np.float64) * mask

    # Initial Saturation Deficit (m)
    sd0 = (params.SDmean0 + params.m * (params.lambda_mean - spatial.lambda_map) * mask)
    sd0[sd0 < 0.0] = 0.0
    state.SD = sd0

    # Initial Root Zone Storage (m)
    state.Srz = ((spatial.Srzmax + spatial.Srzmin) / 2.0 * mask)

    # Initial Unsaturated Zone Storage (m)
    state.Suz = np.zeros((nx, ny), dtype=np.float64) * mask

    # Initial Surface Temperature (K) — unpack Tair_disagg, ignore Tair_mean
    state.Tsurf, _ = disaggregate_Tair(forcing.Ta[0], spatial.elev, forcing.gage_elev, params.LapseRateTair)

    # Initial Deep Soil Temperature (K) — unpack Tair_disagg, ignore Tair_mean
    state.Td, _ = disaggregate_Tair(float(np.mean(forcing.Ta)), spatial.elev, forcing.gage_elev, params.LapseRateTair)

    # Initial Snow Conditions
    state.SWE = np.zeros((nx, ny), dtype=np.float64) * mask
    state.snowdens = np.zeros((nx, ny), dtype=np.float64) * mask
    state.snowdepth = np.zeros((nx, ny), dtype=np.float64) * mask
    state.snowfrac = np.zeros((nx, ny), dtype=np.float64) * mask

    # Initialize step variables
    step_vars.albedo_out = spatial.albedo.copy()
    step_vars.Tsurf_new = mask.copy()
    step_vars.SWE_new = mask.copy()
    step_vars.Td_new = mask.copy()
    step_vars.snowdens_new = mask.copy()
    step_vars.snowdepth_new = mask.copy()
    step_vars.snowfrac_new = mask.copy()

def record_initial_conditions_time_series(model) -> None:
    """Populates index 0 (t=0) of TimeSeriesOutputs with basin-averaged initial state values."""
    # Unpack model containers
    state = model.state
    time_series = model.time_series

    time_series.Tsurf[0] = np.nanmean(state.Tsurf)
    time_series.Srz[0] = np.nanmean(state.Srz)
    time_series.Suz[0] = np.nanmean(state.Suz)
    time_series.SD[0] = np.nanmean(state.SD)
    time_series.SWE[0] = (np.nanmean(state.SWE) / 1000.0)  # Converted to meters as in MATLAB
    time_series.snowdens[0] = np.nanmean(state.snowdens)
    time_series.snowdepth[0] = (np.nanmean(state.snowdepth) / 1000.0)  # Converted to meters as in MATLAB
    time_series.snowfrac[0] = np.nanmean(state.snowfrac)
    time_series.Td[0] = np.nanmean(state.Td)

def record_special_pixel_initial_conditions(model) -> None:
    """Populates index 0 (t=0) of special pixel time-series arrays with initial state values."""
    # Unpack model containers
    params = model.params
    state = model.state
    time_series = model.time_series

    if time_series.n_special_pixels > 0 and params.special_pixels is not None:
        pixels = params.special_pixels  # Coordinate tuple (rows, cols)

        time_series.pixel_Tsurf[0][pixels] = state.Tsurf[pixels]
        time_series.pixel_Srz[0][pixels] = state.Srz[pixels]
        time_series.pixel_Suz[0][pixels] = state.Suz[pixels]
        time_series.pixel_SD[0][pixels] = state.SD[pixels]
        time_series.pixel_SWE[0][pixels] = state.SWE[pixels]
        time_series.pixel_snowdens[0][pixels] = state.snowdens[pixels]
        time_series.pixel_snowdepth[0][pixels] = state.snowdepth[pixels]
        time_series.pixel_snowfrac[0][pixels] = state.snowfrac[pixels]
        time_series.pixel_Td[0][pixels] = state.Td[pixels]

def get_solar_index(model) -> None:
    """Forcing data preprocessing to get the daily solar index used in the cloudy-sky longwave model."""
    # Unpack model containers
    constants = model.constants
    control = model.control
    params = model.params
    forcing = model.forcing

    if params.cloudy_sky_atmos_emiss_model is not None:
        # Number of time steps per hour
        steps_per_hour = int(1.0 / control.dt)

        # 1. Downsample forcing vectors to hourly resolution
        time_hourly = forcing.DOY[::steps_per_hour]
        SW_hourly = forcing.SW[::steps_per_hour]
        qa_hourly = forcing.qa[::steps_per_hour]
        Ta_hourly = forcing.Ta[::steps_per_hour]
        Psfc_hourly = forcing.Psfc[::steps_per_hour]

        # 2. Vectorized Day-of-Year and UTC time calculations (1D arrays)
        DOY = np.floor(time_hourly)         # days
        UTC = (time_hourly - DOY) * 24.0    # hours

        # 3. Vectorized solar geometry & TOA shortwave flux
        RsTOA, zenith_deg, _, _, _, _, _ = TOA_incoming_solar(
            DOY,
            UTC,
            params.time_zone_shift,
            params.lat_mean,
            params.lon_mean,
            constants.S0,
        )

        # 4. Vectorized vapor pressure & precipitable water
        ea = specific_humidity_to_vp(qa_hourly, Psfc_hourly, epsilon=constants.epsilon)
        Wp = Wp_from_near_surface_met_data(
            ea, Ta_hourly, params.precip_water_model_name, T_0=constants.T_0, e_s0=constants.e_s0, Lv=constants.Lv, Rv=constants.Rv
        )

        # 5. Vectorized downwelling clear-sky shortwave radiation
        zenith_rad = np.radians(zenith_deg)
        Rs_down_clear = clear_sky_shortwave_radiation(
            RsTOA,
            zenith_rad,
            Wp,
            params.gamma_dust,
            params.albedo,
            Psfc_hourly,
            params.clear_sky_shortwave_model_name,
        )

        # Compute daily average clear-sky shortwave radiation (n_days,)
        Rs_down_clear_daily = Rs_down_clear.reshape(-1, 24).mean(axis=1)

        # Compute daily averaged incoming shortwave radiation (n_days,)
        SW_daily = SW_hourly.reshape(-1, 24).mean(axis=1)

        # Daily solar index ratio
        solar_index_daily = SW_daily / Rs_down_clear_daily

        # Broadcast daily values across all time steps within each day
        steps_per_day = int(24.0 / control.dt)
        forcing.solar_index = np.repeat(solar_index_daily, steps_per_day)

    else:
        # Set solar_index to 1.0 everywhere (same shape and dtype as forcings.SW)
        forcing.solar_index = np.ones_like(forcing.SW)

