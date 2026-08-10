import numpy as np
from typing import Tuple, Union, Optional

from src.chapter2 import specific_humidity_to_vp, Wp_from_near_surface_met_data
from src.chapter3 import TOA_incoming_solar
from src.chapter6 import disaggregate_Tair

def optical_depth(
    zenith_rad: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Computes atmospheric optical depth mass from solar zenith angle (radians)."""
    return 1.0 / np.cos(zenith_rad)

def direct_sw_transmissivity(
    Wp: Union[float, np.ndarray],
    gamma_dust: float,
    Mopt: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Computes direct beam shortwave transmissivity (-) via Dingman model."""
    a_sa = -0.124 - 0.0207 * Wp
    b_sa = -0.0682 - 0.0248 * Wp
    tau_sa = np.exp(a_sa + b_sa * Mopt)
    return tau_sa - gamma_dust

def diffuse_sw_scattering_coefficient(
    Wp: Union[float, np.ndarray],
    gamma_dust: float,
    Mopt: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Computes diffuse beam shortwave scattering coefficient (-) via Dingman model."""
    a_s = -0.0363 - 0.0084 * Wp
    b_s = -0.0572 - 0.0173 * Wp
    tau_s = np.exp(a_s + b_s * Mopt)
    return 0.5 * (1.0 - tau_s + gamma_dust)

def clear_sky_shortwave_radiation(
    RsTOA: Union[float, np.ndarray],
    zenith_rad: Union[float, np.ndarray],
    Wp: Union[float, np.ndarray],
    gamma_dust: float,
    albedo: Union[float, np.ndarray],
    surface_pressure: Union[float, np.ndarray],
    model_name: str,
) -> Union[float, np.ndarray]:
    """Computes clear-sky downward shortwave radiation (W/m^2) at the surface.

    Args:
        RsTOA: Top-of-atmosphere shortwave solar flux (W/m^2)
        zenith_rad: Solar zenith angle (radians)
        Wp: Precipitable water (cm)
        gamma_dust: Nondimensional dust coefficient (used in Dingman model)
        albedo: Surface albedo (-)
        surface_pressure: Surface atmospheric pressure (Pa)
        model_name: Model selection string ('dingman' or 'crawford')

    Returns:
        Union[float, np.ndarray]: Downwelling clear-sky shortwave radiation (W/m^2)
    """
    model = model_name.lower()

    if model == "dingman":
        # Optical thickness / air mass
        Mopt = optical_depth(zenith_rad)

        # Direct transmissivity and diffuse scattering coefficient
        t_s = direct_sw_transmissivity(Wp, gamma_dust, Mopt)
        beta = diffuse_sw_scattering_coefficient(Wp, gamma_dust, Mopt)

        attenuation_factor = (
            t_s + beta + beta * albedo * t_s + (beta**2) * albedo
        )
        return RsTOA * attenuation_factor

    elif model == "crawford":
        cos_z = np.cos(zenith_rad)

        # Optical air mass
        m = 35.0 * cos_z * (1224.0 * (cos_z**2) + 1.0) ** (-0.5)

        # Convert surface pressure from Pa to kPa
        p_kPa = surface_pressure / 1000.0

        # Transmission coefficients for Rayleigh scattering, water vapor, and aerosols
        tau_R_tau_pg = 1.021 - 0.084 * np.sqrt(m * (0.00949 * p_kPa + 0.051))
        tau_w = 1.0 - 0.077 * np.power(Wp * m, 0.3)
        tau_a = np.power(0.935, m)

        return RsTOA * tau_R_tau_pg * tau_w * tau_a

    else:
        raise ValueError(
            f"Unsupported clear-sky shortwave model: '{model_name}'. Expected 'dingman' or 'crawford'."
        )

def initialize_model_state(model) -> None:
    """Populates initial 2D physical state variables into the ModelState dataclass."""
    # Unpack model containers
    control = model.control
    params = model.params
    spatial = model.spatial
    forcing = model.forcing
    state = model.state

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
        time_hourly = forcing.time[::steps_per_hour]
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
        ea = specific_humidity_to_vp(qa_hourly, Psfc_hourly, constants.epsilon)
        Wp = Wp_from_near_surface_met_data(
            ea, Ta_hourly, params.precip_water_model_name, constants.T_0, constants.e_s0, constants.Lv, constants.Rv
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

