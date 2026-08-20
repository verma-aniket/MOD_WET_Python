import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.chapter2 import air_density, sat_vapor_pressure, sat_vapor_pressure_ice, specific_humidity_to_vp, vp_to_dew_point_temperature, dew_point_temperature_to_vp, vp_to_specific_humidity
from src.chapter3 import TOA_incoming_solar, cloudy_sky_emiss
from src.chapter8 import aero_resistance, mass_transfer, richardson_number, stab_corr_factors

def albedo_usace(T_air: np.ndarray, day_counter: np.ndarray, T_f: float, alpha_0: float = 0.4,  K: float = 0.44,
                 ) -> np.ndarray:
    """
    Description: This function calculates albedo using USACE (1956) formulation:
    albedo = alpha_0+K.*exp(-day_counter * r)
    
    where:
        - alpha_0= min snowpack albedo ~0.4
        - K=constant, ~0.44 [K and alpha_0 define the maximum albedo of fresh snow]
        - day_counter=number of days since the last snowfall
        - r =recession coefficients 0.12 for Ta>T_f and 0.05 for Ta<T_f
    Calculates snow albedo using the USACE (1956) formulation."""
    r = np.where(T_air <= T_f, -0.05, -0.12)
    return alpha_0 + K * np.exp(r * day_counter)

def diagnostic_snow_density(density: np.ndarray, SWE: np.ndarray, dt: float,  snowfall: np.ndarray, 
                            min_density: float=100, max_density: float=450, tau_f: float=0.24, tau_1: float=86400
                            ) -> np.ndarray:
    """
    Description: This function calculates snow density evolution in a simple way. 
    
    The equations below are taken from: Dutra et al., 2010: An improved snow scheme for the ECMWF land surface
    model: Description and offline validation, JHM, doi: 10.1175/2010JHM1249.1.
    Inputs:
        density: snow density at previous time step
        SWE: SWE at previous time step
        dt: time step
        snowfall: snowfall occurring over time step
    
    Outputs:
        snow_density: snow density at new time step

    Defaults:
        min_density=100; % minimum (new snow) density (kg/m^3)
        max_density=450; % maximum snow density (kg/m^3)
        tau_f=0.24; % exponential time scale coefficient 
        tau_1=86400; % timescale (seconds)
    """
    total_snowfall = snowfall * dt  # snowfall over timestep (mm)
    dt_sec = dt * 3600.0            # time step in seconds

    # NEW
    denom = SWE + total_snowfall
    numerator = SWE * density + total_snowfall * min_density

    rho_star = np.divide(
        numerator,
        denom,
        out=np.full_like(denom, min_density),
        where=denom > 0.0
    )

    # OLD
    # denom = SWE + total_snowfall
    # rho_star = np.where(
    #     denom > 0.0,
    #     (SWE * density + total_snowfall * min_density) / denom,
    #     min_density
    # )

    return (rho_star - max_density) * np.exp(-tau_f * dt_sec / tau_1) + max_density

def diagnostic_snow_fraction(snow_depth: np.ndarray, h_soil: float | np.ndarray) -> np.ndarray:
    """Calculates fractional snow-covered area based on the BATS model."""
    return snow_depth / (snow_depth + h_soil)

def disaggregate_Tair(Tair: float | np.ndarray, elevationPixel: np.ndarray, gage_elev: float, LapseRateTair: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Disaggregates air temperature across grid cells using elevation and lapse rate.

    Returns:
        tuple[np.ndarray, np.ndarray]: (Tair_disagg, Tair_mean)
    """

    Tair_disagg = Tair + LapseRateTair * (elevationPixel - gage_elev)
    Tair_mean = (Tair + Tair_disagg) / 2.0
    return Tair_disagg, Tair_mean

def disaggregate_PPT(PPT: np.ndarray, elevation_pixel: np.ndarray, gage_elev: float, lapse_rate_ppt: float
                     ) -> np.ndarray:
    """Disaggregates precipitation field based on elevation lapse rate."""
    dz = (elevation_pixel - gage_elev) / 1000.0  # m -> km
    factor = lapse_rate_ppt * dz
    return PPT * (1.0 + factor) / (1.0 - factor)

def disaggregate_press(press: np.ndarray, tair_mean: np.ndarray, elevation_pixel: np.ndarray, gage_elev: float, g: float, Rd: float
                       ) -> np.ndarray:
    """Disaggregates surface pressure using the barometric formula."""
    dz = elevation_pixel - gage_elev
    return press * np.exp(-g * dz / (Rd * tair_mean))

def disaggregate_qair(qa: np.ndarray, press: np.ndarray, press_disagg: np.ndarray, elevation_pixel: np.ndarray, gage_elev: float, lapse_rate_tdew: float,
                       epsilon: float, T_0: float, e_s0: float, Lv: float, Rv: float
                       ) -> np.ndarray:
    """Disaggregates specific humidity using dew point temperature lapse rates."""
    e_gage = specific_humidity_to_vp(qa, press, epsilon=epsilon)
    Td_gage = vp_to_dew_point_temperature(e_gage, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
    Td_pix = Td_gage + lapse_rate_tdew * (elevation_pixel - gage_elev)
    e_pix = dew_point_temperature_to_vp(Td_pix, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
    return vp_to_specific_humidity(e_pix, press_disagg, epsilon=epsilon)

def disaggregate_SW(
    SWin: np.ndarray,
    press: np.ndarray,
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    zenith_rad: float,
    azimuth_rad: float,
    hrangle: float,
    shade: np.ndarray,
    SVF: np.ndarray,
    sunrise: float,
    sunset: float,
    RsTOA: float,
    mask: np.ndarray,
    albedo: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    This function disaggregates the incoming shortwave radiation flux at the
    suface by accounting for topographic (slope and aspect) effects.

    Based on disaggregation methods in Allen et al. (2006) and Muller and 
    Scherer (2005) and modified for application in CEE 150.

    Inputs:
    SWin: Incoming shortwave radiation data on a horizontal plane array (W/m^2)
    press: Disaggregated pressure data array (Pa)
    slope_rad: Slope (radians)
    aspect_rad: Aspect (radians)
    zenith_rad: Solar zenith angle (radians)
    azimuth_rad: Solar azimuth angle (radians)
    hrangle: Hour angle (radians)
    shade: Shade array
    SVF: Sky view factor array
    sunrise: Local hour of sunrise
    sunset: Local hour of sunset
    RsTOA: TOA solar flux at central lat/lon (W/m^2)
    mask: Basin mask array
    albedo: albedo map array

    Outputs:
    Rs: Total incident radiation at the surface (W/m^2)
    RsDir: Direct beam solar flux: RsDir (W/m^2)
    RsDif: Diffuse solar flux (W/m^2)

    Note: Can run for a single time over a spatial domain
    """
    nx, ny = mask.shape

    # Assumed parameters
    Gamma_SW = 1.0  # Solar flux correction factor
    press_mean_areal = np.nanmean(press)
    AlbedoSurrounding = np.full((nx, ny), np.nan)
    AlbedoSurrounding[:, :] = albedo  # Surrounding albedo associated with mean areal data

    # Allocate
    tau_SW_hor = np.full((nx, ny), np.nan)
    K_B_hor = np.full((nx, ny), np.nan)
    RsDir = np.full((nx, ny), np.nan)
    fcor = np.full((nx, ny), np.nan)
    RsDif = np.full((nx, ny), np.nan)

    # Perform over mask
    imask = (mask == 1)

    # Cosine of solar zenith angle
    cos_sza_hor = np.cos(zenith_rad)

    # Ensure SWin is a broadcastable array matching spatial grid shape
    SWin = np.array(SWin, dtype=float)
    if SWin.ndim == 0:
        SWin = np.full((nx, ny), SWin)

    # Adjust mean areal SW forcings
    SWin = SWin * Gamma_SW

    ## Disaggregate incoming shortwave radiation at the surface from mean areal data:
    # Partition direct from diffuse flux

    # For clear sky: broadband atmos. transmissivity= Solar insolation horizontal / TOA solar radiation
    if RsTOA == 0.0:
        tau_SW_hor[imask] = 0.0
    else:
        tau_SW_hor[imask] = SWin[imask] / RsTOA

    # Direct beam transmissivity (based on Allen et al. 2006 paper)
    ind = (tau_SW_hor <= 0.175) & (mask == 1)
    K_B_hor[ind] = 0.016 * tau_SW_hor[ind]

    ind = (tau_SW_hor > 0.175) & (tau_SW_hor < 0.42) & (mask == 1)
    K_B_hor[ind] = (0.022 - 0.280 * tau_SW_hor[ind] +
                    0.828 * tau_SW_hor[ind]**2.0 + 0.765 * tau_SW_hor[ind]**3.0)

    ind = (tau_SW_hor >= 0.42) & (mask == 1)
    K_B_hor[ind] = 1.56 * tau_SW_hor[ind] - 0.55

    ind = (K_B_hor > tau_SW_hor) & (mask == 1)
    K_B_hor[ind] = tau_SW_hor[ind]

    # Diffuse beam transmissivity
    K_D_hor = tau_SW_hor - K_B_hor

    K_B_hor[K_B_hor < 0.0] = 0.0

    # Check tau values at night vs. day
    SWin[tau_SW_hor == 0.0] = 0.0
    SWin[SWin < 0.0] = 0.0

    # Elevation angle
    ThetaS = np.pi / 2.0 - zenith_rad  # radians

    # Sunrise/sunset at local time
    sunrise_rad = (sunrise - 12.0) * np.pi / 12.0
    sunset_rad = (sunset - 12.0) * np.pi / 12.0

    ## Disaggregate direct component for Elevation/Topography:
    with np.errstate(divide='ignore'): # suppress divide by zero encountered in log warning
        t_MeanAreal = -np.log(np.nanmean(K_B_hor))  # Optical depth associated with mean areal data
    tElev = t_MeanAreal * (press / press_mean_areal)  # At the pixel elevation optical depth
    KB_elev_Sch = np.exp(-tElev)  # Trasmissivity of the direct component at the pixel elevation
    RsDir_Elev = KB_elev_Sch * RsTOA  # Direct flux at the pixel elevation

    # Check if slope is obstructed from sun and apply thresholds to avoid 
    # numerical issues at times close to sunset/sunrise.
    with np.errstate(divide='ignore', invalid='ignore'): # suppress divide by zero encountered in divide warning
        ConstCheck = (np.tan(slope_rad) / np.tan(ThetaS) * np.cos(azimuth_rad - aspect_rad) + 1.0)
    ind = ((ConstCheck > 0.0) & (cos_sza_hor > 0.001) &
           (hrangle > (sunrise_rad + 0.001)) & (hrangle < (sunset_rad - 0.001)) &
           (ThetaS > (np.pi / 180.0)) & (mask == 1))

    # Slice spatial arrays where applicable
    slope_i = slope_rad[ind] if isinstance(slope_rad, np.ndarray) and slope_rad.shape == mask.shape else slope_rad
    aspect_i = aspect_rad[ind] if isinstance(aspect_rad, np.ndarray) and aspect_rad.shape == mask.shape else aspect_rad
    shade_i = shade[ind] if isinstance(shade, np.ndarray) and shade.shape == mask.shape else shade

    fcor[ind] = shade_i * (1.0 + np.tan(slope_i) / np.tan(ThetaS) * np.cos(azimuth_rad - aspect_i))
    RsDir[ind] = RsDir_Elev[ind] * fcor[ind]

    # Account for remaining pixels in mask
    ind2 = imask & ~ind
    RsDir[ind2] = 0.0

    ## Disaggregate diffuse component for elevation/SVF/reflection term...
    # Define the mean areal data diffuse component. Apply thresholds to avoid 
    # numerical issues at times close to sunset/sunrise.
    ind = ((tau_SW_hor > 0.001) & (hrangle > (sunrise_rad + 0.001)) &
           (hrangle < (sunset_rad - 0.001)) & (cos_sza_hor > 0.001) &
           (K_D_hor > 0.001) & (ThetaS > (np.pi / 180.0)) & (mask == 1))
    E_Sdiff = SWin[ind] * K_D_hor[ind] / tau_SW_hor[ind]

    # Disaggregate diffuse component for elevation
    P0 = 101325.0  # Nominal surface pressure, Pa
    Mz = (1.0 - 0.027 * np.exp(2.0 * press[ind] / P0)) * (1.075 - 0.105 * np.log(1.0 / cos_sza_hor))
    M_MeanAreal = (1.0 - 0.027 * np.exp(2.0 * press_mean_areal / P0)) * (1.075 - 0.105 * np.log(1.0 / cos_sza_hor))
    with np.errstate(over='ignore'): # suppress overflow encountered in exp warning
        RsDif_Elev = E_Sdiff * ((Mz - np.exp(-tElev[ind] / cos_sza_hor)) / (M_MeanAreal - np.exp(-t_MeanAreal / cos_sza_hor)))

    # Reflection term
    SVF_i = SVF[ind] if isinstance(SVF, np.ndarray) and SVF.shape == mask.shape else SVF
    ToTSW_Elev = RsDif_Elev * SVF_i + RsDir[ind]
    RsDif[ind] = RsDif_Elev * SVF_i + (AlbedoSurrounding[ind] * ToTSW_Elev) * (1.0 - SVF_i)

    # Account for remaining pixels in mask
    ind2 = imask & ~ind
    RsDif[ind2] = 0.0

    ## Compute total SW flux
    Rs = RsDif + RsDir  # W/m^2

    return Rs, RsDir, RsDif

def distribute_met_forcing(
    PPT: float,
    SW: float,
    Ta: float,
    qa: float,
    U: float,
    Psfc: float,
    maskNaN: np.ndarray,
    elev: np.ndarray,
    gage_elev: float,
    DOY: int,
    UTC: float,
    time_zone_shift: float,
    lat_mean: float,
    lon_mean: float,
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    SVF: np.ndarray,
    mask: np.ndarray,
    LapseRateTair: float,
    LapseRateTdew: float,
    LapseRatePPT: float,
    albedo: np.ndarray,
    shade_calc_flag: int,
    discrete_azimuth_values: np.ndarray,
    discrete_zenith_values: np.ndarray,
    shade_lookup_table: np.ndarray,
    clear_sky_atmos_emiss_model: str,
    cloudy_sky_atmos_emiss_model: str,
    solar_index: float,
    LW_up: np.ndarray,
    g: float,
    Rd: float,
    T_0: float,
    e_s0: float,
    Lv: float,
    Rv: float,
    epsilon: float,
    S0: float,
    SB_const: float

) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function takes a set of meteorological forcings (from a single gage) and distributes
    it across the domain using topographic data and simple disaggregation functions.

    Inputs:
        PPT: precipitation from gage (mm/h)
        SW: solar radiation from gage (W/m^2)
        Ta: air temperature from gage (K)
        qa: specific humidity from gage (kg/kg)
        U: windspeed from gage (m/s)
        Psfc: surface pressure from gage (Pa)
        maskNaN: mask array for domain (1 or NaN values)
        elev: elevation array for domain
        gage_elev: elevation of gage from which data comes (m)
        DOY: day of year
        UTC: UTC time
        time_zone_shift: time zone shift between UTC and local time
        lat_mean: mean latitude of domain
        lon_mean: mean longitude of domain
        slope_rad: slope array for domain
        aspect_rad: aspect array for domain
        SVF: sky view factor for array for domain
        mask: mask array for watershed
        LapseRateTair: lapse rate in air temperature
        LapseRateTdew: lapse rate in dewpoint temperature
        LapseRatePPT: lapse rate in precipitation
        albedo: Map of albedo over domain
        shade_calc_flag: binary flag indicating whether to perform shade calcs using the shade lookup table (1=calculate; 0=do not calculate)
        discrete_azimuth_values: azimuth angles associated with shade lookup table
        discrete_zenith_values: zenith angles associated with shade lookup table
        shade_lookup_table: Shade lookup table used in interpolation of shade
        clear_sky_atmos_emiss_model: string indicating name of clear-sky longwave model
        cloudy_sky_atmos_emiss_model: string indicating name of cloudy-sky longwave model
        solar index: metric of cloudiness [-]; computed in initialize.m
        LW_up: upwelling longwave flux (from previous time step) used to estimate the contribution of surrounding terrain to incoming longwave
        Constants:
            g: Acceleration of gravity (m/s^2)
            Rd: Ideal gas constant of dry air (J/kg/K)
            T_0: Reference temperature in Clausius-Clapeyron Equatioin (K)
            e_s0: Reference staurated vapor pressure in Clausius-Clapeyron Equatioin (Pa)
            Lv: Latent heat of vaporzation (J/kg)
            Rv: Ideal gas constant of water vapor (J/kg/K)
            epsilon: Rd/Rv (-)
            S0: Solar constant (W/m^2)
            SB_const: Stefan-Boltzman constant (W/m^2/K^4)
    
    Outputs:
        PPT0: distributed precipitation field (actually constant)
        U0: distributed wind field (actually constant)
        Ta0: distributed temperature field (via elevation/lapse rate)
        Psfc0: distributed surface pressure
        qa0: distributed specific humidity
        SW0: distributed solar radiation (as a function of slope/aspect/elev.)
        LWdown0: distributed incoming longwave radiation
    """
    # Pre-allocate gridded arrays from point forcings
    PPT0 = PPT * maskNaN
    SW_hor = SW * maskNaN
    Tair = Ta * maskNaN
    qair = qa * maskNaN
    U0 = U * maskNaN
    press = Psfc * maskNaN

    # Temperature disaggregation
    Ta0, Ta_mean = disaggregate_Tair(Tair, elev, gage_elev, LapseRateTair)

    # Precipitation disaggregation
    PPT0 = disaggregate_PPT(PPT0, elev, gage_elev, LapseRatePPT)

    # Surface pressure disaggregation
    Psfc0 = disaggregate_press(press, Ta_mean, elev, gage_elev, g=g, Rd=Rd)

    # Specific humidity disaggregation
    qa0 = disaggregate_qair(qair, press, Psfc0, elev, gage_elev, LapseRateTdew, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv, epsilon=epsilon)

    # TOA incoming solar radiation and geometry
    RsTOA, zenith_deg, azimuth_deg, sunrise, sunset, _, hour_angle_rad = TOA_incoming_solar(
        DOY, UTC, time_zone_shift, lat_mean, lon_mean, S0=S0
    )

    zenith_rad = np.radians(zenith_deg)
    azimuth_rad = np.radians(azimuth_deg)

    # Topographic shade calculation from 4D table lookup
    if shade_calc_flag:
        # Interpolate across zenith and azimuth axes (dimensions 2 and 3)
        interp = RegularGridInterpolator(
            (discrete_zenith_values, discrete_azimuth_values),
            np.moveaxis(shade_lookup_table, [2, 3], [0, 1]),
            bounds_error=False,
            fill_value=1.0
        )
        shade = interp((zenith_deg, azimuth_deg))
    else:
        shade = np.ones_like(mask)

    # Shortwave radiation disaggregation
    SW0, _, _ = disaggregate_SW(
        SW_hor, Psfc0, slope_rad, aspect_rad, zenith_rad, azimuth_rad,
        hour_angle_rad, shade, SVF, sunrise, sunset, RsTOA, mask, albedo
    )

    # Atmospheric emissivity & downwelling longwave radiation
    ea = specific_humidity_to_vp(qa0, Psfc0, epsilon=epsilon)
    esat = sat_vapor_pressure(Ta0, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
    ea = np.minimum(ea, esat)

    cloud_cover_frac = 1.0 - solar_index
    emiss_a = cloudy_sky_emiss(
        ea / 100.0, Ta0, clear_sky_atmos_emiss_model,
        cloud_cover_frac, solar_index, cloudy_sky_atmos_emiss_model,
        T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv
    )
    emiss_a = np.minimum(emiss_a, 1.0)

    # Downwelling longwave including terrain contribution
    LWdown0 = (
        SVF * emiss_a * SB_const * (Ta0 ** 4) +
        (1.0 - SVF) * LW_up
    )

    return PPT0, U0, Ta0, Psfc0, qa0, SW0, LWdown0

def snow_model(
    P: np.ndarray,
    SW: np.ndarray,
    Psrf: np.ndarray,
    Ta: np.ndarray,
    qa: np.ndarray,
    wind: np.ndarray,
    LWdown: np.ndarray,
    Tsnow0: np.ndarray,
    SWE0: np.ndarray,
    emiss: np.ndarray,
    day_counter: np.ndarray,
    z_snow: float,
    h_snow: float,
    dt: float,
    snow_dens0: np.ndarray,
    h_soil: float | np.ndarray,
    rhow: float,
    ci: float,
    cw: float,
    Lf: float,
    Ls: float,
    SB_const: float,
    T_f: float,
    g: float,
    kappa: float,
    Rd: float,
    Rv: float,
    cp: float,
    Lv: float,
    epsilon: float,
    e_s0: float,
    T_0: float
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Spatially distributed version of a simple one-layer snow mass/energy balance model.

    Inputs:
    P: precipitation map (in mm/hr)
    SW: solar radiation map (in W/m2)
    Psrf: surface pressure map (in Pa)
    Ta: air temperature map (in K)
    qa: specific humidity map (dimensionless)
    wind: wind speed map at reference height 2 m (in m/s)
    LWdown: downwelling longwave radiation map (in W/m2)
    Tsnow0: existing snow temperature map (in K)
    SWE0: existing SWE map (in mm)
    emiss: snow emissivity map
    day_counter: elapsed days since last snowfall (day)
    z_snow: velocity reference height (m)
    h_snow: characteristic snow roughness height (m)
    dt: time step factor (in hours)
    snow_dens0: initial snow density map (kg/m3)
    h_soil: characteristic soil roughness height (m)
    Constants:
        rhow: density of water (kg/m3)
        ci: specific heat of ice (J/kg/K)
        cw: Specific heat capacity of water (J/kg/K)
        Lf: Latent heat of fusion (J/kg)
        Ls: Latent heat of sublimation (J/kg)
        SB_const: Stefan-Boltzman constant (W/m^2/K^4)
        T_f: Water freezing temperature (K)
        g: Acceleration of gravity (m/s^2)
        kappa: Von Karman constant (-)
        Rd: Ideal gas constant of dry air (J/kg/K)
        Rv: Ideal gas constant of water vapor (J/kg/K)
        cp: Specific heat capacity of air (J/kg/K)
        Lv: Latent heat of vaporization (J/kg)
        epsilon: Rd/Rv (-)
        e_s0: Reference saturated vapor pressure in Clausius-Clapeyron equation (Pa)
        T_0: Reference temperature in Clausius-Clapeyron Equation (K)

    Outputs:
    SWE_map: updated SWE map (in mm)
    Tsnow_map: updated snow temperature map (in K)
    melt: resulting melt from change in SWE (in mm/hr)
    LE: latent heat flux map (in W/m2)
    H: sensible heat flux map (in W/m2)
    Rn: net radiation map (in W/m2)
    albedo: snow albedo map (-)
    LWup: upwelling longwave radiation map from snow (in W/m2)
    snow_density: snow density map (kg/m3)
    snow_depth: snow depth map (mm)
    snow_fraction: snow fraction map (-)
    """

    min_density = 100.0  # minimum (new snow) density (kg/m^3)
    
    # Copy array to prevent mutating input arguments in place
    snow_dens0 = np.copy(snow_dens0)
    snow_dens0[snow_dens0 == 0.0] = min_density

    # Conversion factors
    sec2hr = 3600.0  # [s] (3600s = 1hr)
    mm2m = 1000.0    # (1000mm = 1m)

    # Compute vapor pressure
    ea = specific_humidity_to_vp(qa, Psrf, epsilon=epsilon)

    # Determine Air Density Map
    rho = air_density(Ta, ea, Psrf, Rd=Rd, epsilon=epsilon)

    # Assume surface humidity is equal to ICE-sat. vapor pressure
    esat_ice = sat_vapor_pressure_ice(Tsnow0, T_0=T_0, e_s0=e_s0, Ls=Ls, Rv=Rv)
    qsurf = vp_to_specific_humidity(esat_ice, Psrf, epsilon=epsilon)  # kg/kg

    # Aerodynamic resistance (s/m) -- Assumes neutral conditions
    # check for near-zero (less than 0.5 m/s) windspeed and set to low, but
    # positive value
    # wind_calc = np.copy(wind)
    # wind_calc[wind_calc < 0.5] = 0.5  # m/s
    ra = aero_resistance(z_snow, h_snow, wind, kappa=kappa)
    
    # Stability corrections
    RiB = richardson_number(z=z_snow, Tair=Ta, U=wind, Tsurf=Tsnow0, g=g)
    phi_m, phi_h = stab_corr_factors(RiB)
    ra = ra * phi_m * phi_h

    # Evaporation and Sensible heat flux
    E = rho * (qsurf - qa) / ra                             # evaporation rate [kg/m^2/s]
    LE = Ls * E                                             # latent heat of sublimation
    E = E / rhow * (mm2m * sec2hr)                          # evaporation rate [mm/hr]
    
    ET_method_flag = 1
    _, H = mass_transfer(
        Psrf, Tsnow0, Ta, qa, ra, 0.0, 1.0, ET_method_flag,
        cp=cp, Lv=Lv, Rd=Rd, epsilon=epsilon, e_s0=e_s0, Rv=Rv, T_0=T_0
    )  # [W/m^2]

    # Limit evaporation by available moisture (SWE + PPT)
    E_max = (SWE0 + dt * P) / dt                            # maximum possible evap. (mm/hr)
    E = np.minimum(E_max, E)                                # minimum of these two possibilities
    LE = Ls * (E * rhow / (mm2m * sec2hr))                  # latent heat flux (of sublimation)

    # Compute up-welling longwave radiation
    LWup = emiss * SB_const * (Tsnow0 ** 4)                 # [W/m^2]

    # Compute net shortwave radiation
    albedo = albedo_usace(Ta, day_counter, T_f=T_f)
    SW_net = SW * (1.0 - albedo)                            # [W/m^2]

    # Compute net radiation
    Rn = SW_net + LWdown - LWup                             # [W/m^2]

    # Classify precip. occurring at air temp. above freezing as rain
    rain_mask = (Ta > T_f) & (P > 0.0)
    P_melt = np.zeros_like(P)
    advec_energy = np.zeros_like(P)
    latent_energy = np.zeros_like(P)

    if np.any(rain_mask):
        # Advected energy from warm rain (W/m^2)
        advec_energy[rain_mask] = P[rain_mask] * (Ta[rain_mask] - T_f) * cw * rhow / mm2m / sec2hr
        # Energy released via freezing of rain (W/m^2)
        latent_energy[rain_mask] = P[rain_mask] * rhow / mm2m / sec2hr * Lf

    # Set SWE denominator at least equal to 10 mm to avoid numerical instabilities
    dummySWE = np.copy(SWE0)
    dummySWE[dummySWE < 10.0] = 10.0

    # ENERGY BALANCE equation --> Surface temperature update (K)
    Tsnow_map = Tsnow0 + dt * (Rn - LE - H + advec_energy + latent_energy) / (ci * dummySWE * rhow) * (sec2hr * mm2m)

    # # Safety Guard: Snow surface temperature physically cannot exceed freezing (T_f)
    # #               Also, set lower bound relative to air temp to prevent explicit Euler overshoots
    # # This is identical to the lower bound in the surface energy balance (SEB) portion of the simulation model 
    # Tsnow_map = np.clip(Tsnow_map, Ta - 25.0, T_f)

    # Check for phase change
    melt_mask = (Tsnow_map >= T_f) & (SWE0 > 0.0)
    MeltedSWE = np.zeros_like(SWE0)

    if np.any(melt_mask):
        # Energy that would have gone into melting [J/m2/s]
        melt_energy = ((Tsnow_map[melt_mask] - T_f) / dt) * (ci * SWE0[melt_mask]) * rhow / (sec2hr * mm2m)
        MeltedSWE[melt_mask] = melt_energy / (rhow * Lf) * (mm2m * sec2hr)  # [mm/hr]
        Tsnow_map[melt_mask] = T_f                                         # [K]
        
        # Cap melt rate to upper limit of available SWE + incoming precipitation
        PossibleMeltedSWE = (SWE0[melt_mask] + dt * (P[melt_mask] - E[melt_mask])) / dt  # [mm/hr]
        MeltedSWE[melt_mask] = np.minimum(PossibleMeltedSWE, MeltedSWE[melt_mask])

    # MASS BALANCE equation --> SWE update (mm)
    SWE_map = SWE0 + dt * (P - E - MeltedSWE)

    # Set negative SWE values to 0
    SWE_map[SWE_map < 0.0] = 0.0

    # Diagnostic state calculations
    snow_density = diagnostic_snow_density(snow_dens0, SWE_map, dt, P)  # kg/m^3
    snow_depth = rhow / snow_density * SWE_map                           # mm
    snow_fraction = diagnostic_snow_fraction(snow_depth / 1000.0, h_soil)

    # Reset diagnostic values where SWE is zero
    zero_swe = (SWE_map == 0.0)
    snow_density[zero_swe] = 0.0
    snow_depth[zero_swe] = 0.0
    snow_fraction[zero_swe] = 0.0

    # Compute melt from change in SWE
    melt = MeltedSWE + P_melt # [mm/hr]
    melt[melt < 0.0] = 0.0

    return (
        SWE_map, Tsnow_map, melt, LE, H, Rn, albedo, LWup,
        snow_density, snow_depth, snow_fraction
    )

