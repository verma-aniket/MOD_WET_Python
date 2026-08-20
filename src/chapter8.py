import numpy as np

from src.chapter2 import air_density, sat_vapor_pressure, specific_humidity_to_vp, vp_to_specific_humidity

def aero_resistance(z: float | np.ndarray, h: float | np.ndarray, v: np.ndarray, kappa: float = 0.4
                    ) -> np.ndarray:
    """
    Calculate aerodynamic resistance for evapotranspiration models.

    Parameters
    ----------
    z : float or numpy.ndarray
        Velocity reference height in meters.
    h : float or numpy.ndarray
        Characteristic roughness height in meters.
    v : numpy.ndarray
        Horizontal wind velocity at reference height (z) in m/s.
    kappa : float, default=0.4
        Von Karman constant (-).

    Returns
    -------
    numpy.ndarray
        Aerodynamic resistance in s/m.
    """
    d = 0.7 * h     # Zero-plane displacement height approximation
    z_0 = 0.1 * h   # Momentum roughness height approximation

    # Equation
    r_a = np.log((z - d) / z_0) ** 2 / ((kappa ** 2) * v)

    return r_a

def mass_transfer(P: np.ndarray, T_surf: np.ndarray, T_air: np.ndarray, q_air: np.ndarray, r_a: np.ndarray, r_c: float | np.ndarray,  beta: float | np.ndarray, method: int,
                  cp: float, Lv: float, Rd: float, epsilon: float, e_s0: float, Rv: float, T_0: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Computes latent and sensible heat fluxes using Mass Transfer / Diffusion Analogy."""
    e_air = specific_humidity_to_vp(q_air, P, epsilon=epsilon)
    density = air_density(T_air, e_air, P, Rd=Rd, epsilon=epsilon)
    
    esat_surf = sat_vapor_pressure(T_surf, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
    q_sat = vp_to_specific_humidity(esat_surf, P, epsilon=epsilon)

    if method == 0:
        latent_heat_flux = density * Lv * (beta * q_sat - q_air) / (r_a + r_c)
    elif method == 1:
        latent_heat_flux = density * Lv * beta * (q_sat - q_air) / (r_a + r_c)
    else:
        latent_heat_flux = np.zeros_like(P)

    sensible_heat_flux = density * cp * (T_surf - T_air) / r_a
    return latent_heat_flux, sensible_heat_flux

def richardson_number(z: float | np.ndarray, Tair: np.ndarray, U: np.ndarray, Tsurf: np.ndarray, g: float
                      ) -> np.ndarray:
    """Determines the bulk Richardson number (dimensionless)."""
    return g * z * (Tair - Tsurf) / (Tsurf * (U ** 2))

def stab_corr_factors(RiB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Determines atmospheric stability correction factors from the bulk Richardson number."""
    # Takes care of RiB values above the allowable limit
    # RiB_capped = np.minimum(RiB, 0.19) # CAUSES massive thresholding swings when wind speed is close to 0.19, DO NOT USE
    RiB_capped = np.where(RiB >= 0.2, 0.19, RiB) # replicates original MATLAB behavior, more stable

    phi_m = np.full_like(RiB_capped, np.nan, dtype=np.float64)
    phi_h = np.full_like(RiB_capped, np.nan, dtype=np.float64)

    # Unstable case (RiB <= 0)
    unstable = RiB_capped <= 0.0
    phi_h[unstable] = (1.0 - 15.0 * RiB_capped[unstable]) ** -0.5
    phi_m[unstable] = phi_h[unstable] ** 0.5

    # Stable case (0 < RiB < 0.2)
    stable = (RiB_capped > 0.0) & (RiB_capped < 0.2)
    phi_h[stable] = (1.0 - 5.0 * RiB_capped[stable]) ** -1.0
    phi_m[stable] = phi_h[stable]

    return phi_m, phi_h

def soil_SEB_solver_prognostic(
    SW: np.ndarray,
    Ta: np.ndarray,
    qa: np.ndarray,
    U: np.ndarray,
    Psfc: np.ndarray,
    LWdown: np.ndarray,
    theta_rz: np.ndarray,
    Ts0: np.ndarray,
    theta_wp: np.ndarray,
    theta_fc: np.ndarray,
    emiss: np.ndarray,
    albedo: np.ndarray,
    Td0: np.ndarray,
    z_m: float,
    h_rough: float,
    Csoil: float,
    dg: float,
    dt: float,
    SB_const: float,
    kappa: float,
    g: float,
    cp: float,
    Lv: float,
    Rd: float,
    epsilon: float,
    e_s0: float,
    Rv: float,
    T_0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    This function solves the surface energy balance prognostically
    as a function of surface temperature.

    Inputs:
    SW: Incoming shortwave radiation (W/m^2)
    Ta: Reference-level air temperature (K)
    qa: Reference-level specific humidity (kg/kg)
    U: Reference-level wind speed (m/s)
    Psfc: Reference-level air pressure (Pa)
    LWdown: Downwelling longwave radiation (W/m^2)
    theta_rz: rootzone soil moisture (-)
    Ts0: initial guess for surface temperature (from previous time step) (K)
    theta_wp: wilting point (-)
    theta_fc: field capacity (-)
    emiss: Surface emissivity (-)
    albedo: Surface albedo (-)
    Td0: initial guess for deep soil temperature (from previous time step) (K)
    z_m: Meteorological reference measurement height (m)
    h_rough: Characteristic soil roughness height (m)
    Csoil: Soil heat capacity (J/m^3/K)
    dg: Surface soil depth (m)
    dt: Timestep (hours)
    Constants:
        SB_const: Stefan-Boltzmann constant (W/m^2/K^4)
        kappa: Von Karman constant (-)
        g: Gravitational acceleration (m/s^2)
        cp: Specific heat capacity of air (J/kg/K)
        Lv: Latent heat of vaporization (J/kg)
        Rd: Gas constant for dry air (J/kg/K)
        epsilon: Ratio of molecular weights (H2O/Dry Air)
        e_s0: Saturation vapor pressure at reference temperature (Pa)
        Rv: Gas constant for water vapor (J/kg/K)
        T_0: Reference temperature (K)

    Outputs:
    Tsurf: Surface temperature (K)
    LE: Latent heat flux (W/m^2)
    H: Sensible heat flux (W/m^2)
    G: Ground heat flux (W/m^2)
    Rn: Net radiation (W/m^2)
    Td: Deep soil temperature [K]
    LWup: Upwelling longwave radiation (W/m^2)
    """
    omega = 1.0 / 86400.0  # diurnal frequency (1/s)
    
    # Compute net shortwave radiation (W/m^2)
    SW_net = SW * (1.0 - albedo)

    # Compute net radiation
    LWup = emiss * SB_const * (Ts0 ** 4)
    Rn = SW_net + LWdown - LWup

    # Compute actual evaporation
    beta = np.full_like(theta_rz, np.nan)
    beta[theta_rz >= theta_fc] = 1.0
    beta[theta_rz <= theta_wp] = 0.0
    ind = np.isnan(beta)
    beta[ind] = (theta_rz[ind] - theta_wp[ind]) / (theta_fc[ind] - theta_wp[ind])

    # Compute aerodynamic resistance
    # check for near-zero (less than 0.5 m/s) windspeed and set to low, but
    # positive value
    U_calc = np.copy(U)
    U_calc[U_calc < 0.5] = 0.5  # m/s
    r_a = aero_resistance(z_m, h_rough, U_calc, kappa=kappa)

    # use stability corrections
    RiB = richardson_number(z=z_m, Tair=Ta, U=U_calc, Tsurf=Ts0, g=g)
    phi_m, phi_h = stab_corr_factors(RiB)
    r_a = r_a * phi_m * phi_h

    ET_method_flag = 1
    LE, H = mass_transfer(Psfc, Ts0, Ta, qa, r_a, 0, beta, ET_method_flag, cp, Lv, Rd, epsilon, e_s0, Rv, T_0)
    CT = 1.0 / (Csoil * dg)
    G = Rn - LE - H

    # Mimics the force restore equation (with a constant soil heat capacity)
    Tsurf = Ts0 + dt * (CT * G - 2.0 * np.pi * omega * (Ts0 - Td0)) * 3600.0
    Td = Td0 + dt * (omega * (Ts0 - Td0)) * 3600.0

    # # Safety Guard: Prevent explicit Euler overshoots relative to air temp
    # Tsurf = np.clip(Tsurf, Ta - 25.0, Ta + 25.0)

    return Tsurf, LE, H, G, Rn, Td, LWup

