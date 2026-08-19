import numpy as np
from typing import Callable

def air_density(T: float | np.ndarray, e: float | np.ndarray, P: float | np.ndarray, Rd: float, epsilon: float
                ) -> float | np.ndarray:
    """Computes air density (kg/m^3)."""
    return P / (Rd * virtual_temperature(T, e, P, epsilon=epsilon))

def bisect(f: Callable[[float], float], a0: float, b0: float, ep: float, max_iterate: int) -> float | None:
    """Solve an equation f(x) = 0 using the bisection method.

    Parameters
    ----------
    f : Callable[[float], float]
        User-defined objective function for which the root is sought (f(x) = 0).
    a0 : float
        Lower bound of root range.
    b0 : float
        Upper bound of root range.
    ep : float
        Allowable error tolerance.
    max_iterate : int
        Maximum number of iterations.

    Returns
    -------
    float or None
        Root of function f(x) in bounds a0 and b0, or None if bounds/signs are invalid.

    Notes
    -----
    The function f is to be continuous on the interval [a0, b0], and it is to be of
    opposite signs at a0 and b0. The quantity ep is the error tolerance. The routine
    guarantees this as an error bound provided: (1) the restrictions on the initial
    interval are correct, and (2) ep is not too small when machine epsilon is taken
    into account. Most of these conditions are not checked in the program!

    Examples
    --------
    >>> def target_func(x: float) -> float:
    ...     return x**6 - x - 1.0
    ...
    >>> root = bisect(target_func, 1.0, 1.5, 1.0e-6, 10)
    """
    if a0 >= b0:
        print("a0 < b0 is not true. Stop!")
        return None

    a = float(a0)
    b = float(b0)
    fa = f(a)
    fb = f(b)

    if np.sign(fa) * np.sign(fb) > 0:
        print("f(a0) and f(b0) are of the same sign. Stop!")
        return None

    c = (a + b) / 2.0
    it_count = 0

    while (b - c) > ep and it_count < max_iterate:
        it_count += 1
        fc = f(c)

        # Internal print of bisection method. Press Enter key to continue computation.
        print("    it_count        a            b            c            b-c            f(c)")
        print(f"    {it_count:<8d} {a:12.5e} {b:12.5e} {c:12.5e} {b-c:12.5e} {fc:12.5e}")

        if np.sign(fb) * np.sign(fc) <= 0:
            a = c
            fa = fc
        else:
            b = c
            fb = fc

        c = (a + b) / 2.0
        input()  # Replaces MATLAB pause

    root = c
    return root

def dew_point_temperature_to_vp(T_d: float | np.ndarray,  T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """Determine vapor pressure from dew point temperature.

    Parameters
    ----------
    T_d : float or numpy.ndarray
        Dew point temperature in K.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    e : float or numpy.ndarray
        Vapor pressure in Pa.
    """
    # Equation
    e = sat_vapor_pressure(T_d, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv) 
    return e

def precipitable_water(q: np.ndarray, press: np.ndarray, g: float = 9.81, rhow: float = 1000.0) -> float | np.ndarray:
    """
    Calculate total precipitable water by integrating over an atmospheric column.

    This function takes the atmospheric profile of pressure and specific humidity
    and computes total precipitable water via numerical integration.

    Parameters
    ----------
    q : numpy.ndarray
        Specific humidity profile in kg/kg.
    press : numpy.ndarray
        Atmospheric pressure profile in Pa.
    g : float, default=9.81
        Acceleration of gravity in m/s^2.
    rhow : float, default=1000.0
        Density of water in kg/m^3.

    Returns
    -------
    float or numpy.ndarray
        Total precipitable water in cm.
    """
    q = np.asarray(q)
    press = np.asarray(press)

    # Check that first element of profile data is the surface value
    # based on decreasing pressure with altitude.
    if press[0] < press[-1]:
        press = np.flip(press)
        q = np.flip(q)

    # Precipitable water (kg/m^2)
    # Note: np.trapz takes (y, x) order
    wp = -1.0 / g * np.trapz(q, x=press)

    # Convert to depth units (cm)
    wp = (wp / rhow) * 100.0

    return wp

def RH_to_vp(RH: float | np.ndarray, T: float | np.ndarray, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """
    Compute the vapor pressure from relative humidity.

    Parameters
    ----------
    RH : float or numpy.ndarray
        Relative humidity in percent.
    T : float or numpy.ndarray
        Air temperature in K.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    float or numpy.ndarray
        Vapor pressure in Pa.
    """
    # Determine saturated vapor pressure based on T
    e_s = sat_vapor_pressure(T, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)

    # Determine vapor pressure
    e = (RH * e_s) / 100.0

    return e

def sat_vapor_pressure(T: float | np.ndarray, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """
    Compute the saturated vapor pressure using the Clausius-Clapeyron equation.

    Parameters
    ----------
    T : float or numpy.ndarray
        Air temperature in K.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    float or numpy.ndarray
        Saturated vapor pressure in Pa.
    """
    # Equation
    e_s = e_s0 * np.exp((Lv / Rv) * ((1.0 / T_0) - (1.0 / T)))

    return e_s

def sat_vapor_pressure_ice(T: float | np.ndarray, T_0: float = 273.15, e_s0: float = 611.0, Ls: float = 2.83e6, Rv: float = 461.0,) -> float | np.ndarray:

    """
    Computes ice saturated vapor pressure in Pa using the Clausius-Clapeyron Equation.

    Parameters
    ----------
    T : float or numpy.ndarray
        Air temperature in K.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Ls : float, default=2.83e6
        Latent heat of sublimation (J/kg)
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    float or numpy.ndarray
        Saturated vapor pressure in Pa.
    """
    # Equation
    e_s = e_s0 * np.exp((Ls / Rv) * ((1.0 / T_0) - (1.0 / T)))

    return e_s

def specific_humidity_to_vp(q: float | np.ndarray, P: float | np.ndarray, epsilon: float = 0.622,) -> float | np.ndarray:
    """Convert specific humidity into vapor pressure using e = q * P / epsilon.

    Parameters
    ----------
    q : float or numpy.ndarray
        Specific humidity in kg H2O/kg air.
    P : float or numpy.ndarray
        Atmospheric pressure, must be in same units of pressure as desired for e.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (Rd/Rv) (-).

    Returns
    -------
    e : float or numpy.ndarray
        Vapor pressure in same units as P.
    """
    e = (q * P) / epsilon
    return e

def vapor_pressure_deficit(e: float | np.ndarray, T: float | np.ndarray, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """
    Compute vapor pressure deficit.

    Parameters
    ----------
    e : float or numpy.ndarray
        Vapor pressure in Pa.
    T : float or numpy.ndarray
        Air temperature in K.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    de : float or numpy.ndarray
        Vapor pressure deficit in Pa.
    """
    # Compute saturated vapor pressure using Clausius-Clapeyron equation
    e_s = sat_vapor_pressure(T, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)

    # Equation
    de = e_s - e
    return de

def virtual_temperature(T: float | np.ndarray, e: float | np.ndarray, P: float | np.ndarray, epsilon: float = 0.622,) -> float | np.ndarray:
    """
    Compute virtual temperature.

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature in K.
    e : float or numpy.ndarray
        Vapor pressure, must be in same units of pressure as P.
    P : float or numpy.ndarray
        Total pressure, must be in same units of pressure as e.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (Rd/Rv) (-).

    Returns
    -------
    T_v : float or numpy.ndarray
        Virtual temperature in K.

    Notes
    -----
    The units of e and P may be in any unit of pressure as long as they
    are both input in the same units (e.g., e in Pa and P in Pa or e in psi
    and P in psi).
    """
    # Equation
    T_v = T / (1.0 - (1.0 - epsilon) * (e / P))
    return T_v

def vp_to_dew_point_temperature(e: float | np.ndarray, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """
    Compute dew point temperature using the Clausius-Clapeyron equation.

    Parameters
    ----------
    e : float or numpy.ndarray
        Vapor pressure in Pa.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    T_d : float or numpy.ndarray
        Dew point temperature in K.
    """
    # Equation
    T_d = 1.0 / ((1.0 / T_0) - (Rv / Lv) * np.log(e / e_s0))
    return T_d

def vp_to_RH(e: float | np.ndarray, T: float | np.ndarray, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """
    Compute relative humidity from vapor pressure.

    Parameters
    ----------
    e : float or numpy.ndarray
        Vapor pressure in Pa.
    T : float or numpy.ndarray
        Air temperature in K.
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    RH : float or numpy.ndarray
        Relative humidity in percent.
    """
    # Determine saturated vapor pressure based on T
    e_s = sat_vapor_pressure(T, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)

    # Compute relative humidity
    RH = 100.0 * (e / e_s)
    return RH

def vp_to_specific_humidity(e: float | np.ndarray, P: float | np.ndarray, epsilon: float = 0.622,) -> float | np.ndarray:
    """
    Convert vapor pressure into specific humidity using q = epsilon * e / P.

    Parameters
    ----------
    e : float or numpy.ndarray
        Vapor pressure, must be in same units of pressure as P.
    P : float or numpy.ndarray
        Atmospheric pressure, must be in same units of pressure as e.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (Rd/Rv) (-).

    Returns
    -------
    q : float or numpy.ndarray
        Specific humidity in units of kg H2O/kg air.

    Notes
    -----
    The units of e and P may be in any unit of pressure as long as they
    are both input using the same units (e.g., e in Pa and P in Pa or e
    in mb and P in mb).
    """
    # Equation
    q = epsilon * e / P
    return q

def vp_to_wet_bulb_temperature(e: float, T: float, P: float, 
                               cp: float = 1004.0, epsilon: float = 0.622, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | None:
    """
    Determine wet bulb temperature from vapor pressure via bisection method.

    This function determines the wet bulb temperature from vapor pressure by
    implicitly solving the wet bulb temperature equation through the use of
    the bisection root finding method. This solution has an error of 1e-4 K.

    Parameters
    ----------
    e : float
        Vapor pressure in Pa.
    T : float
        Air temperature in K.
    P : float
        Air pressure in Pa.
    cp : float, default=1004.0
        Specific heat capacity of air in J/kg/K.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (Rd/Rv) (-).
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    T_w : float or None
        Wet bulb temperature in K, or None if bounds fail.
    """
    ep = 0.0001
    max_iterate = 1000

    # Compute dew point temperature to be used as lower bound
    T_d = vp_to_dew_point_temperature(e, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)

    # Define root upper and lower bounds
    T_up = T
    # Using slightly larger value than actual dew point to avoid singularity
    T_low = T_d * 1.001

    # Start bisect method
    a = T_low
    b = T_up
    fa = wet_bulb(a, e, T, P, cp=cp, epsilon=epsilon, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
    fb = wet_bulb(b, e, T, P, cp=cp, epsilon=epsilon, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)

    if np.sign(fa) * np.sign(fb) > 0:
        print(
            "The function bounds are of the same sign. This means the root finder will fail. Stopping code ..."
        )
        return None

    c = (a + b) / 2.0
    it_count = 0

    while (b - c) > ep and it_count < max_iterate:
        it_count += 1
        fc = wet_bulb(c, e, T, P, cp=cp, epsilon=epsilon, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)

        if np.sign(fb) * np.sign(fc) <= 0:
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        c = (a + b) / 2.0

    T_w = c
    return T_w

def wet_bulb(Tw: float | np.ndarray, e: float | np.ndarray, T: float | np.ndarray, P: float | np.ndarray, 
             cp: float = 1004.0, epsilon: float = 0.622, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """
    Objective function used in root finding to determine wet bulb temperature.

    Parameters
    ----------
    Tw : float or numpy.ndarray
        Estimated wet bulb temperature in K.
    e : float or numpy.ndarray
        Vapor pressure in Pa.
    T : float or numpy.ndarray
        Air temperature in K.
    P : float or numpy.ndarray
        Air pressure in Pa.
    cp : float, default=1004.0
        Specific heat capacity of air in J/kg/K.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (Rd/Rv) (-).
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    val : float or numpy.ndarray
        Function value equal to zero when Tw is correct.
    """
    q = vp_to_specific_humidity(e, P, epsilon=epsilon)
    q_s = vp_to_specific_humidity(
        sat_vapor_pressure(Tw, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv), P, epsilon=epsilon
    )

    val = (Lv / cp) - (T - Tw) / (q_s - q)
    return val

def wet_bulb_temperature_to_vp(T_w: float | np.ndarray, T: float | np.ndarray, P: float | np.ndarray, 
                               cp: float = 1004.0, epsilon: float = 0.622, T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """Determine vapor pressure from wet bulb temperature.

    Parameters
    ----------
    T_w : float or numpy.ndarray
        Wet bulb temperature in K.
    T : float or numpy.ndarray
        Air temperature in K.
    P : float or numpy.ndarray
        Air pressure in Pa.
    cp : float, default=1004.0
        Specific heat capacity of air in J/kg/K.
    epsilon : float, default=0.622
        Ratio of dry air to water vapor gas constants (Rd/Rv) (-).
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    e : float or numpy.ndarray
        Vapor pressure in Pa.
    """
    # Compute saturated specific humidity
    q_s = vp_to_specific_humidity(
        sat_vapor_pressure(T_w, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv),
        P,
        epsilon=epsilon,
    )

    # Determine specific humidity from wet bulb equation
    q = q_s - (T - T_w) * (cp / Lv)

    # Convert specific humidity to vapor pressure
    e = specific_humidity_to_vp(q, P, epsilon=epsilon)

    return e

def Wp_from_near_surface_met_data(ea: float | np.ndarray, Ta: float | np.ndarray, model_name: str,
                                  T_0: float = 273.15, e_s0: float = 611.0, Lv: float = 2.5e6, Rv: float = 461.0,) -> float | np.ndarray:
    """
    Calculate total precipitable water using an empirical model.

    Script calculates the total precipitable water using an empirical model
    as specified by the model_name string.

    Parameters
    ----------
    ea : float or numpy.ndarray
        Reference-level vapor pressure in Pa.
    Ta : float or numpy.ndarray
        Reference-level temperature in K.
    model_name : str
        Descriptor of which model to use:
        'dingman' : use model from Dingman textbook (from Bolsenga, 1964)
        'prata'   : use Prata (1996) model
    T_0 : float, default=273.15
        Reference temperature in Clausius-Clapeyron equation in K.
    e_s0 : float, default=611.0
        Reference saturated vapor pressure in Clausius-Clapeyron equation in Pa.
    Lv : float, default=2.5e6
        Latent heat of vaporization in J/kg.
    Rv : float, default=461.0
        Ideal gas constant of water vapor in J/kg/K.

    Returns
    -------
    Wp : float or numpy.ndarray
        Total precipitable water in cm.

    Notes
    -----
    If using the Dingman model, air temperature is not a necessary input
    and can take any numeric placeholder.
    """
    model = model_name.lower()

    if model == "dingman":
        # Compute dewpoint temperature
        Td = vp_to_dew_point_temperature(ea, T_0=T_0, e_s0=e_s0, Lv=Lv, Rv=Rv)
        # Convert to deg. Celsius
        Td = Td - 273.15
        # Compute precipitable water (centimeters)
        Wp = 1.12 * np.exp(0.0614 * Td)

    elif model == "prata":
        # Convert vapor pressure to mb
        ea_mb = ea / 100.0
        # Compute precipitable water (centimeters)
        Wp = 46.5 * ea_mb / Ta

    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Expected 'dingman' or 'prata'.")

    return Wp

def Wp_from_Td(Td: float | np.ndarray,) -> float | np.ndarray:
    """Calculate total precipitable water using an empirical surface dew point equation.

    Calculates the total precipitable water using an empirical (Dingman 2008)
    equation when the surface dew point temperature is specified.

    Parameters
    ----------
    Td : float or numpy.ndarray
        Dew point temperature in degrees Celsius.

    Returns
    -------
    Wp : float or numpy.ndarray
        Total precipitable water in cm.
    """
    Wp = 1.12 * np.exp(0.0614 * Td)
    return Wp

