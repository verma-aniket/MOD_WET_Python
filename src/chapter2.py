import numpy as np

# Functions to add:
#   bisect
#   precipitable_water
#   RH_to_vp
#   vapor_pressure_deficit
#   vp_to_RH
#   vp_to_wet_bulb_temperature
#   wet_bulb_temperature_to_vp
#   Wp_from_Td


def air_density(T: float | np.ndarray, e: float | np.ndarray, P: float | np.ndarray, Rd: float, epsilon: float
                ) -> float | np.ndarray:
    """Computes air density (kg/m^3)."""
    return P / (Rd * virtual_temperature(T, e, P, epsilon=epsilon))

def dew_point_temperature_to_vp(T_d: float | np.ndarray, T_0: float, e_s0: float, Lv: float, Rv: float
                                ) -> float | np.ndarray:
    """Computes vapor pressure (Pa) from dew point temperature (K)."""
    return sat_vapor_pressure(T_d, T_0, e_s0, Lv, Rv)

def sat_vapor_pressure(T: float | np.ndarray, T_0: float, e_s0: float, Lv: float, Rv: float
                       ) -> float | np.ndarray:
    """Computes saturation vapor pressure (Pa) via Clausius-Clapeyron equation."""
    return e_s0 * np.exp((Lv / Rv) * (1.0 / T_0 - 1.0 / T))

def sat_vapor_pressure_ice(T: float | np.ndarray, T_0: float, e_s0: float, Ls: float, Rv: float,
                           ) -> float | np.ndarray:
    """Computes ice saturated vapor pressure in Pa using the Clausius-Clapeyron Equation."""
    return e_s0 * np.exp((Ls / Rv) * (1.0 / T_0 - 1.0 / T))

def specific_humidity_to_vp(q: float | np.ndarray, P: float | np.ndarray, epsilon: float
                            ) -> float | np.ndarray:
    """Converts specific humidity (kg/kg) into vapor pressure (same units as P)."""
    return q * P / epsilon

def virtual_temperature(T: float | np.ndarray, e: float | np.ndarray,  P: float | np.ndarray,  epsilon: float
                        ) -> float | np.ndarray:
    """Computes virtual temperature (K)."""
    return T / (1.0 - (1.0 - epsilon) * (e / P))

def vp_to_dew_point_temperature(e: float | np.ndarray, T_0: float, e_s0: float, Lv: float, Rv: float
                                ) -> float | np.ndarray:
    """Computes dew point temperature (K) from vapor pressure (Pa) using Clausius-Clapeyron."""
    return 1.0 / ( (1.0 / T_0) - (Rv / Lv) * np.log(e / e_s0) )

def vp_to_specific_humidity(e: float | np.ndarray, P: float | np.ndarray, epsilon: float) -> float | np.ndarray:
    """Converts vapor pressure to specific humidity (kg/kg)."""
    return epsilon * e / P

def Wp_from_near_surface_met_data(ea: float | np.ndarray, Ta: float | np.ndarray, model_name: str, T_0: float, e_s0: float, Lv: float, Rv: float
                                  ) -> float | np.ndarray:
    """Calculates total precipitable water (Wp in cm) using an empirical model."""
    model = model_name.lower()

    if model == "dingman":
        # Pass physical constants explicitly to dewpoint calculation
        Td = vp_to_dew_point_temperature(ea, T_0, e_s0, Lv, Rv)
        Td_c = Td - 273.15
        return 1.12 * np.exp(0.0614 * Td_c)

    elif model == "prata":
        ea_mb = ea / 100.0
        return 46.5 * ea_mb / Ta

    else:
        raise ValueError(
            f"Unsupported precipitable water model: '{model_name}'. Expected 'dingman' or 'prata'."
        )

