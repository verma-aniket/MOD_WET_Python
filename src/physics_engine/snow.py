import numpy as np

import numpy as np
from src.chapter6 import snow_model


def step_snow_model(model, PPT0: np.ndarray, U0: np.ndarray, Ta0: np.ndarray, Psfc0: np.ndarray, qa0: np.ndarray, 
                    SW0: np.ndarray, LWdown0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes the snow module step using distributed meteorological forcings and 
    dynamic model states, updating model.step_vars in place.

    Parameters
    ----------
    model : MODWETModel
        Model container instance holding constants, parameters, spatial maps, forcing data, and model states.
    PPT0, U0, Ta0, Psfc0, qa0, SW0, LWdown0 : np.ndarray
        2D distributed meteorological inputs returned by step_met_forcing_distribution().
            PPT0: distributed precipitation field (actually constant)
            U0: distributed wind field (actually constant)
            Ta0: distributed temperature field (via elevation/lapse rate)
            Psfc0: distributed surface pressure
            qa0: distributed specific humidity
            SW0: distributed solar radiation (as a function of slope/aspect/elev.)
            LWdown0: distributed incoming longwave radiation
    
    Returns
    -------
    masksnow : np.ndarray
        2D boolean mask of active snow-covered or snowing grid cells.
    maskSEB : np.ndarray
        2D boolean mask of active bare soil / snow-free grid cells within the watershed.
    """

    # 1. Identify active snow-covered or active snowfall cells
    # State names match ModelState dataclass: SWE, Tsurf, snowdens, NDayLastSnow
    masksnow = (model.state.SWE > 0.0) | ((Ta0 < model.constants.T_f) & (PPT0 > 0.0))

    # 2. Identify bare soil cells inside active watershed mask
    maskSEB = (~masksnow) & (model.spatial.mask == 1)

    # 3. Pre-allocate / initialize step_vars defaults for current timestep
    model.step_vars.Rn_out = np.full((model.control.nx, model.control.ny), np.nan, dtype=np.float64)
    model.step_vars.LE_out = np.full((model.control.nx, model.control.ny), np.nan, dtype=np.float64)
    model.step_vars.H_out = np.full((model.control.nx, model.control.ny), np.nan, dtype=np.float64)
    model.step_vars.G_out = np.full((model.control.nx, model.control.ny), np.nan, dtype=np.float64)
    model.step_vars.Rlup_out = np.full((model.control.nx, model.control.ny), np.nan, dtype=np.float64)
    model.step_vars.melt_out = np.zeros((model.control.nx, model.control.ny))

    # 4. Execute snow physics if snow is present or falling
    if np.any(masksnow):
        # Reset albedo day counter on significant snowfall event
        big_snow = (PPT0 >= model.params.RestoreAlbedo) & masksnow
        model.state.NDayLastSnow[big_snow] = 0.0

        # Handle spatial or scalar snow emissivity
        snow_emiss = (
            model.spatial.snow_emiss[masksnow]
            if model.spatial.snow_emiss is not None
            else model.params.snow_emiss
        )

        (
            SWE_map,
            Tsnow_map,
            melt_snow,
            LE_snow,
            H_snow,
            Rn_snow,
            albedo_snow,
            Rlup_snow,
            snow_density_map,
            snow_depth_map,
            snow_fraction_map,
        ) = snow_model(
            P=PPT0[masksnow],
            SW=SW0[masksnow],
            Psrf=Psfc0[masksnow],
            Ta=Ta0[masksnow],
            qa=qa0[masksnow],
            wind=U0[masksnow],
            LWdown=LWdown0[masksnow],
            Tsnow0=model.state.Tsurf[masksnow],
            SWE0=model.state.SWE[masksnow],
            emiss=snow_emiss,
            day_counter=model.state.NDayLastSnow[masksnow],
            z_snow=model.params.z_snow,
            h_snow=model.params.h_snow,
            dt=model.control.dt,
            snow_dens0=model.state.snowdens[masksnow],
            h_soil=model.params.h_rough,
            rhow=model.constants.rhow,
            ci=model.constants.ci,
            cw=model.constants.cw,
            Lf=model.constants.Lf,
            Ls=model.constants.Ls,
            SB_const=model.constants.SB_const,
            T_f=model.constants.T_f,
            g=model.constants.g,
            kappa=model.constants.kappa,
            Rd=model.constants.Rd,
            Rv=model.constants.Rv,
            cp=model.constants.cp,
            Lv=model.constants.Lv,
            epsilon=model.constants.epsilon,
            e_s0=model.constants.e_s0,
            T_0=model.constants.T_0
        )

        # Map 1D physics outputs directly into model.step_vars
        model.step_vars.Tsurf_new[masksnow] = Tsnow_map
        model.step_vars.SWE_new[masksnow] = SWE_map
        model.step_vars.Td_new[masksnow] = model.constants.T_f
        model.step_vars.snowdens_new[masksnow] = snow_density_map
        model.step_vars.snowdepth_new[masksnow] = snow_depth_map
        model.step_vars.snowfrac_new[masksnow] = snow_fraction_map

        model.step_vars.Rn_out[masksnow] = Rn_snow
        model.step_vars.LE_out[masksnow] = LE_snow
        model.step_vars.H_out[masksnow] = H_snow
        model.step_vars.G_out[masksnow] = Rn_snow - LE_snow - H_snow
        model.step_vars.Rlup_out[masksnow] = Rlup_snow
        model.step_vars.albedo_out[masksnow] = albedo_snow
        model.step_vars.melt_out[masksnow] = melt_snow / 1000.0 # # Convert mm/hr to m/hr

        # # clear memory
        # del SWE_map, Tsnow_map, melt_snow, LE_snow, H_snow, Rn_snow, albedo_snow, Rlup_snow, snow_density_map, snow_depth_map, snow_fraction_map

    return masksnow, maskSEB
