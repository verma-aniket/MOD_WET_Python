import numpy as np

from src.chapter8 import soil_SEB_solver_prognostic

def step_bare_soil_seb(model, maskSEB: np.ndarray, SW0: np.ndarray, Ta0: np.ndarray, qa0: np.ndarray, 
                       U0: np.ndarray, Psfc0: np.ndarray, LWdown0: np.ndarray, step_idx: int, ts_idx: int
                       ) -> None:
    """
    Executes the bare soil surface energy balance solver for non-snow pixels,
    updates step_vars in place, records time-series outputs, and updates accumulators.
    """
    # Call Surface Energy Balance (SEB) model
    # Define rootzone soil moisture for this time step (set in static maps
    # even though it is evolving because it is not saved later)
    theta_rz = model.state.Srz / model.params.d_rz  # [-]

    if np.any(maskSEB):
        # surface energy balance function (for non-snowy pixels)
        (
            Tsurf_SEB,
            LE_SEB,
            H_SEB,
            G_SEB,
            Rn_SEB,
            Td_SEB,
            Rlup_SEB,
        ) = soil_SEB_solver_prognostic(
            SW=SW0[maskSEB],
            Ta=Ta0[maskSEB],
            qa=qa0[maskSEB],
            U=U0[maskSEB],
            Psfc=Psfc0[maskSEB],
            LWdown=LWdown0[maskSEB],
            theta_rz=theta_rz[maskSEB],
            Ts0=model.state.Tsurf[maskSEB],
            theta_wp=model.spatial.THETApwp[maskSEB],
            theta_fc=model.spatial.THETAfc[maskSEB],
            emiss=model.spatial.emiss[maskSEB],
            albedo=model.spatial.albedo[maskSEB],
            Td0=model.state.Td[maskSEB],
            z_m=model.params.z_m,
            h_rough=model.params.h_rough,
            Csoil=model.params.Csoil,
            dg=model.params.dg,
            dt=model.control.dt,
            SB_const=model.constants.SB_const,
            kappa=model.constants.kappa,
            g=model.constants.g,
            cp=model.constants.cp,
            Lv=model.constants.Lv,
            Rd=model.constants.Rd,
            epsilon=model.constants.epsilon,
            e_s0=model.constants.e_s0,
            Rv=model.constants.Rv,
            T_0=model.constants.T_0
        )

        # Store bare pixel states/fluxes in whole-domain array
        # states
        model.step_vars.Tsurf_new[maskSEB] = Tsurf_SEB  # (K)
        model.step_vars.Td_new[maskSEB] = Td_SEB  # (K)
        model.step_vars.SWE_new[maskSEB] = 0.0  # (mm)
        model.step_vars.snowdens_new[maskSEB] = 0.0  # (kg/m^3)
        model.step_vars.snowdepth_new[maskSEB] = 0.0  # (mm)
        model.step_vars.snowfrac_new[maskSEB] = 0.0  # (-)
        # fluxes
        model.step_vars.LE_out[maskSEB] = LE_SEB  # (W/m^2)
        model.step_vars.H_out[maskSEB] = H_SEB  # (W/m^2)
        model.step_vars.G_out[maskSEB] = G_SEB  # (W/m^2)
        model.step_vars.Rn_out[maskSEB] = Rn_SEB  # (W/m^2)
        model.step_vars.Rlup_out[maskSEB] = Rlup_SEB  # (W/m^2)
        # albedo
        model.step_vars.albedo_out[maskSEB] = model.spatial.albedo[maskSEB]  # (-)

    # Store state/flux variables for this time step
    if (step_idx + 1) % model.control.timeseries_frq2store == 0:
        model.time_series.Tsurf[ts_idx + 1] = float(np.nanmean(model.step_vars.Tsurf_new))  # (K)
        model.time_series.Td[ts_idx + 1] = float(np.nanmean(model.step_vars.Td_new))  # (K)
        model.time_series.SWE[ts_idx + 1] = float(np.nanmean(model.step_vars.SWE_new) / 1000.0)  # (m)
        model.time_series.snowdepth[ts_idx + 1] = float(np.nanmean(model.step_vars.snowdepth_new) / 1000.0)  # (m)
        model.time_series.snowdens[ts_idx + 1] = float(np.nanmean(model.step_vars.snowdens_new))  # (kg/m^3)
        model.time_series.snowfrac[ts_idx + 1] = float(np.nanmean(model.step_vars.snowfrac_new))  # (-)
        
        model.time_series.Rn[ts_idx] = float(np.nanmean(model.step_vars.Rn_out))  # (W/m^2)
        model.time_series.LE[ts_idx] = float(np.nanmean(model.step_vars.LE_out))  # (W/m^2)
        model.time_series.H[ts_idx] = float(np.nanmean(model.step_vars.H_out))  # (W/m^2)
        model.time_series.Rlup[ts_idx] = float(np.nanmean(model.step_vars.Rlup_out))  # (W/m^2)
        model.time_series.albedo[ts_idx] = float(np.nanmean(model.step_vars.albedo_out))  # (-)

        # Special pixels "if model.params.special_pixels" checks if special pixels is not None and not empty
        if model.params.special_pixels and len(model.params.special_pixels[0]) > 0:
            sp_rows, sp_cols = model.params.special_pixels
            model.time_series.pixel_Tsurf[ts_idx + 1][sp_rows, sp_cols] = model.step_vars.Tsurf_new[sp_rows, sp_cols]  # (K)
            model.time_series.pixel_Td[ts_idx + 1][sp_rows, sp_cols] = model.step_vars.Td_new[sp_rows, sp_cols]  # (K)
            model.time_series.pixel_SWE[ts_idx + 1][sp_rows, sp_cols] = model.step_vars.SWE_new[sp_rows, sp_cols] / 1000.0  # (m)
            model.time_series.pixel_Rn[ts_idx][sp_rows, sp_cols] = model.step_vars.Rn_out[sp_rows, sp_cols]  # (W/m^2)
            model.time_series.pixel_LE[ts_idx][sp_rows, sp_cols] = model.step_vars.LE_out[sp_rows, sp_cols]  # (W/m^2)
            model.time_series.pixel_H[ts_idx][sp_rows, sp_cols] = model.step_vars.H_out[sp_rows, sp_cols]  # (W/m^2)
            model.time_series.pixel_Rlup[ts_idx][sp_rows, sp_cols] = model.step_vars.Rlup_out[sp_rows, sp_cols]  # (W/m^2)
            model.time_series.pixel_snowdepth[ts_idx + 1][sp_rows, sp_cols] = model.step_vars.snowdepth_new[sp_rows, sp_cols] / 1000.0  # (m)
            model.time_series.pixel_snowdens[ts_idx + 1][sp_rows, sp_cols] = model.step_vars.snowdens_new[sp_rows, sp_cols]  # (kg/m^3)
            model.time_series.pixel_snowfrac[ts_idx + 1][sp_rows, sp_cols] = model.step_vars.snowfrac_new[sp_rows, sp_cols]  # (-)

    # accumulate states/fluxes for time averaging.
    model.accumulators.cumulTsurf += model.step_vars.Tsurf_new  # (K)
    model.accumulators.cumulTd += model.step_vars.Td_new  # (K)
    model.accumulators.cumulSWE += model.step_vars.SWE_new / 1000.0  # (m)
    #
    model.accumulators.cumulsnowdepth += model.step_vars.snowdepth_new / 1000.0  # (m)
    model.accumulators.cumulsnowdens += model.step_vars.snowdens_new  # (kg/m^3)
    model.accumulators.cumulsnowfrac += model.step_vars.snowfrac_new  # (-)
    #
    model.accumulators.cumulalbedo += model.step_vars.albedo_out  # (-)
    model.accumulators.cumulRn += model.step_vars.Rn_out  # (W/m^2)
    model.accumulators.cumulLE += model.step_vars.LE_out  # (W/m^2)
    model.accumulators.cumulH += model.step_vars.H_out  # (W/m^2)
    model.accumulators.cumulRlup += model.step_vars.Rlup_out  # (W/m^2)

