import numpy as np

def mass_balance_check(PPT: np.ndarray, ET: np.ndarray, qse: np.ndarray, qb: np.ndarray, qie: np.ndarray, dx: float, dy: float, 
                       dSD: np.ndarray, dSuz: np.ndarray, dSrz: np.ndarray, dSWE: np.ndarray, dt: float, rhow: float = 1000.0
                       ) -> float:
    """
    Description: This function does a basin-wide mass balance check at the
    end of each time step.

    Inputs:
        PPT: precipitation field (m/h)
        ET: evaporation field (m/h)
        qse: saturation excess field (m)
        qb: baseflow field (m)
        qie: infiltration excess field (m)
        dx: spatial resolution (x)
        dy: spatial resolution (y)
        dSD: change in saturation deficit over time step (m)
        dSuz: change in unsat. zone moisture over time step (m)
        dSrz: change in rootzone moisture over time step (m)
        dSWE: change in SWE over time step (mm)
        dt: time step (h)
        rhow: density of water (kg/m^3)

    Outputs:
        mass_resid: Residual of mass balance
    """
    # Convert fluxes to vol. rates (m^3/hr)
    # Conversion from depth to volumetric flow rate
    depth2flow = dx * dy / dt

    # Flow Maps (converted below to vol. flow units (m^3/hr))
    qie_vol = qie * depth2flow
    qse_vol = qse * depth2flow
    # Baseflow
    qb_vol = qb * dx

    # Flux Maps
    ET_vol = ET * depth2flow * dt  # converted to depth units (m)
    PPT_vol = PPT * depth2flow * dt  # converted to depth units (m)

    # Convert storage terms to vol. rates (m^3/hr)
    # Conversion between mm and meters
    mm2m = 1000.0

    # Convert storage change maps to equivalent vol. rates (m^3/hr)
    dSD_vol = dSD * depth2flow
    dSuz_vol = dSuz * depth2flow
    dSrz_vol = dSrz * depth2flow
    dSWE_vol = dSWE / mm2m * depth2flow

    # Sum of fluxes over mapped domain
    # Runoff
    cum_qie_vol = np.nansum(qie_vol)
    cum_qse_vol = np.nansum(qse_vol)
    cum_qb_vol = np.nansum(qb_vol)
    cum_runoff_vol = cum_qse_vol + cum_qb_vol + cum_qie_vol

    # Fluxes
    cum_ET_vol = np.nansum(ET_vol)
    cum_PPT_vol = np.nansum(PPT_vol)

    # Change in storage (m^3/hr)
    cum_dSD_vol = np.nansum(dSD_vol)
    cum_dSuz_vol = np.nansum(dSuz_vol)
    cum_dSrz_vol = np.nansum(dSrz_vol)
    cum_dSWE_vol = np.nansum(dSWE_vol)

    # Convert to mass fluxes (kg/hr)
    dSD_massflux = cum_dSD_vol * rhow
    dSuz_massflux = cum_dSuz_vol * rhow
    dSrz_massflux = cum_dSrz_vol * rhow
    dSWE_massflux = cum_dSWE_vol * rhow
    PPT_massflux = cum_PPT_vol * rhow
    ET_massflux = cum_ET_vol * rhow
    runoff_massflux = cum_runoff_vol * rhow

    # Compute mass balance residual
    # Total storage change (kg/hr)
    dSdt = -dSD_massflux + dSuz_massflux + dSrz_massflux + dSWE_massflux

    # Total change in fluxes (kg/hr)
    fluxes = PPT_massflux - ET_massflux - runoff_massflux

    # Mass balance residual (Normalized difference)
    if (dSdt + fluxes) == 0.0:
        mass_resid = 0.0
    else:
        mass_resid = (dSdt - fluxes) / (dSdt + fluxes)

    return float(mass_resid)

def step_mass_balance(model, step_idx: int, PPT0: np.ndarray):
    """
    Executes the basin-wide mass balance check at the end of each simulation time step.
    Parameters
    ----------
    model : Model
        Master container holding state, step_vars, accumulators, map_outputs, and control.
    step_idx : int
        Current 0-indexed timestep iteration.
    PPT0: np.ndarray
        distributed precipitation field (actually constant)
    """
    # Compute storage changes for balance check
    model.step_vars.dSD = model.step_vars.SD_new - model.state.SD
    model.step_vars.dSuz = model.step_vars.Suz_new - model.state.Suz
    model.step_vars.dSrz = model.step_vars.Srz_new - model.state.Srz
    model.step_vars.dSWE = model.step_vars.SWE_new - model.state.SWE

    # Perform mass balance check and generate error flag if out of bounds
    mass_residual = mass_balance_check(
        PPT=PPT0,
        ET=model.step_vars.ET_out,
        qse=model.step_vars.qse_new,
        qb=model.step_vars.qb_new,
        qie=model.step_vars.qie_new,
        dx=model.control.dx,
        dy=model.control.dy,
        dSD=model.step_vars.dSD,
        dSuz=model.step_vars.dSuz,
        dSrz=model.step_vars.dSrz,
        dSWE=model.step_vars.dSWE,
        dt=model.control.dt,
        rhow=model.constants.rhow,
    )

    # Check whether mass balance residual is sufficiently close to zero
    if abs(mass_residual) > model.params.mass_balance_tolerance:
        print(
            f"Mass balance error out of bounds, i.e.: "
            f"{mass_residual * 100:.6f} % at time step: {step_idx} "
            f", compared to tolerance of: "
            f"{model.params.mass_balance_tolerance * 100:.6f}% \n"
        )

