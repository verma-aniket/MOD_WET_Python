import numpy as np

from src.chapter11 import TOPMODEL

def step_topmodel(model, f: np.ndarray, ETsoil: np.ndarray) -> None:
    """
    Executes a single timestep of the TOPMODEL subsurface soil moisture balance,
    saturation-excess runoff, and baseflow routine, storing updated fluxes
    and states in step_vars.

    Inputs:
        model : Model
            Master container holding state, step_vars, accumulators, map_outputs, and control.
        f: array
            infiltration rate
        ETsoil: array
            soil ET rate

    """
    # Call TOPMODEL and step through one time-step
    (
        Srz_new,
        Suz_new,
        SD_new,
        qv_new,
        qb_new,
        qse_new,
        Qv_new,
        Qb_new,
    ) = TOPMODEL(
        INFIL=f,
        ET=ETsoil,
        Srz0=model.state.Srz,
        Srzmax=model.spatial.Srzmax,
        Srzmin=model.spatial.Srzmin,
        Suz0=model.state.Suz,
        SD0=model.state.SD,
        T0=model.spatial.T0,
        slope_deg=model.spatial.slope_deg,
        dx=model.control.dx,
        mask=model.spatial.maskNaN,
        lambda_mean=model.params.lambda_mean,
        lambda_val=model.spatial.lambda_map,
        m=model.params.m,
        K0=model.spatial.K0,
        dt=model.control.dt,
    )

    # allocate topmodel results
    model.step_vars.Srz_new = Srz_new
    model.step_vars.Suz_new = Suz_new
    model.step_vars.SD_new = SD_new
    model.step_vars.qv_new = qv_new
    model.step_vars.qb_new = qb_new
    model.step_vars.qse_new = qse_new
    model.step_vars.Qv_new = Qv_new # not used 
    model.step_vars.Qb_new = Qb_new # not used

    