import numpy as np

def step_update_state_and_maps(model, step_idx: int, map_t_idx: int):
    """
    Updates active model states for the next timestep, processes time-averaged/summed
    map outputs when the storage interval is reached, and resets accumulators.

    Parameters
    ----------
    model : Model
        Master container holding state, step_vars, accumulators, map_outputs, and control.
    step_idx : int
        Current 0-indexed timestep iteration.
    map_t_idx : int
        Current 0-indexed array index for MapOutputs (n_map dimension).
    """
    # Update state variables for next time step
    model.state.Tsurf = model.step_vars.Tsurf_new.copy()
    model.state.Td = model.step_vars.Td_new.copy()
    model.state.SWE = model.step_vars.SWE_new.copy()
    model.state.Srz = model.step_vars.Srz_new.copy()
    model.state.Suz = model.step_vars.Suz_new.copy()
    model.state.SD = model.step_vars.SD_new.copy()
    model.state.snowdens = model.step_vars.snowdens_new.copy()
    model.state.snowdepth = model.step_vars.snowdepth_new.copy()
    model.state.snowfrac = model.step_vars.snowfrac_new.copy()

    # Compute time averages and re-initialize cumulative sum arrays
    # Convert 0-indexed step_idx to 1-based step number for modulo check
    if (step_idx + 1) % model.control.map_frq2store == 0:
        denom = float(model.control.map_frq2store)
        idx = map_t_idx  # Slot index in MapOutputs 3D arrays (n_map, nx, ny)

        # --- Time-Averaged States ---
        model.map_outputs.Srz[idx] = model.accumulators.cumulSrz / denom
        model.map_outputs.Suz[idx] = model.accumulators.cumulSuz / denom
        model.map_outputs.SD[idx] = model.accumulators.cumulSD / denom
        model.map_outputs.Tsurf[idx] = model.accumulators.cumulTsurf / denom
        model.map_outputs.Td[idx] = model.accumulators.cumulTd / denom
        model.map_outputs.SWE[idx] = model.accumulators.cumulSWE / denom
        model.map_outputs.snowdepth[idx] = model.accumulators.cumulsnowdepth / denom
        model.map_outputs.snowdens[idx] = model.accumulators.cumulsnowdens / denom
        model.map_outputs.snowfrac[idx] = model.accumulators.cumulsnowfrac / denom

        # Save days since last snowfall if defined in MapOutputs
        model.map_outputs.NDayLastSnow[idx] = model.state.NDayLastSnow

        # --- Time-Averaged Fluxes ---
        model.map_outputs.Rn[idx] = model.accumulators.cumulRn / denom
        model.map_outputs.LE[idx] = model.accumulators.cumulLE / denom
        model.map_outputs.ET[idx] = model.accumulators.cumulET / denom
        model.map_outputs.H[idx] = model.accumulators.cumulH / denom
        model.map_outputs.Rlup[idx] = model.accumulators.cumulRlup / denom

        # --- Time-Averaged Forcings ---
        model.map_outputs.Tair[idx] = model.accumulators.cumulTair / denom
        model.map_outputs.albedo[idx] = model.accumulators.cumulalbedo / denom
        model.map_outputs.Rs[idx] = model.accumulators.cumulRs / denom
        model.map_outputs.Rldown[idx] = model.accumulators.cumulRldown / denom
        model.map_outputs.qair[idx] = model.accumulators.cumulqair / denom
        model.map_outputs.Psfc[idx] = model.accumulators.cumulPsfc / denom
        model.map_outputs.PPT[idx] = model.accumulators.cumulPPT  # Summed depth

        # --- Period-Summed Fluxes (Depth over period) ---
        model.map_outputs.qie[idx] = model.accumulators.cumulqie
        model.map_outputs.qse[idx] = model.accumulators.cumulqse
        model.map_outputs.qb[idx] = model.accumulators.cumulqb
        model.map_outputs.qv[idx] = model.accumulators.cumulqv
        model.map_outputs.infil[idx] = model.accumulators.cumulinfil
        model.map_outputs.snowmelt[idx] = model.accumulators.cumulsnowmelt

        # Re-initialize cumulative counters to zero
        model.accumulators.reset()