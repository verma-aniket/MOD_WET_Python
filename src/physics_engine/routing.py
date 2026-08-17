import numpy as np

from src.chapter11 import routing_celerity_check, routing_muskingum_cunge

# New Python-based routing function (uses 2D grid index tuples)
def step_routing(model, step_idx: int, ts_idx: int) -> None:
    """
    Executes the Muskingum-Cunge channel and overland flow routing module for
    a single simulation timestep, updates stream/outlet hydrographs, updates
    state/flux time-series, and accumulates flux maps using 2D grid index tuples.
    Parameters
    ----------
    model : Model
        Master container holding state, step_vars, accumulators, map_outputs, and control.
    step_idx : int
        Current 0-indexed timestep iteration.
    ts_idx : int
        Current 0-indexed array index for TimeSeries (nt dimension).
    """
    # Call routing to predict watershed hydrograph
    # Uses Muskingum-Cunge routing method
    # Set the new inflow to the runoff generated from model (current
    # timestep) and the inflows from upstream. Need to convert units to
    # m^3/s.
    if model.control.routing_time_step_flag == 1:
        # This allows for higher temporal resolution of routing (based on
        # computed ratio). Assumes the total runoff from each pixel is the
        # same over the whole timestep, but partitions it into equal amounts
        # based on the ratio.
        dummy1 = (
            (model.step_vars.qie_new + model.step_vars.qse_new) / model.control.dt
            + model.step_vars.qb_new / model.control.dx
        ) * (model.control.dx**2) / 3600.0
        dummy2 = model.step_vars.NEW_INFLOWS + dummy1

        dt_ratio = routing_celerity_check(
            INFLOW_old=model.step_vars.INFLOW_old,
            INFLOW_new=dummy2,
            OUTFLOW_old=model.step_vars.OUTFLOW_old,
            n=model.network.manning_n,
            bed_slope=model.network.bed_slope,
            dx=model.control.dx,
            dt=model.control.dt,
            width=model.network.width,
        )
    else:
        dt_ratio = 1
    
    # Loop through multiple routing time steps (if specified and necessary)
    dummy_sum = 0.0

    if model.params.stream_pixels and len(model.params.stream_pixels[0]) > 0:
        # preallocate outflow array for interior stream pixels
        dummy_sum_2 = np.zeros(len(model.params.stream_pixels[0]), dtype=float)
    else:
        dummy_sum_2 = None

    for irout_step in range(dt_ratio):
        RUNOFF_new = (
            (model.step_vars.qie_new + model.step_vars.qse_new) / model.control.dt
            + model.step_vars.qb_new / model.control.dx
        ) * (model.control.dx**2) / 3600.0 / dt_ratio
        INFLOW_new = model.step_vars.NEW_INFLOWS + RUNOFF_new

        # Call Muskingum-Cunge routing functions
        NEW_INFLOWS, NEW_OUTFLOWS = routing_muskingum_cunge(
            INFLOW_old=model.step_vars.INFLOW_old,
            INFLOW_new=INFLOW_new,
            OUTFLOW_old=model.step_vars.OUTFLOW_old,
            Iupstream=model.network.Iupstream,
            Idownstream=model.network.Idownstream,
            mask=model.spatial.maskNaN,
            n=model.network.manning_n,
            bed_slope=model.network.bed_slope,
            dx=model.control.dx,
            dt=model.control.dt / dt_ratio,
            width=model.network.width,
        )

        # Set the new values to old values for next time step
        model.step_vars.INFLOW_old = INFLOW_new
        model.step_vars.OUTFLOW_old = NEW_OUTFLOWS

        # this line is new - store NEW_INFLOWS for next iteration
        model.step_vars.NEW_INFLOWS = NEW_INFLOWS

        # Compute flows at interior stream pixels (accumulates flow over loop)
        if model.params.stream_pixels and len(model.params.stream_pixels[0]) > 0:
            st_rows, st_cols = model.params.stream_pixels
            dummy_sum_2 += NEW_OUTFLOWS[st_rows, st_cols]  # m^3/s

        # compute outlet hydrograph (accumulates flows over loop). Note:
        # The outlet is a special case. The "NEW_OUTFLOWS" variable cannot
        # be used directly because there is no "downstream" node for the
        # output to compute flow to. So instead the inflow is added
        # directly to the runoff from the outlet pixel itself. This may
        # introduce some (small) error as the flow is essentially just
        # translated (no storage) in the outlet pixel.
        out_r, out_c = model.network.Ioutlet
        dummy_sum += NEW_INFLOWS[out_r, out_c] + RUNOFF_new[out_r, out_c]  # m^3/s

    # Store outlet hydrograph
    Q_at_outlet = float(dummy_sum)

    if model.params.stream_pixels and len(model.params.stream_pixels[0]) > 0:
        # Store interior stream hydrographs
        Q_at_stream = dummy_sum_2

    # Store state/runoff flux results
    if (step_idx + 1) % model.control.timeseries_frq2store == 0:
        # These are basin-averaged, but instantaneous at time of saving
        model.time_series.Srz[ts_idx + 1] = float(np.nanmean(model.state.Srz))  # (m)
        model.time_series.Suz[ts_idx + 1] = float(np.nanmean(model.state.Suz))  # (m)
        model.time_series.SD[ts_idx + 1] = float(np.nanmean(model.state.SD))    # (m)

        model.time_series.qie[ts_idx] = float(np.nanmean(model.step_vars.qie_new / model.control.dt))   # (m/hr)
        model.time_series.qse[ts_idx] = float(np.nanmean(model.step_vars.qse_new / model.control.dt))   # (m/hr)
        model.time_series.qb[ts_idx] = float(np.nanmean(model.step_vars.qb_new / model.control.dx))     # (m/hr)
        model.time_series.qv[ts_idx] = float(np.nanmean(model.step_vars.qv_new / model.control.dt))     # (m/hr)

        model.time_series.outlet_hydrograph[ts_idx] = Q_at_outlet   # (m^3/s)

        # Special pixels/interior stream nodes
        if model.params.stream_pixels and len(model.params.stream_pixels[0]) > 0:
            # Store the stream hydrographs
            model.time_series.pixel_stream_hydrograph[ts_idx][st_rows, st_cols] = Q_at_stream  # (m^3/s)

        if model.params.special_pixels is not None and len(model.params.special_pixels[0]) > 0:
            sp_rows, sp_cols = model.params.special_pixels
            model.time_series.pixel_Srz[ts_idx + 1][sp_rows, sp_cols] = model.state.Srz[sp_rows, sp_cols]  # (m)
            model.time_series.pixel_Suz[ts_idx + 1][sp_rows, sp_cols] = model.state.Suz[sp_rows, sp_cols]  # (m)
            model.time_series.pixel_SD[ts_idx + 1][sp_rows, sp_cols] = model.state.SD[sp_rows, sp_cols]    # (m)

            model.time_series.pixel_qie[ts_idx][sp_rows, sp_cols] = model.step_vars.qie_new[sp_rows, sp_cols] / model.control.dt   # (m/hr)
            model.time_series.pixel_qse[ts_idx][sp_rows, sp_cols] = model.step_vars.qse_new[sp_rows, sp_cols] / model.control.dt   # (m/hr)
            model.time_series.pixel_qb[ts_idx][sp_rows, sp_cols] = model.step_vars.qb_new[sp_rows, sp_cols] / model.control.dx     # (m/hr)
            model.time_series.pixel_qv[ts_idx][sp_rows, sp_cols] = model.step_vars.qv_new[sp_rows, sp_cols] / model.control.dt     # (m/hr)

    # Accumulate mapped state/flux variables for later averaging/saving
    # Storage maps
    model.accumulators.cumulSrz += model.state.Srz  # (m)
    model.accumulators.cumulSuz += model.state.Suz  # (m)
    model.accumulators.cumulSD += model.state.SD    # (m)

    # Flux maps (Note: The values being added are instantaneous from every time step,
    # keep depths for these instead
    model.accumulators.cumulqie += model.step_vars.qie_new  # (m)
    model.accumulators.cumulqse += model.step_vars.qse_new  # (m)
    # Note: [qb]= m^2/hr so divide by pixel dimension to get m 
    model.accumulators.cumulqb += model.step_vars.qb_new / model.control.dx * model.control.dt # (m)
    model.accumulators.cumulqv += model.step_vars.qv_new    # (m)

