import warnings
import numpy as np
import scipy.sparse as sp

def manning_roughness_mean_to_pixel(n_mean: float, slope_deg: np.ndarray) -> np.ndarray:
    """Distribute mean Manning roughness over the basin based on local slope."""
    slope_rad = np.radians(slope_deg)
    slope_deg_mean = float(np.nanmean(slope_deg))
    slope_rad_mean = np.radians(slope_deg_mean)

    tan_slope = np.tan(slope_rad)
    tan_slope_mean = np.tan(slope_rad_mean)

    if tan_slope_mean == 0:
        return np.full_like(slope_deg, n_mean)

    return n_mean * (tan_slope / tan_slope_mean) ** (1.0 / 3.0)


def channel_width(flowacc: np.ndarray, alpha: float, c: float, dx: float, dy: float) -> np.ndarray:
    """Compute distributed channel width (m) across the basin using scaling power law."""
    m2km = 1000.0
    # Catchment area in km^2
    ai = (flowacc * dx * dy) / (m2km**2)
    # Channel width in meters
    return alpha * (ai**c)

import numpy as np
import scipy.sparse as sp

# New version that is Python-based (2D row/col indices, C order)
def flow_network(flowdir: np.ndarray | sp.coo_matrix | sp.csr_matrix, mask: np.ndarray
                 ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """
    Determine upstream/downstream pixel pairs and outlet index from flow direction matrix.

    Returns 0-based 2D index tuples (row_indices, col_indices) for Iupstream, Idownstream,
    and a 2D coordinate tuple (row, col) for Ioutlet.
    """
    # Ensure COO format
    if not isinstance(flowdir, (sp.coo_matrix, sp.coo_array)):
        flowdir = flowdir.tocoo()

    # 1. Extract row (U) and column (D) indices where flowdir == 1
    if sp.issparse(flowdir):
        ones_mask = flowdir.data == 1
        U, D = flowdir.row[ones_mask], flowdir.col[ones_mask]
    else:
        U, D = np.where(flowdir == 1)

    # 2. Identify active basin pixels using C-contiguous memory layout
    mask_flat = mask.ravel(order="C")
    in_basin = mask_flat == 1.0

    # 3. Keep only pairs where both upstream and downstream pixels are inside the basin
    valid_pairs = in_basin[U] & in_basin[D]
    UU_1d = U[valid_pairs]
    DD_1d = D[valid_pairs]

    # 4. Outlet pixel: first downstream node (in DD order) that is not an upstream node
    outlet_candidates = DD_1d[~np.isin(DD_1d, UU_1d)]
    Ioutlet_1d = int(outlet_candidates[0]) if len(outlet_candidates) > 0 else -1

    # Convert 1D C-order indices to 2D row/col coordinate tuples
    Iupstream = np.unravel_index(UU_1d, mask.shape, order="C")
    Idownstream = np.unravel_index(DD_1d, mask.shape, order="C")

    if Ioutlet_1d >= 0:
        r_out, c_out = np.unravel_index(Ioutlet_1d, mask.shape, order="C")
        Ioutlet = (int(r_out), int(c_out))
    else:
        Ioutlet = (-1, -1)

    return Iupstream, Idownstream, Ioutlet


# Old version that is MATLAB-based (1D liear indices, Fortran order)
def flow_network_old(flowdir: np.ndarray | sp.coo_matrix | sp.csr_matrix, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Determine upstream/downstream pixel pairs and outlet index from flow direction matrix.

    Returns 0-based 1D linear indices for Iupstream, Idownstream, and Ioutlet.
    """
    # Ensure COO format
    if isinstance(flowdir, (sp.coo_matrix, sp.coo_array)):
        # Matrix is already COO
        pass
    else:
        flowdir = flowdir.tocoo()

    # 1. Extract row (U) and column (D) indices where flowdir == 1
    if sp.issparse(flowdir):
        ones_mask = flowdir.data == 1
        U, D = flowdir.row[ones_mask], flowdir.col[ones_mask]
    else:
        U, D = np.where(flowdir == 1)

    # 2. Identify active basin pixels using Fortran ('F') memory layout to match MATLAB
    mask_flat = mask.ravel(order="F")
    in_basin = mask_flat == 1.0

    # 3. Keep only pairs where both upstream and downstream pixels are inside the basin
    valid_pairs = in_basin[U] & in_basin[D]
    UU = U[valid_pairs]
    DD = D[valid_pairs]

    # 4. Outlet pixel: first downstream node (in DD order) that is not an upstream node
    outlet_candidates = DD[~np.isin(DD, UU)]
    Ioutlet = int(outlet_candidates[0]) if len(outlet_candidates) > 0 else -1

    return UU, DD, Ioutlet

def derive_channel_properties(model) -> None:
    """Derive channel width, Manning's n, bed slope, and routing flow network."""
    control = model.control
    spatial = model.spatial
    network = model.network
    params = model.params
    mask = spatial.maskNaN

    # 1. Manning roughness map
    network.manning_n = (
        manning_roughness_mean_to_pixel(params.manning_n_mean, spatial.slope_deg)
        * mask
    )

    # 2. Channel width map
    network.width = (
        channel_width(
            flowacc=spatial.flowacc,
            alpha=params.channel_width_coeff_alpha,
            c=params.channel_width_exponent_c,
            dx=control.dx,
            dy=control.dy,
        )
        * mask
    )

    # Resolution warning check
    max_width = float(np.nanmax(network.width))
    if max_width > control.dx:
        warnings.warn(
            f"MOD-WET warning: Maximum computed stream width ({max_width:.2f} m) "
            f"exceeds DEM resolution dx ({control.dx:.2f} m). "
            "May want to consider coarsening DEM to make more consistent.",
            UserWarning,
        )

    # 3. Converted bed slope
    network.bed_slope = np.tan(np.radians(spatial.slope_deg)) * mask

    # 4. Flow network routing pairs
    network.Iupstream, network.Idownstream, network.Ioutlet = flow_network(
        network.flowdir, mask
    )