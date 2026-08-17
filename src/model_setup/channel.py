import sys
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

def matlab_ismember(A, B, zero_based=True):
    """
    Replicates MATLAB [tf, loc] = ismember(A, B) for sorted array B (like Imask).
    """
    # Perform binary search
    idx = np.searchsorted(B, A)
    
    # Clamp indices to prevent out-of-bound array checks
    idx_clamped = np.clip(idx, 0, len(B) - 1)
    
    # Confirm exact value match
    mask_match = (B[idx_clamped] == A)
    
    if zero_based:
        # Python 0-based: matches -> 0 to N-1 | non-matches -> -1
        loc = np.where(mask_match, idx, -1)
    else:
        # MATLAB 1-based: matches -> 1 to N   | non-matches -> 0
        loc = np.where(mask_match, idx + 1, 0)
        
    return mask_match, loc

# New version that returns 2D row/col index tuples isntead of 1D Fortran order index vectors
def flow_network(flowdir: np.ndarray | sp.coo_matrix | sp.csr_matrix, mask: np.ndarray
                 ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """
    Determine upstream/downstream pixel pairs and outlet index from flow direction matrix.
    Replicates MATLAB's equivalent function exactly (1D Fortran indices) up until the 
    last step, where everything is converted from 1D Fortran indices to 2D C index arrays.

    NOTE:
    Input rows and cols MUST BE Fortran order, will update to accept C order in the future.

    Returns 0-based 2D index tuples (row_indices, col_indices) for Iupstream, Idownstream,
    and a 2D coordinate tuple (row, col) for Ioutlet.
    """
    # Ensure COO format
    if not isinstance(flowdir, (sp.coo_matrix, sp.coo_array)):
        flowdir = flowdir.tocoo()

    # 1. Extract row (U) and column (D) indices where flowdir == 1 (in 1D/Fortran order)
    if sp.issparse(flowdir):
        ones_mask = flowdir.data == 1
        U, D = flowdir.row[ones_mask], flowdir.col[ones_mask]
    else:
        U, D = np.where(flowdir == 1)

    # 2. Identify pixels in basin (using 1D/Fortran order)
    index_0 = False # work with 1-based indexing instead of 0-based for testing and validation with MATLAB
    if index_0:
        remove_val = -1
    else:
        remove_val = 0
    Imask = np.where(mask.ravel(order='F') == 1)[0]

    # Basin pixels with flowdir=1 using a replicate MATLAB ismember() function
    _, b = matlab_ismember(U, Imask, zero_based=index_0)
    _, bb = matlab_ismember(D, Imask, zero_based=index_0)
    
    # Remove any pixels that are not in the basin
    remove_mask = (b != remove_val) ^ (bb != remove_val)
    b[remove_mask] = remove_val
    bb[remove_mask] = remove_val

    # Extract upstream and downstream indices for basin pixels only
    UU = U[b > remove_val]
    DD = D[bb > remove_val]

    Iupstream = np.unravel_index(UU, mask.shape, order='F')
    Idownstream = np.unravel_index(DD, mask.shape, order='F')

    I = ~np.isin(DD, UU)
    Ioutlet_1d = DD[I][0] if np.any(I) else None # specifying DD[I][0] replicates 'first' in Ioutlet=DD(find(I,1,'first'));

    if Ioutlet_1d is None:
        sys.exit("Unable to find valid outlet point.")
    else:
        r_out, c_out = np.unravel_index(Ioutlet_1d, mask.shape, order="F")
        Ioutlet = (int(r_out), int(c_out))

    return Iupstream, Idownstream, Ioutlet

def flow_network_old(flowdir: np.ndarray | sp.coo_matrix | sp.csr_matrix, mask: np.ndarray
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