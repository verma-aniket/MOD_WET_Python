import numpy as np
import re
from pyproj import Transformer

def parse_utm_epsg(utmzone_str: str) -> int:
    """Parse any UTM zone string into its corresponding EPSG code."""
    # Matches zone number and optional band/hemisphere indicator
    match = re.match(r"^(\d{1,2})\s*([A-Za-z]+)?$", utmzone_str.strip())
    if not match:
        raise ValueError(f"Invalid UTM zone format: '{utmzone_str}'. Standard format is [<Zone Number> <Latitude Band Letter>] (e.g., 10 S, 32 T, 18 M)")
    zone = int(match.group(1))
    if not (1 <= zone <= 60):
        raise ValueError(f"UTM zone must be between 1 and 60, got {zone}")
    indicator = (match.group(2) or "N").upper()
    # 1. Handle explicit hemisphere words
    if indicator in ("SOUTH", "SOUTHERN"):
        is_south = True
    elif indicator in ("NORTH", "NORTHERN"):
        is_south = False
    # 2. Handle MGRS Latitude Bands (C through M = South; N through X = North)
    else:
        is_south = indicator[0] <= "M"
    return (32700 if is_south else 32600) + zone


def convert_utm_to_latlon(easting_1d: np.ndarray, northing_1d: np.ndarray, utm_zone_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert 1D Easting and Northing UTM vectors into 1D lat/lon vectors"""

    # Parse UTM zone string 
    epsg = parse_utm_epsg(utm_zone_str)

    transformer = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )

    # Create 2D meshgrid arrays from 1D coordinate vectors
    E, N = np.meshgrid(easting_1d, northing_1d)
    lon_2d, lat_2d = transformer.transform(E, N)

    # Collapse to 1D vectors
    lat_1d = np.mean(lat_2d, axis=1)  # Mean across columns
    lon_1d = np.mean(lon_2d, axis=0)  # Mean across rows

    return lat_1d, lon_1d

def derive_terrain_maps(model) -> None:
    """Derive terrain grids, convert UTM to Lat/Lon, and compute basin metadata."""
    # 1. Build NaN mask grid
    model.spatial.maskNaN = np.where(model.spatial.mask == 1, 1.0, np.nan)

    # 2. Convert angles to radians (masked to domain)
    model.spatial.slope_rad = np.radians(model.spatial.slope_deg)
    model.spatial.aspect_rad = np.radians(model.spatial.aspect_deg)

    # 3. Derive 1D Lat/Lon vectors from 1D Easting/Northing vectors
    if model.spatial.easting is not None and model.spatial.northing is not None:
        lat_1d, lon_1d = convert_utm_to_latlon(
            model.spatial.easting,
            model.spatial.northing,
            model.params.utmzone,
        )
        model.spatial.lat = lat_1d
        model.spatial.lon = lon_1d

        # 4. Compute mean lat/lon coordinates
        model.params.lat_mean = float(np.mean(lat_1d))
        model.params.lon_mean = float(np.mean(lon_1d))