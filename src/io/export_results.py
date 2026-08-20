from pathlib import Path
import numpy as np
import xarray as xr

def save_simulation_results_netcdf(model, output_path: str | Path) -> None:
    """
    Exports simulation results (spatial map outputs, basin time-series, and pixel time-series)
    to a NetCDF4 file using xarray, matching the precision reduction and metadata standard.
    """
    output_path = Path(output_path)

    # 1. Construct Time Coordinates (days)
    end_day = model.control.start_day + model.control.n_days
    time_states_val = np.linspace(model.control.start_day, end_day, model.control.nt + 1, dtype=np.float32)
    time_fluxes_val = np.linspace(model.control.start_day, end_day, model.control.nt, dtype=np.float32)
    # time_maps_val = np.linspace(model.control.start_day, end_day, model.map_outputs.n_map, dtype=np.float32)

    # Convert time coordinates to time stamps
    first_date_string = model.forcing.start_date_time
    first_date = np.datetime64(first_date_string)
    time_states = first_date + (time_states_val * 86400).astype("timedelta64[s]")
    time_fluxes = first_date + (time_fluxes_val * 86400).astype("timedelta64[s]")
    # time_maps = first_date + (time_maps_val * 86400).astype("timedelta64[s]") 
    time_maps = first_date + (np.arange(0,model.map_outputs.n_map)).astype("timedelta64[D]")
    print(time_maps)

    # 2. Extract Spatial Coordinates
    northing = (
        model.spatial.northing.astype(np.float32)
        if model.spatial.northing is not None
        else np.arange(model.control.nx, dtype=np.float32)
    )
    easting = (
        model.spatial.easting.astype(np.float32)
        if model.spatial.easting is not None
        else np.arange(model.control.ny, dtype=np.float32)
    )

    # latitude = (
    #     model.spatial.lat.astype(np.float32)
    #     if model.spatial.lat is not None
    #     else np.arange(model.control.nx, dtype=np.float32)
    # )
    # longitude = (
    #     model.spatial.lon.astype(np.float32)
    #     if model.spatial.lon is not None
    #     else np.arange(model.control.ny, dtype=np.float32)
    # )

    n_special = model.time_series.n_special_pixels
    n_stream = model.time_series.n_stream_pixels

    data_vars = {}

    # 3. Process 3D Map Outputs (time_maps, northing, easting)
    map_meta = {
        "Srz": ("Rootzone soil moisture storage", "m"),
        "Suz": ("Unsaturated zone storage", "m"),
        "SD": ("Saturation deficit", "m"),
        "Tsurf": ("Surface temperature", "K"),
        "SWE": ("Snow water equivalent", "m"),
        "snowdepth": ("Snow depth", "m"),
        "snowdens": ("Snow density", "kg/m^3"),
        "snowfrac": ("Snow cover fraction", "-"),
        "Td": ("Deep soil temperature", "K"),
        "snowmelt": ("Period snowmelt depth", "m"),
        "Rn": ("Net radiation", "W/m^2"),
        "LE": ("Latent heat flux", "W/m^2"),
        "ET": ("Evapotranspiration rate", "m/h"),
        "H": ("Sensible heat flux", "W/m^2"),
        "qie": ("Infiltration excess runoff", "m"),
        "qse": ("Saturation excess runoff", "m"),
        "qb": ("Baseflow", "m"),
        "qv": ("Recharge to saturated zone", "m"),
        "infil": ("Infiltration depth", "m"),
        "Rlup": ("Upwelling longwave radiation", "W/m^2"),
        "Tair": ("Air temperature", "K"),
        "albedo": ("Surface albedo", "-"),
        "Rldown": ("Downwelling longwave radiation", "W/m^2"),
        "Rs": ("Solar radiation", "W/m^2"),
        "qair": ("Specific humidity", "kg/kg"),
        "Psfc": ("Surface atmospheric pressure", "Pa"),
        "PPT": ("Precipitation depth", "m"),
    }

    for var_name, (long_name, units) in map_meta.items():
        arr = getattr(model.map_outputs, var_name)
        if arr is not None:
            data_vars[f"map_{var_name}"] = (
                ("time_maps", "northing", "easting"),
                arr.astype(np.float32),
                {"long_name": long_name, "units": units},
            )

    # Save NDayLastSnow map if present
    if hasattr(model.map_outputs, "NDayLastSnow"):
        arr = getattr(model.map_outputs, "NDayLastSnow")
        if arr is not None:
            data_vars["map_NDayLastSnow"] = (
                ("time_maps", "northing", "easting"),
                arr.astype(np.float32),
                {"long_name": "Days since last major snowfall", "units": "days"},
            )

    # 4. Process 1D Basin-Averaged States (time_states)
    basin_state_meta = {
        "Srz": ("Basin-average rootzone soil moisture", "m"),
        "Suz": ("Basin-average unsaturated storage", "m"),
        "SD": ("Basin-average saturation deficit", "m"),
        "Tsurf": ("Basin-average surface temperature", "K"),
        "SWE": ("Basin-average snow water equivalent", "m"),
        "snowdepth": ("Basin-average snow depth", "m"),
        "snowdens": ("Basin-average snow density", "kg/m^3"),
        "snowfrac": ("Basin-average snow cover fraction", "-"),
        "Td": ("Basin-average deep soil temperature", "K"),
    }

    for var_name, (long_name, units) in basin_state_meta.items():
        arr = getattr(model.time_series, var_name)
        if arr is not None:
            data_vars[f"basin_{var_name}"] = (
                ("time_states",),
                arr.astype(np.float32),
                {"long_name": long_name, "units": units},
            )

    # 5. Process 1D Basin-Averaged Fluxes & Forcings (time_fluxes)
    basin_flux_meta = {
        "snowmelt": ("Basin-average snowmelt rate", "m/h"),
        "Rn": ("Basin-average net radiation", "W/m^2"),
        "LE": ("Basin-average latent heat flux", "W/m^2"),
        "ET": ("Basin-average evapotranspiration rate", "m/h"),
        "H": ("Basin-average sensible heat flux", "W/m^2"),
        "qie": ("Basin-average infiltration excess runoff rate", "m/h"),
        "qse": ("Basin-average saturation excess runoff rate", "m/h"),
        "qb": ("Basin-average baseflow rate", "m/h"),
        "qv": ("Basin-average recharge rate to saturated zone", "m/h"),
        "outlet_hydrograph": ("Basin outlet hydrograph", "m^3/s"),
        "Rlup": ("Basin-average upwelling longwave radiation", "W/m^2"),
        "infil": ("Basin-average infiltration rate", "m/h"),
        "Rs": ("Basin-average solar radiation", "W/m^2"),
        "Tair": ("Basin-average air temperature", "K"),
        "albedo": ("Basin-average surface albedo", "-"),
        "qair": ("Basin-average specific humidity", "kg/kg"),
        "Psfc": ("Basin-average surface pressure", "Pa"),
        "Rldown": ("Basin-average downwelling longwave radiation", "W/m^2"),
        "PPT": ("Basin-average precipitation rate", "m/h"),
    }

    for var_name, (long_name, units) in basin_flux_meta.items():
        arr = getattr(model.time_series, var_name)
        if arr is not None:
            data_vars[f"basin_{var_name}"] = (
                ("time_fluxes",),
                arr.astype(np.float32),
                {"long_name": long_name, "units": units},
            )

    # 6. Process Special Pixel Outputs (3D: time x northing x easting)
    if model.time_series.n_special_pixels > 0:
        pixel_state_meta = {
            "pixel_Srz": ("Special pixel rootzone moisture", "m"),
            "pixel_Suz": ("Special pixel unsaturated storage", "m"),
            "pixel_SD": ("Special pixel saturation deficit", "m"),
            "pixel_Tsurf": ("Special pixel surface temperature", "K"),
            "pixel_SWE": ("Special pixel snow water equivalent", "m"),
            "pixel_snowdepth": ("Special pixel snow depth", "m"),
            "pixel_snowdens": ("Special pixel snow density", "kg/m^3"),
            "pixel_snowfrac": ("Special pixel snow cover fraction", "-"),
            "pixel_Td": ("Special pixel deep soil temperature", "K"),
        }
        for var_name, (long_name, units) in pixel_state_meta.items():
            arr = getattr(model.time_series, var_name)
            if arr is not None:
                data_vars[var_name] = (
                    ("time_states", "northing", "easting"),
                    arr.astype(np.float32),
                    {"long_name": long_name, "units": units},
                )

        pixel_flux_meta = {
            "pixel_snowmelt": ("Special pixel snowmelt rate", "m/h"),
            "pixel_Rn": ("Special pixel net radiation", "W/m^2"),
            "pixel_LE": ("Special pixel latent heat flux", "W/m^2"),
            "pixel_ET": ("Special pixel evapotranspiration rate", "m/h"),
            "pixel_H": ("Special pixel sensible heat flux", "W/m^2"),
            "pixel_qie": (
                "Special pixel infiltration excess runoff rate",
                "m/h",
            ),
            "pixel_qse": ("Special pixel saturation excess runoff rate", "m/h"),
            "pixel_qb": ("Special pixel baseflow rate", "m/h"),
            "pixel_qv": ("Special pixel recharge rate", "m/h"),
            "pixel_Rlup": (
                "Special pixel upwelling longwave radiation",
                "W/m^2",
            ),
            "pixel_infil": ("Special pixel infiltration rate", "m/h"),
        }
        for var_name, (long_name, units) in pixel_flux_meta.items():
            arr = getattr(model.time_series, var_name)
            if arr is not None:
                data_vars[var_name] = (
                    ("time_fluxes", "northing", "easting"),
                    arr.astype(np.float32),
                    {"long_name": long_name, "units": units},
                )

    # 7. Process Stream Pixel Hydrographs (3D: time_fluxes x northing x easting)
    if model.time_series.n_stream_pixels > 0:
        arr = model.time_series.pixel_stream_hydrograph
        if arr is not None:
            data_vars["pixel_stream_hydrograph"] = (
                ("time_fluxes", "northing", "easting"),
                arr.astype(np.float32),
                {"long_name": "Stream pixel hydrograph", "units": "m^3/s"},
            )

    # 8. Add basin area as a stand-alone variable
    if model.params.basin_area is not None:
        data_vars["basin_area"] = (
                        ("x"), # x is a dummy coordiante variable
                        [float(model.params.basin_area)],
                        {"long_name": "Total watershed surface area", "units": "m^2"},
                    )

    # 9. Define Coordinates
    coords = {
        "northing": ("northing", northing, {"units": "m"}),
        "easting": ("easting", easting, {"units": "m"}),
        "time_maps": ("time_maps", time_maps),
        "time_states": ("time_states", time_states),
        "time_fluxes": ("time_fluxes", time_fluxes),
    }
    if n_special > 0:
        coords["special_pixel"] = ("special_pixel", np.arange(n_special))
    if n_stream > 0:
        coords["stream_pixel"] = ("stream_pixel", np.arange(n_stream))

    # 10. Extract Global Attributes
    attrs = {
        "dx": float(model.control.dx),
        "dy": float(model.control.dy),
        "dt": float(model.control.dt),
        "start_day": float(model.control.start_day),
        "n_days": int(model.control.n_days),
        "utmzone": str(model.params.utmzone),
    }

    if model.params.lat_mean is not None:
        attrs["lat_mean"] = float(model.params.lat_mean)
    if model.params.lon_mean is not None:
        attrs["lon_mean"] = float(model.params.lon_mean)
    # if model.params.basin_area is not None:
    #     attrs["basin_area"] = float(model.params.basin_area)

    # 11. Assemble Dataset and Export NetCDF4
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)

    print("MOD-WET simulation completed successfully.")
    print(f"Outputs stored in NetCDF format at: {output_path}")