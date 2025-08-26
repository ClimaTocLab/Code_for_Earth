

def concat_global(path_input, path_output):
    import xarray as xr
    import os

    files_pattern = os.path.join(path_input, '*.nc')
    ds_merged = xr.open_mfdataset(files_pattern, combine='by_coords', decode_timedelta=True)
    ds_filtered = ds_merged.isel(forecast_period=slice(0, 8))
    ds_daily_avg = ds_filtered.mean(dim="forecast_period")
    ds_daily_avg = ds_daily_avg.rename({'forecast_reference_time': 'time'})

    # Copy global attributes
    ds_daily_avg.attrs = ds_filtered.attrs

    # Copy variable attributes
    for var in ds_filtered.data_vars:
        ds_daily_avg[var].attrs = ds_filtered[var].attrs

    ds_daily_avg.to_netcdf(path_output)
    print(ds_daily_avg)
    return ds_daily_avg


def concat_regional(path_in, path_out):
    import xarray as xr
    import pandas as pd
    import os

    path_list = os.listdir(path_in)
    ds_list = []

    for path in path_list:
        ds_out = xr.open_dataset(os.path.join(path_in, path), decode_timedelta=True)

        long_name = ds_out['time'].attrs.get('long_name', '')
        long_name = long_name.split("FORECAST time from ")[1]
        long_name = long_name[0:4]+"-"+long_name[4:6]+"-"+long_name[6:8]
        print(f"Base date extracted: {long_name}")

        base_date = pd.Timestamp(long_name)

        ds_with_datetime = ds_out.assign_coords(datetime=base_date + ds_out.time)

        ds_daily = ds_with_datetime.groupby("datetime.date").mean(dim="time")

        ds_daily = ds_daily.rename({'date': 'time'})

        ds_daily = ds_daily.squeeze(dim="level")

        ds_daily['time'] = pd.to_datetime(ds_daily['time'].values)

        ds_list.append(ds_daily)

    # Concatenate all datasets along 'time'
    ds_concat = xr.concat(ds_list, dim='time')

    # Sort by time just in case
    ds_concat = ds_concat.sortby('time')

    new_lon = ds_concat.longitude.where(ds_concat.longitude < 180, ds_concat.longitude - 360)
    ds_concat = ds_concat.assign_coords(longitude=new_lon).sortby("longitude")

    # Save concatenated dataset
    ds_concat.to_netcdf(path_out)

    print(ds_concat)
    return ds_concat


def new_regional_to_daily():
    import xarray as xr

    # Abrir el archivo original (3h)
    ds = xr.open_dataset("data/all_cams_ens_fc_pm2p5_level0_3h_2019_2025.nc")

    # Asegurarse que xarray reconoce el tiempo como datetime
    ds["time"] = xr.decode_cf(ds).time

    # Resamplear a resoluciï¿½n diaria y calcular la media
    ds_daily = ds.resample(time="1D").mean()

    # Guardar el nuevo archivo NetCDF
    ds_daily.to_netcdf(
        "data/all_cams_ens_fc_pm2p5_level0_daily_2019_2025.nc",
        engine="netcdf4",
        compute=True
    )


def regrid(pattern_path, slave_path, output_path):
    import xarray as xr
    import os

    # Open datasets
    pattern_ds = xr.open_dataset(pattern_path)
    print(pattern_ds)
    print("_" * 80)
    slave_ds = xr.open_dataset(slave_path)
    print(slave_ds)
    print("_" * 80)

    # Detect coordinate names in slave_ds
    slave_lat_name = 'latitude' if 'latitude' in slave_ds.coords else 'lat'
    slave_lon_name = 'longitude' if 'longitude' in slave_ds.coords else 'lon'

    # Temporarily rename to 'latitude'/'longitude' if needed
    if slave_lat_name != 'latitude' or slave_lon_name != 'longitude':
        slave_ds = slave_ds.rename({slave_lat_name: 'latitude', slave_lon_name: 'longitude'})

    # Destination coordinates from pattern
    new_lats = pattern_ds['latitude']
    new_lons = pattern_ds['longitude']

    # Interpolation
    regridded_vars = {}
    for var_name, var in slave_ds.data_vars.items():
        dims = var.dims
        if 'latitude' in dims and 'longitude' in dims:
            regridded_vars[var_name] = var.interp(latitude=new_lats, longitude=new_lons, method='linear')
        else:
            regridded_vars[var_name] = var  # Variables without lat/lon

    # Create new dataset
    new_ds = xr.Dataset(regridded_vars, coords={
        'latitude': new_lats,
        'longitude': new_lons
    })

    # Copy additional coordinates
    for coord in slave_ds.coords:
        if coord not in ['latitude', 'longitude']:
            new_ds = new_ds.assign_coords({coord: slave_ds.coords[coord]})

    # Copy global attributes
    new_ds.attrs = slave_ds.attrs

    # Copy variable attributes
    for var_name in regridded_vars:
        new_ds[var_name].attrs = slave_ds[var_name].attrs

    # Save to NetCDF file
    new_ds.to_netcdf(output_path)
    print(new_ds)

    return new_ds



def print_nc_in_map(file_path, name_var, time):

    import xarray as xr
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np

    ds = xr.open_dataset(file_path, decode_timedelta=True)

    if name_var not in ds.data_vars:
        raise ValueError(f"Variable '{name_var}' not found in dataset variables: {list(ds.data_vars)}")

    data = ds[name_var]

    if 'time' in data.dims:
        data = data.isel(time=time)

    # Detectar nombres posibles de lat y lon
    lat_name = None
    lon_name = None
    for coord in data.coords:
        if coord.lower() in ['lat', 'latitude']:
            lat_name = coord
        if coord.lower() in ['lon', 'longitude']:
            lon_name = coord

    if lat_name is None or lon_name is None:
        raise ValueError("No se encontraron coordenadas latitud o longitud en el dataset.")

    # Mostrar valores mï¿½nimos y mï¿½ximos de coordenadas para chequear orden
    print(f"Lat min/max: {np.min(data[lat_name].values)}, {np.max(data[lat_name].values)}")
    print(f"Lon min/max: {np.min(data[lon_name].values)}, {np.max(data[lon_name].values)}")

    # Ordenar explï¿½citamente por lat y lon
    data = data.sortby(lat_name)
    data = data.sortby(lon_name)

    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', cbar_kwargs={'label': data.name})

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    title = f"{data.name} - First time step" if 'time' in data.dims else data.name
    ax.set_title(title)

    plt.show()





#print_nc_in_map("data/CAMS_regional_forecast_2023-2025.nc","pm2p5_conc", 100)
#print_nc_in_map("data/CAMS_global_forecast_2015-2025.nc","pm2p5",100)
#print_nc_in_map("data/ETOPO_2022_v1_60s_N90W180_surface_global_regridded.nc","z", 0)
#print_nc_in_map("data/ETOPO_2022_v1_60s_N90W180_surface_regional_regridded.nc","z", 0)
#print_nc_in_map("data/GHS_population_spatial_resol_0.1_global_regridded.nc","__xarray_dataarray_variable__",0)
print_nc_in_map("/media/server/code4earth/US_data/Population_US/GHS_population_spatial_resol_0.1_eeuu_regridded_non_nan_v2.nc","population",0)
print_nc_in_map("/media/server/code4earth/US_data/Orography_US/LandElevation_lsmGHSPopulation_ETOPO_2022_v1_60s_N90W180_surface_US.nc","z", 100)
print_nc_in_map("Baseline_U-net/data/CAMS_global_forecast_US.nc","pm2p5", 10)



if __name__ == "__maino__":

    # Concatenate global
    concat_global(path_input="CAMS_global_forecast/CAMS_global_forecast/", path_output="CAMS_global_forecast_2015-2025.nc")
    print("=" * 80)

    # Concatenate regional
    concat_regional(path_in="CAMS_europe_forecast/CAMS_europe_forecast",path_out="CAMS_europe_forecast_2023-2025.nc")
    print("=" * 80)

    # Regrid population to global
    regrid(
        "CAMS_global_forecast_2015-2025.nc",
        "GHS_population_spatial_resol_0.1/GHS_population_spatial_resol_0.1.nc",
        "GHS_population_spatial_resol_0.1_regridded_.nc"
    )
    print("=" * 80)

    # Regrid topo to global
    regrid(
        "CAMS_global_forecast_2015-2025.nc",
        "ETOPO_2022_v1_60s_N90W180_surface/ETOPO_2022_v1_60s_N90W180_surface.nc",
        "ETOPO_2022_v1_60s_N90W180_surface_regridded.nc"
    )
    print("=" * 80)

    # Regrid population to regional
    regrid(
        "CAMS_europe_forecast_2023-2025.nc",
        "GHS_population_spatial_resol_0.1/GHS_population_spatial_resol_0.1.nc",
        "GHS_population_spatial_resol_0.1_regional_regridded_.nc"
    )
    print("=" * 80)

    # Regrid topo to regional
    regrid(
        "CAMS_europe_forecast_2023-2025.nc",
        "ETOPO_2022_v1_60s_N90W180_surface/ETOPO_2022_v1_60s_N90W180_surface.nc",
        "ETOPO_2022_v1_60s_N90W180_surface_regional_regridded.nc"
    )
    print("=" * 80)


