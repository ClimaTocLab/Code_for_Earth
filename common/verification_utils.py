import numpy as np
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt
import os.path
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def reorganize_table(ds_points):
    shape = ds_points.shape
    table = np.full(shape, np.nan)
    dims = ds_points.dims
    
    for ist in range(len(ds_points.station)):
        table[:,ist] = ds_points.isel(station=ist).values
    df = pd.DataFrame(data=table, index=ds_points.time.values, columns=ds_points.station.values)
    return df

def extract_nearest_gridpoint(grid, stations_code, stations_longitude, stations_latitude):
    var = list(grid.data_vars.keys())[0]

    grid_coords = list(grid.coords.keys())
    if "longitude" in grid_coords and "latitude" in grid_coords:
        xx = xr.DataArray(stations_longitude.values, dims=["station"], coords = {"station":stations_code})
        yy = xr.DataArray(stations_latitude.values, dims=["station"], coords = {"station":stations_code})
        gridpoints = grid[var].sel(longitude=xx, latitude=yy, method="nearest")
    else:
        AssertionError("No coordinates named longitude or latitude in the grid")

    gridpoints_points_df = reorganize_table(gridpoints)

    return gridpoints_points_df

def fairmode_metrics(obs, sim, measurement_uncertainty=0.25, return_global=True):
    """
    Compute FAIRMODE + extended metrics for multiple stations.
    Parameters:
    obs (pd.DataFrame): Observed values (index = time, columns = station names)
    sim (pd.DataFrame): Simulated values (same shape and structure)
    measurement_uncertainty (float): Relative uncertainty (default is 0.25)
    return_global (bool): If True, compute aggregated metrics across all stations
    Returns:
    metrics_df (pd.DataFrame): Metrics per station
    global_metrics (pd.Series): Aggregated metrics across all stations (if return_global=True)
    """
    # Align both DataFrames
    obs, sim = obs.align(sim, join='inner', axis=0)
    stations = obs.columns.intersection(sim.columns)
    results = []
    for station in stations:
        o = obs[station]
        s = sim[station]
        valid = o.notna() & s.notna()
        o = o[valid]
        s = s[valid]
        if len(o) == 0:
            continue
        # FAIRMODE metrics
        bias = (s - o).mean()
        nmb = 100 * bias / o.mean()
        nme = 100 * (np.abs(s - o)).mean() / o.mean()
        rmse = np.sqrt(((s - o) ** 2).mean())
        fac2 = ((s / o).between(0.5, 2)).mean() * 100
        corr = np.corrcoef(o, s)[0, 1] if len(o) > 1 else np.nan
        u = measurement_uncertainty
        mqi = np.sqrt(((s - o) ** 2).mean()) / (u * np.sqrt(o.mean() ** 2 + (s - o).var()))
        mqo_passed = mqi < 1
        # Extra metrics
        mae = mean_absolute_error(o, s)
        mse = mean_squared_error(o, s)
        # r2 = r2_score(o, s) if len(o) > 1 else np.nan
        # PSNR / SSIM (serie temporal como 1D)
        try:
            data_range = np.nanmax(o) - np.nanmin(o)
            psnr_val = psnr(o.values, s.values, data_range=data_range) if data_range > 0 else np.nan
            ssim_val = ssim(o.values, s.values, data_range=data_range) if data_range > 0 else np.nan
        except Exception:
            psnr_val, ssim_val = np.nan, np.nan
        results.append({
            "Station": station,
            "Bias": bias,
            "NMB [%]": nmb,
            "NME [%]": nme,
            "RMSE": rmse,
            "FAC2 [%]": fac2,
            "Correlation (R)": corr,
            "MQI": mqi,
            "Pass MQO": mqo_passed,
            "MAE": mae,
            "MSE": mse,
            # "R²": r2,
            "PSNR": psnr_val,
            "SSIM": ssim_val
        })
    metrics_df = pd.DataFrame(results).set_index("Station")
    return metrics_df

def compute_spatiotemporal_metrics(obs: pd.DataFrame, sim: pd.DataFrame):
    """
    obs, sim: DataFrames con forma (tiempo, estaciones)
    """
    # Asegurar alineación
    obs, sim = obs.align(sim, join="inner")
    # --- Espaciales (dimensión tiempo, calculando sobre estaciones) ---
    spatial_rmse = []
    spatial_corr = []
    spatial_mae = []
    spatial_mse = []
    for t in obs.index:
        o_t = obs.loc[t].dropna()
        s_t = sim.loc[t].dropna()
        valid = o_t.index.intersection(s_t.index)
        if len(valid) > 1:
            o_t = o_t.loc[valid]
            s_t = s_t.loc[valid]
            spatial_rmse.append(np.sqrt(mean_squared_error(o_t, s_t)))
            spatial_mse.append(mean_squared_error(o_t, s_t))
            spatial_mae.append(mean_absolute_error(o_t, s_t))
            spatial_corr.append(np.corrcoef(o_t, s_t)[0, 1])
        else:
            spatial_rmse.append(np.nan)
            spatial_mse.append(np.nan)
            spatial_mae.append(np.nan)
            spatial_corr.append(np.nan)
    # --- Temporales (dimensión estaciones, calculando sobre tiempo) ---
    temporal_rmse = []
    temporal_corr = []
    temporal_mae = []
    temporal_mse = []
    for st in obs.columns:
        o_s = obs[st].dropna()
        s_s = sim[st].dropna()
        valid = o_s.index.intersection(s_s.index)
        if len(valid) > 1:
            o_s = o_s.loc[valid]
            s_s = s_s.loc[valid]
            temporal_rmse.append(np.sqrt(mean_squared_error(o_s, s_s)))
            temporal_mse.append(mean_squared_error(o_s, s_s))
            temporal_mae.append(mean_absolute_error(o_s, s_s))
            temporal_corr.append(np.corrcoef(o_s, s_s)[0, 1])
        else:
            temporal_rmse.append(np.nan)
            temporal_mse.append(np.nan)
            temporal_mae.append(np.nan)
            temporal_corr.append(np.nan)
    # --- Agregados ---
    results = {
        "Spatial": {
            "RMSE_mean": np.nanmean(spatial_rmse),
            "MSE_mean": np.nanmean(spatial_mse),
            "MAE_mean": np.nanmean(spatial_mae),
            "Corr_mean": np.nanmean(spatial_corr),
        },
        "Temporal": {
            "RMSE_mean": np.nanmean(temporal_rmse),
            "MSE_mean": np.nanmean(temporal_mse),
            "MAE_mean": np.nanmean(temporal_mae),
            "Corr_mean": np.nanmean(temporal_corr),
        }
    }
    return results, {
        "Spatial_RMSE": spatial_rmse,
        "Spatial_MSE": spatial_mse,
        "Spatial_MAE": spatial_mae,
        "Spatial_Corr": spatial_corr,
        "Temporal_RMSE": temporal_rmse,
        "Temporal_MSE": temporal_mse,
        "Temporal_MAE": temporal_mae,
        "Temporal_Corr": temporal_corr
    }
if __name__ == "__main__":

    # observations:
    path_obs_us = "/media/cide/datasets/CAMS/US_data/Observations_US/"
    file_us = "pm25_US_2019-2025.csv"
    file_us_metad = "metadata-pm25_US_2019-2025.csv"

    df = pd.read_csv(path_obs_us + file_us, index_col=0, header=0, parse_dates=True)
    df_stations = pd.read_csv(path_obs_us + file_us_metad, index_col=0).drop_duplicates("Sampling Point Id")

    # grid:
    path_grid = "/media/cide/datasets/CAMS/inferencias/u-net/"
    # file_cams_europe = "CAMS_regional_forecast_2023-2025.nc"
    file_grid = "preds_targets_lowres_US.nc"
    var_name = "unet"
    ds_grid = xr.open_dataset(path_grid + file_grid)["preds"].to_dataset(name=var_name)

    df_res = extract_nearest_gridpoint(ds_grid, df_stations["Sampling Point Id"], df_stations["Longitude"], df_stations["Latitude"])

    res, metrics_df_st  = compute_spatiotemporal_metrics(df, df_res)