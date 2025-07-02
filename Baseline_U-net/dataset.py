import torch
import xarray as xr
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import numpy as np

# --- Funciones para preparar datos ---

def upsample_to_hr(tensor_lr: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    tensor_lr = tensor_lr.unsqueeze(1).float()
    tensor_hr = F.interpolate(tensor_lr, size=target_hw, mode='bilinear', align_corners=True)
    return tensor_hr  # [T, 1, H_hr, W_hr]

def upsample_static_to_hr(tensor_static: torch.Tensor, target_hw: tuple[int, int], T: int) -> torch.Tensor:
    if tensor_static.ndim == 2:
        tensor_static = tensor_static.unsqueeze(0)  # [1, H_static, W_static]
    tensor_static = tensor_static.unsqueeze(0).float()  # [1, C_static, H_static, W_static]
    tensor_static_hr = F.interpolate(tensor_static, size=target_hw, mode='bilinear', align_corners=True)  # [1, C_static, H_hr, W_hr]
    tensor_static_hr = tensor_static_hr.repeat(T, 1, 1, 1)  # [T, C_static, H_hr, W_hr]
    return tensor_static_hr

def prepare_inputs_and_target(path_dynamic: str, path_static1: str, path_static2: str, path_target: str):
    # --- Load datasets ---
    ds_dyn = xr.open_dataset(path_dynamic)
    ds_target = xr.open_dataset(path_target)

    # --- Intersect and align time dimension ---
    common_times = np.intersect1d(ds_dyn['time'].values, ds_target['time'].values)
    if len(common_times) == 0:
        raise ValueError("? No hay timestamps comunes entre dinï¿½mico y target.")

    ds_dyn = ds_dyn.sel(time=common_times)
    ds_target = ds_target.sel(time=common_times)
    T = len(common_times)

    # --- Target shape and geo info ---
    lat = ds_target['latitude'].values
    lon = ds_target['longitude'].values
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    with open("data/geo_info.json", "w") as f:
        json.dump({
            'lat_min': float(lat_min),
            'lat_max': float(lat_max),
            'lon_min': float(lon_min),
            'lon_max': float(lon_max)
        }, f, indent=4)

    target_array = torch.tensor(ds_target['mdens'].values)  # [T, H, W]
    T, H_hr, W_hr = target_array.shape

    print(f"[{path_target}] pm2p5: min={torch.min(target_array)}, "
          f"max={torch.max(target_array)}, NaNs={torch.isnan(target_array).sum()}, shape={target_array.shape}")

    # --- Load dynamic variables and upsample ---
    dyn_vars = ['pm2p5', 'u10', 'v10', 't2m', 'blh', 'd2m']
    dyn_tensors = []
    dyn_stats = {}

    for var in dyn_vars:
        arr = ds_dyn[var].values  # [T, H_lr, W_lr]
        tensor_lr = torch.tensor(arr).float()
        tensor_hr = upsample_to_hr(tensor_lr, (H_hr, W_hr))  # [T, 1, H_hr, W_hr]

        if var == "pm2p5":
            tensor_hr = tensor_hr * 1e9  # convertir a g/m3

        print(f"[{path_dynamic}] {var} (upsampled): min={tensor_hr.min().item():.4f}, "
              f"max={tensor_hr.max().item():.4f}, NaNs={(~torch.isfinite(tensor_hr)).sum().item()}, "
              f"shape={tuple(tensor_hr.shape)}")

        mean = tensor_hr.mean().item()
        std = tensor_hr.std().item() + 1e-6
        dyn_stats[var] = {'mean': mean, 'std': std}

        tensor_hr = (tensor_hr - mean) / std
        dyn_tensors.append(tensor_hr)

    X_dyn = torch.cat(dyn_tensors, dim=1)  # [T, C_dyn, H_hr, W_hr]

    # --- Load and process static variables ---
    static_tensors = []
    static1_vars = ['z']
    static2_vars = ['__xarray_dataarray_variable__']

    for path_static, var_list in zip([path_static1, path_static2], [static1_vars, static2_vars]):
        ds_static = xr.open_dataset(path_static)
        for var in var_list:
            if var not in ds_static:
                print(f"?? Warning: Variable '{var}' not found in {path_static}")
                continue

            arr = ds_static[var].values
            arr = np.where(np.isfinite(arr), arr, np.nan)

            print(f"[{path_static}] {var}: min={np.nanmin(arr)}, max={np.nanmax(arr)}, "
                  f"NaNs={np.isnan(arr).sum()}, shape={arr.shape}")

            if np.isnan(arr).all():
                print(f"?? All values are NaN for static variable '{var}', skipping.")
                continue

            tensor_static = torch.tensor(arr).float()
            tensor_static = torch.nan_to_num(tensor_static, nan=0.0, posinf=0.0, neginf=0.0)
            tensor_static_hr = upsample_static_to_hr(tensor_static, (H_hr, W_hr), T)

            mean = tensor_static_hr.mean()
            std = tensor_static_hr.std() + 1e-6
            tensor_static_hr = (tensor_static_hr - mean) / std

            static_tensors.append(tensor_static_hr)

    if static_tensors:
        X_static = torch.cat(static_tensors, dim=1)
        X = torch.cat([X_dyn, X_static], dim=1)
    else:
        X = X_dyn

    y = target_array.unsqueeze(1).float()  # [T, 1, H_hr, W_hr]

    # --- Final cleanup ---
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    y_mean = y.mean()
    y_std = y.std() + 1e-6
    y = (y - y_mean) / y_std

    return X, y, y_mean, y_std, dyn_stats


# --- Dataset personalizado ---

class ClimateSuperResDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]