import torch
import xarray as xr
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import numpy as np

# --- Functions for data preparation ---

def upsample_to_hr(tensor_lr: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """
    Upsample a low-resolution dynamic tensor [T, H, W] to high-resolution [T, 1, H_hr, W_hr].
    Uses bilinear interpolation.
    """
    tensor_lr = tensor_lr.unsqueeze(1).float()  # add channel dimension → [T, 1, H_lr, W_lr]
    tensor_hr = F.interpolate(tensor_lr, size=target_hw, mode='bilinear', align_corners=True)
    return tensor_hr


def upsample_static_to_hr(tensor_static: torch.Tensor, target_hw: tuple[int, int], T: int) -> torch.Tensor:
    """
    Upsample a static tensor [H, W] or [C, H, W] to match high resolution and expand across time.
    Returns shape [T, C, H_hr, W_hr].
    """
    if tensor_static.ndim == 2:  # single-channel static variable
        tensor_static = tensor_static.unsqueeze(0)  # → [1, H, W]
    tensor_static = tensor_static.unsqueeze(0).float()  # add batch dim → [1, C, H, W]

    # interpolate to HR grid
    tensor_static_hr = F.interpolate(tensor_static, size=target_hw, mode='bilinear', align_corners=True)

    # repeat across time steps
    tensor_static_hr = tensor_static_hr.repeat(T, 1, 1, 1)  # [T, C, H_hr, W_hr]
    return tensor_static_hr


def prepare_inputs_and_target(path_dynamic: str, path_static1: str, path_static2: str, path_target: str,
                              y_mean: float = None, y_std: float = None):
    """
    Prepares input (X) and target (y) tensors for training super-resolution climate models.

    Args:
        path_dynamic: path to dynamic variables (NetCDF).
        path_static1: path to first static dataset (NetCDF).
        path_static2: path to second static dataset (NetCDF).
        path_target: path to target pollutant concentrations (NetCDF).
        y_mean, y_std: optional precomputed normalization stats for target.

    Returns:
        X: torch.Tensor [T, C_total, H_hr, W_hr] – inputs (normalized).
        y: torch.Tensor [T, 1, H_hr, W_hr] – target (normalized).
        y_mean, y_std: normalization parameters for target.
        dyn_stats: dictionary with mean/std for each dynamic variable.
    """

    # --- Load datasets ---
    ds_dyn = xr.open_dataset(path_dynamic)
    ds_target = xr.open_dataset(path_target)

    # --- Align time dimension ---
    common_times = np.intersect1d(ds_dyn['time'].values, ds_target['time'].values)
    if len(common_times) == 0:
        raise ValueError("No common timestamps between dynamic and target datasets.")
    ds_dyn = ds_dyn.sel(time=common_times)
    ds_target = ds_target.sel(time=common_times)
    T = len(common_times)

    # --- Extract target shape and geo info ---
    lat = ds_target['latitude'].values
    lon = ds_target['longitude'].values
    with open("data/geo_info.json", "w") as f:
        json.dump({
            'lat_min': float(lat.min()),
            'lat_max': float(lat.max()),
            'lon_min': float(lon.min()),
            'lon_max': float(lon.max())
        }, f, indent=4)

    # PM2.5 target, scaled to g/m³
    target_array = torch.tensor(ds_target['pm2p5_conc'].values) * 1e9  # [T, H, W]
    T, H_hr, W_hr = target_array.shape

    print(f"[{path_target}] pm2p5: min={torch.min(target_array)}, "
          f"max={torch.max(target_array)}, NaNs={torch.isnan(target_array).sum()}, shape={target_array.shape}")

    # --- Load and normalize dynamic variables ---
    dyn_vars = ['pm2p5', 'u10', 'v10', 't2m', 'blh', 'd2m']
    dyn_tensors = []
    dyn_stats = {}

    for var in dyn_vars:
        arr = ds_dyn[var].values  # [T, H_lr, W_lr]
        tensor_lr = torch.tensor(arr).float()
        tensor_hr = upsample_to_hr(tensor_lr, (H_hr, W_hr))  # [T, 1, H_hr, W_hr]

        if var == "pm2p5":  # convert units
            tensor_hr = tensor_hr * 1e9

        print(f"[{path_dynamic}] {var} (upsampled): min={tensor_hr.min().item():.4f}, "
              f"max={tensor_hr.max().item():.4f}, NaNs={(~torch.isfinite(tensor_hr)).sum().item()}, "
              f"shape={tuple(tensor_hr.shape)}")

        # normalization
        mean = tensor_hr.mean().item()
        std = tensor_hr.std().item() + 1e-6
        dyn_stats[var] = {'mean': mean, 'std': std}
        tensor_hr = (tensor_hr - mean) / std

        dyn_tensors.append(tensor_hr)

    X_dyn = torch.cat(dyn_tensors, dim=1)  # concatenate dynamic vars → [T, C_dyn, H_hr, W_hr]

    # --- Process static variables ---
    static_tensors = []
    static1_vars = ['z']  # topography
    static2_vars = ['__xarray_dataarray_variable__']  # placeholder

    for path_static, var_list in zip([path_static1, path_static2], [static1_vars, static2_vars]):
        ds_static = xr.open_dataset(path_static)
        for var in var_list:
            if var not in ds_static:
                print(f"Variable '{var}' not found in {path_static}")
                continue

            arr = ds_static[var].values
            arr = np.where(np.isfinite(arr), arr, np.nan)  # mask invalids

            print(f"[{path_static}] {var}: min={np.nanmin(arr)}, max={np.nanmax(arr)}, "
                  f"NaNs={np.isnan(arr).sum()}, shape={arr.shape}")

            if np.isnan(arr).all():
                print(f"All values are NaN in variable '{var}', skipping.")
                continue

            tensor_static = torch.tensor(arr).float()
            tensor_static = torch.nan_to_num(tensor_static, nan=0.0, posinf=0.0, neginf=0.0)

            # upsample + normalize
            tensor_static_hr = upsample_static_to_hr(tensor_static, (H_hr, W_hr), T)
            mean = tensor_static_hr.mean()
            std = tensor_static_hr.std() + 1e-6
            tensor_static_hr = (tensor_static_hr - mean) / std

            static_tensors.append(tensor_static_hr)

    # Concatenate static + dynamic variables
    if static_tensors:
        X_static = torch.cat(static_tensors, dim=1)
        X = torch.cat([X_dyn, X_static], dim=1)
    else:
        X = X_dyn

    y = target_array.unsqueeze(1).float()  # add channel → [T, 1, H_hr, W_hr]

    # --- Cleanup and normalization ---
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if y_mean is None or y_std is None:
        y_mean = y.mean()
        y_std = y.std() + 1e-6
    y = (y - y_mean) / y_std

    return X, y, y_mean, y_std, dyn_stats


# --- Custom Dataset class ---

class ClimateSuperResDataset(Dataset):
    """
    Custom PyTorch dataset for super-resolution climate data.
    Provides time-indexed access to input-output pairs (X, y).
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]  # number of time steps

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # one sample pair
