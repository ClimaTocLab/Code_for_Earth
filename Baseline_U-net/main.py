import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from train import train
from model import UNetSuperRes
from test import test_model, to_netcdf
from dataset import prepare_inputs_and_target, ClimateSuperResDataset
import xarray as xr

# --- Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Print environment info
print(torch.__version__)
print("CUDA IS AVAILABLE? " + str(torch.cuda.is_available()))

# --- Execution flags ---
TRAIN = 1   # Set to 1 to train
TEST = 0    # Set to 1 to evaluate


if TRAIN or TEST:
    # --- Data paths ---
    path_dynamic_train = "/media/server/code4earth/Final_Dataset/daily_data_extended/train_input.nc"
    path_target_train = '/media/server/code4earth/Final_Dataset/daily_data_extended/train_output.nc'

    path_dynamic_val = "/media/server/code4earth/Final_Dataset/daily_data_extended/valid_input.nc"
    path_target_val = "/media/server/code4earth/Final_Dataset/daily_data_extended/valid_output.nc"

    path_dynamic_test = "/media/server/code4earth/Final_Dataset/daily_data_extended/test_input_2025.nc"
    path_target_test = "/media/server/code4earth/Final_Dataset/daily_data_extended/test_output_2025.nc"

    # Static variables (topography, population, etc.)
    path_static1 = '/media/server/code4earth/code4earth_datasets/ETOPO_2022_v1_60s_N90W180_surface_regional_regridded.nc'
    path_static2 = '/media/server/code4earth/code4earth_datasets/GHS_population_spatial_resol_0.1_regional_regridded.nc'

    # --- Data preparation ---
    # Training set (also computes normalization statistics)
    X_train, y_train, y_mean, y_std, dyn_stats = prepare_inputs_and_target(
        path_dynamic_train, path_static1, path_static2, path_target_train
    )

    # Validation set (normalized using training stats)
    X_val, y_val, _, _, _ = prepare_inputs_and_target(
        path_dynamic_val, path_static1, path_static2, path_target_val,
        y_mean=y_mean, y_std=y_std
    )

    # Test set (normalized using training stats)
    X_test, y_test, _, _, _ = prepare_inputs_and_target(
        path_dynamic_test, path_static1, path_static2, path_target_test,
        y_mean=y_mean, y_std=y_std
    )

    # --- Optional: debug mode with small temporal subset ---
    fast_debug = True
    if fast_debug:
        cut = 10
        X_train = X_train[:cut]  # [T', C, H, W]
        y_train = y_train[:cut]

        cut = 2
        X_val = X_val[:cut]
        y_val = y_val[:cut]

        X_test = X_test[:cut]
        y_test = y_test[:cut]

    # Print dataset shapes
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shape:   X={X_val.shape}, y={y_val.shape}")

    # --- Build datasets & loaders ---
    train_dataset = ClimateSuperResDataset(X_train, y_train)
    val_dataset = ClimateSuperResDataset(X_val, y_val)
    test_dataset = ClimateSuperResDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # --- Model setup ---
    model = UNetSuperRes(n_channels_in=X_train.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Training ---
if TRAIN:
    if fast_debug: epochs=2 
    else: epochs=200
    train(model, train_loader, val_loader, epochs=epochs, device=device)


# --- Testing / Evaluation ---
if TEST:
    path = 'Baseline_U-net/model_20250801_1701.pth'

    # Load trained model weights
    # model.load_state_dict(torch.load(path))  # GPU default
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    # Evaluate on test set
    test_model(model, test_loader, device, y_mean, y_std)

    # Export results to NetCDF
    to_netcdf(path_target_test)


