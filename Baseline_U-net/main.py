import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from train import train
from model import UNetSuperRes
from test import test_model
from dataset import prepare_inputs_and_target, ClimateSuperResDataset

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
print(torch.__version__)
print("CUDA IS AVAIABLE? "+str(torch.cuda.is_available()))

# --- Ejecuciï¿½n ---
TRAIN=1
TEST=1
PLOT=0



if TRAIN or TEST:
    # Rutas a los datos (cï¿½mbialas por las reales)
    path_dynamic = 'data/CAMS_global_forecast_2015-2025.nc'
    path_static1 = 'data/ETOPO_2022_v1_60s_N90W180_surface_regional_regridded.nc'
    path_static2 = 'data/GHS_population_spatial_resol_0.1_regional_regridded.nc'
    path_target = 'data/all_cams_ens_fc_pm2p5_level0_daily_2019_2025.nc'

    # Preparar inputs y target
    X, y, y_mean, y_std, dyn_stats = prepare_inputs_and_target(path_dynamic, path_static1, path_static2, path_target)

    # --- DEBUG: usar solo un subconjunto temporal pequeï¿½o ---
    X = X[:10]  # Usa solo los primeros n pasos temporales
    y = y[:10]

    print(f"Input shape: {X.shape}, Target shape: {y.shape}")

    # Split data
    T = X.shape[0]
    train_end = int(0.8 * T)
    val_end = int(0.9 * T)

    # Divide los datos
    train_dataset = ClimateSuperResDataset(X[:train_end], y[:train_end])
    val_dataset = ClimateSuperResDataset(X[train_end:val_end], y[train_end:val_end])
    test_dataset = ClimateSuperResDataset(X[val_end:], y[val_end:])

    # Crea los dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Modelo
    model = UNetSuperRes(n_channels_in=X.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if TRAIN:
    # Entrenamiento
    train(model, train_loader, val_loader, epochs=100, device=device)

if TEST:
    # Inferencia
    model.load_state_dict(torch.load('outputs/model_20250623_1303.pth'))
    test_model(model, test_loader, device, y_mean, y_std)


    


