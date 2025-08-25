import xarray as xr
import torch
import os
from rich.progress import Progress
import torch.nn.functional as F


def test_model(model, test_loader, device, y_mean, y_std):
    """
    Evaluates the trained model on the test set, denormalizes outputs,
    interpolates low-res inputs, and saves results in a .pt file.
    """
    model.to(device)
    model.eval()
    preds_all = []
    targets_all = []
    lowres_all = []

    os.makedirs('outputs', exist_ok=True)

    # Progress bar for evaluation
    with torch.no_grad(), Progress() as progress:
        task = progress.add_task("[cyan]Evaluating model...", total=len(test_loader))

        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            preds = model(batch_X)

            # --- Denormalization ---
            preds_denorm = preds * y_std + y_mean
            targets_denorm = batch_y * y_std + y_mean

            # --- Interpolation of low-res input (assume channel 0 = PM2.5) ---
            lowres_interp = F.interpolate(
                batch_X[:, 0:1], size=batch_y.shape[-2:], mode='bilinear', align_corners=False
            )
            lowres_interp_denorm = lowres_interp * y_std + y_mean

            # Collect results
            preds_all.append(preds_denorm.cpu())
            targets_all.append(targets_denorm.cpu())
            lowres_all.append(lowres_interp_denorm.cpu())

            progress.update(task, advance=1)

    # Concatenate all results into tensors
    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    lowres_all = torch.cat(lowres_all, dim=0)

    # Save to file
    torch.save(
        {'preds': preds_all, 'targets': targets_all, 'lowres': lowres_all},
        'outputs/preds_targets_lowres.pt'
    )

    # Debug info
    print(f"✔ Predictions denormalized: shape={preds_all.shape}, min={preds_all.min():.3f}, max={preds_all.max():.3f}")
    print(f"✔ Targets denormalized: shape={targets_all.shape}, min={targets_all.min():.3f}, max={targets_all.max():.3f}")
    print(f"✔ Low-res interpolated: shape={lowres_all.shape}, min={lowres_all.min():.3f}, max={lowres_all.max():.3f}")
    print("✔ Predictions, targets and low-res saved in 'outputs/preds_targets_lowres.pt'")


def to_netcdf(nc_source):
    """
    Converts saved .pt predictions into a NetCDF (.nc) file with proper
    time, latitude, and longitude coordinates from the original dataset.
    """
    # 1. Load saved .pt file
    data = torch.load("outputs/preds_targets_lowres.pt")
    preds = data["preds"].numpy()     # (time, 1, H, W)
    targets = data["targets"].numpy()
    lowres = data["lowres"].numpy()

    # 2. Load temporal and spatial coordinates from original NetCDF
    ds_input = xr.open_dataset(nc_source)
    fechas = ds_input.time.values
    latitude = ds_input.latitude.values
    longitude = ds_input.longitude.values

    # 3. Build xarray.Dataset
    ds_resultado = xr.Dataset(
        {
            "preds": xr.DataArray(
                preds.squeeze(1),  # remove channel dim → (time, H, W)
                dims=["time", "latitude", "longitude"],
                coords={"time": fechas, "latitude": latitude, "longitude": longitude}
            ),
            "targets": xr.DataArray(
                targets.squeeze(1),
                dims=["time", "latitude", "longitude"],
                coords={"time": fechas, "latitude": latitude, "longitude": longitude}
            ),
            "lowres": xr.DataArray(
                lowres.squeeze(1),
                dims=["time", "latitude", "longitude"],
                coords={"time": fechas, "latitude": latitude, "longitude": longitude}
            ),
        }
    )

    # 4. Save as NetCDF
    ds_resultado.to_netcdf("preds_targets_lowres.nc")
    print("✔ Saved as preds_targets_lowres.nc")
