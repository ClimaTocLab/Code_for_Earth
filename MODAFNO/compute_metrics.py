import os
import numpy as np
import xarray as xr
import torch
import hydra
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim


@hydra.main(
    version_base=None, config_path="config", config_name="spatial_downscaling.yaml"
)
def main(cfg):
    compute_metrics(**OmegaConf.to_container(cfg))

def compute_metrics(**cfg):
    # Load datasets
    pred_ds = xr.open_dataset(os.path.join(cfg["metrics"]["pred_path"], cfg["metrics"]["pred_dataset"]))
    gt_ds = xr.open_dataset(os.path.join(cfg["metrics"]["gt_path"], cfg["metrics"]["gt_dataset"]))

    # Ensure the datasets have the same dimensions
    assert pred_ds.dims == gt_ds.dims, "Prediction and ground truth datasets must have the same dimensions."

    # Compute metrics for each variable
    metrics = {}
    for var in cfg["metrics"]["variables"]:
        if var in pred_ds and var in gt_ds:
            metrics[var] = compute_axis_aware_metrics(pred_ds[var], gt_ds[var])
        else:
            print(f"Variable {var} not found in both datasets.")

    # Save metrics to a file
    output_file = cfg["metrics"]["metrics_file"]
    with open(output_file, 'w') as f:
        for var, metric in metrics.items():
            f.write(f"Metrics for {var}:\n")
            for key, value in metric.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

def compute_axis_aware_metrics(pred: xr.DataArray, gt: xr.DataArray):
    assert pred.shape == gt.shape, "Prediction and target must match in shape."

    # Convert to torch tensors
    pred = torch.from_numpy(pred.values).float()
    gt   = torch.from_numpy(gt.values).float()

    # # Mask NaNs
    # mask = (~np.isnan(pred)) & (~np.isnan(gt))
    # pred = pred.where(mask)
    # gt = gt.where(mask)

    # --- RMSE, MSE & MAE: overall ---
    rmse_total = ((pred - gt) ** 2).mean().sqrt()
    mse_total = ((pred - gt) ** 2).mean()
    mae_total = (pred - gt).abs().mean()

    # --- RMSE/MAE/MSE over time axis (per pixel, then averaged) ---
    rmse_over_time = ((pred - gt) ** 2).mean(dim=0).sqrt().mean()
    mse_over_time = ((pred - gt) ** 2).mean(dim=0).mean()
    mae_over_time = (pred - gt).abs().mean(dim=0).mean()

    # --- RMSE/MAE/MSE over space axis (per timestep, then averaged) ---
    rmse_over_space = ((pred - gt) ** 2).mean(dim=[1, 2]).sqrt().mean()
    mse_over_space = ((pred - gt) ** 2).mean(dim=[1, 2]).mean()
    mae_over_space = (pred - gt).abs().mean(dim=[1, 2]).mean()

    # --- Pearson Correlation over time (for each spatial point, then averaged) ---
    def time_corr(pred, gt):
        x_mean = pred.mean(dim=0)
        y_mean = gt.mean(dim=0)
        cov = ((pred - x_mean) * (gt - y_mean)).mean(dim=0)
        std_x = pred.std(dim=0)
        std_y = gt.std(dim=0)
        return (cov / (std_x * std_y)).mean()  # mean over space

    # --- Pearson Correlation over space (for each time step, then averaged) ---
    def spatial_corr(pred, gt):
      x_mean = pred.mean(dim=[1, 2])
      y_mean = gt.mean(dim=[1, 2])
      cov = ((pred - x_mean[:, None, None]) * (gt - y_mean[:, None, None])).mean(dim=[1, 2])
      std_x = pred.std(dim=[1, 2])
      std_y = gt.std(dim=[1, 2])
      return (cov / (std_x * std_y)).mean()

    corr_time = time_corr(pred, gt)
    corr_space = spatial_corr(pred, gt)

    # --- R² Score: Overall Variance Explained ---
    ss_res = ((gt - pred) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    # --- SSIM: Spatial fidelity (time averaged) ---
    ssim_vals = []
    for t in range(pred.shape[0]):
        x = pred[t].cpu().numpy()
        y = gt[t].cpu().numpy()
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            continue
        score = ssim(x, y, data_range=y.max() - y.min())
        ssim_vals.append(score)
    ssim_mean = np.mean(ssim_vals) if ssim_vals else np.nan

    # --- PSNR: Peak Signal-to-Noise Ratio ---
    # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    # Higher values indicate better quality
    if mse_total.item() > 0:
        max_pixel_value = max(gt.max().item(), pred.max().item())
        psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse_total.item())
    else:
        psnr = float('inf')  # Perfect match case


    # --- Physical Constraint: Total Mass Error (spatial sum) ---
    total_mass_pred = pred.sum(dim=[1, 2])
    total_mass_gt = gt.sum(dim=[1, 2])
    mass_error = (total_mass_pred - total_mass_gt).abs().mean()

    return {
        "RMSE_total": rmse_total.item(),
        "MAE_total": mae_total.item(),
        "MSE_total": mse_total.item(),
        "RMSE_over_time": rmse_over_time.item(),
        "MAE_over_time": mae_over_time.item(),
        "MSE_over_time": mse_over_time.item(),
        "RMSE_over_space": rmse_over_space.item(),
        "MAE_over_space": mae_over_space.item(),
        "MSE_over_space": mse_over_space.item(),
        "Corr_over_time": corr_time.item(),
        "Corr_over_space": corr_space.item(),
        "R_squared": r_squared.item(),
        "SSIM_mean": ssim_mean,
        "PSNR": psnr,
        "Mass_Conservation_Error": mass_error.item()
    }


if __name__ == "__main__":
    main()