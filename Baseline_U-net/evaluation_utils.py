def metricas(): 
    import torch
    import torch.nn as nn
    import torchmetrics.image as t_metrics
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    # Define las m�tricas y configuraciones
    metric_settings = {
        'MAE': {},
        'MSE': {},
        'Bias_time': {},
        'Bias_space': {},
        'Corr_time': {},
        'Corr_space': {},
        'RMSE_time': {},
        'RMSE_space': {},
        'PSNR': {},
        'SSIM': {}
    }

    # FUNCIONES M�TRICAS

    def compute_loss(pred, target, loss_fn):
        return loss_fn(pred, target).item()

    def compute_bias_time(pred, target):
        bias_map_time = torch.mean(pred - target, dim=0)
        return bias_map_time, torch.nanmean(bias_map_time).item()

    def compute_bias_space(pred, target):
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        bias_map_space = torch.mean(pred_flat - target_flat, dim=1)
        return bias_map_space, torch.nanmean(bias_map_space).item()

    def compute_corr_time(pred, target):
        pred_mean = torch.mean(pred, dim=0)
        target_mean = torch.mean(target, dim=0)
        cov = torch.mean(pred * target, dim=0) - pred_mean * target_mean
        pred_var = torch.mean(pred ** 2, dim=0) - pred_mean ** 2
        target_var = torch.mean(target ** 2, dim=0) - target_mean ** 2
        corr = cov / (torch.sqrt(pred_var) * torch.sqrt(target_var))
        corr[corr > 1] = float('nan')
        corr[corr < -1] = float('nan')
        return corr, torch.nanmean(corr).item()

    def compute_corr_space(pred, target):
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        pred_mean = torch.mean(pred_flat, dim=1)
        target_mean = torch.mean(target_flat, dim=1)
        cov = torch.mean(pred_flat * target_flat, dim=1) - pred_mean * target_mean
        pred_var = torch.mean(pred_flat ** 2, dim=1) - pred_mean ** 2
        target_var = torch.mean(target_flat ** 2, dim=1) - target_mean ** 2
        corr = cov / (torch.sqrt(pred_var) * torch.sqrt(target_var))
        corr[torch.isnan(corr)] = 0
        return corr, torch.nanmean(corr).item()

    def compute_rmse_time(pred, target):
        rmse_map = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
        return rmse_map, torch.nanmean(rmse_map).item()

    def compute_rmse_space(pred, target):
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        rmse_map = torch.sqrt(torch.mean((pred_flat - target_flat) ** 2, dim=1))
        return rmse_map, torch.nanmean(rmse_map).item()

    def compute_skimage_psnr_ssim(pred, target):
        pred_np = pred.cpu().float().numpy()
        target_np = target.cpu().float().numpy()
        psnr_total, ssim_total = 0, 0
        num_images = pred_np.shape[0]

        for i in range(num_images):
            tgt = target_np[i, 0]
            prd = pred_np[i, 0]
            d_range = tgt.max() - tgt.min() + 1e-8  # para evitar divisi�n por cero
            psnr_total += psnr(tgt, prd, data_range=d_range)
            ssim_total += ssim(tgt, prd, data_range=d_range)

        return psnr_total / num_images, ssim_total / num_images

    def calculate_torchmetric(name, pred, target, settings={}, part=5000):
        # Busca la clase de la m�trica en torchmetrics.image
        metric_classes = [m for m in t_metrics.__dict__.keys() if name == m or name in m]
        if not metric_classes:
            raise ValueError(f'Metric {name} not found in torchmetrics.image')

        metric_cls = t_metrics.__dict__[metric_classes[0]]
        metric = metric_cls(**settings)
        metric = metric.to(pred.device)

        total = 0.0
        count = 0
        for i in range(0, pred.size(0), part):
            pred_b = pred[i:i + part]
            target_b = target[i:i + part]
            with torch.no_grad():
                val = metric(pred_b, target_b).item()
                total += val
                count += 1
        return total / count

    # MAIN

    if __name__ == "__main__":
        print("Cargando datos...")

        # Cargar tensores
        data = torch.load('outputs/preds_targets_lowres.pt')
        preds = data['preds']
        targets = data['targets']

        # Corregir shape: eliminar canales extra (de [B,1,1,H,W] a [B,1,H,W])
        while preds.ndim > 4:
            preds = preds.squeeze(1)
            targets = targets.squeeze(1)

        print(f"Shape corregido: {preds.shape}, {targets.shape}")
        assert preds.shape == targets.shape, "Predicci�n y objetivo deben tener el mismo tama�o"

        # Normalizar a [0, 1] para usar con PSNR y SSIM
        def normalize(x):
            x = x.clone()
            x = x - x.min()
            x = x / (x.max() + 1e-8)
            return x

        preds_norm = normalize(preds)
        targets_norm = normalize(targets)

        print("\nCalculando m�tricas:\n")
        results = {}

        for name, settings in metric_settings.items():
            print(f">>> {name}")
            try:
                if name == 'MAE':
                    results[name] = compute_loss(preds, targets, nn.L1Loss())
                elif name == 'MSE':
                    results[name] = compute_loss(preds, targets, nn.MSELoss())
                elif name == 'Bias_time':
                    _, results[name] = compute_bias_time(preds, targets)
                elif name == 'Bias_space':
                    _, results[name] = compute_bias_space(preds, targets)
                elif name == 'Corr_time':
                    _, results[name] = compute_corr_time(preds, targets)
                elif name == 'Corr_space':
                    _, results[name] = compute_corr_space(preds, targets)
                elif name == 'RMSE_time':
                    _, results[name] = compute_rmse_time(preds, targets)
                elif name == 'RMSE_space':
                    _, results[name] = compute_rmse_space(preds, targets)
                elif name == 'PSNR':
                    psnr_avg, _ = compute_skimage_psnr_ssim(preds, targets)
                    results[name] = psnr_avg
                elif name == 'SSIM':
                    _, ssim_avg = compute_skimage_psnr_ssim(preds, targets)
                    results[name] = ssim_avg
                else:
                    # Otras m�tricas de torchmetrics (si decides usarlas)
                    results[name] = calculate_torchmetric(name, preds_norm, targets_norm, settings.get("torchmetric_settings", {}))
            except Exception as e:
                results[name] = f"Error: {e}"

        print("\n---- RESULTADOS ----\n")
        for k, v in results.items():
            print(f"{k}: {v}")



def plott_triple(fecha_idx=100):
    import xarray as xr
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Cargar dataset
    ds = xr.open_dataset("Baseline_U-net/outputs/preds_targets_lowres.nc")

    # Variables a graficar
    variables = ["preds", "targets", "lowres"]
    fecha = str(ds.time.values[fecha_idx])[:10]

    # Determinar vmin y vmax
    try: vmin = min(ds[var].isel(time=fecha_idx).min().item() for var in variables)
    except:
        variables = ["preds", "lowres"]
        vmin = min(ds[var].isel(time=fecha_idx).min().item() for var in variables)
        
    vmax = 80  # fijo como pediste

    # Figura y ejes
    fig, axes = plt.subplots(
        1, 3,
        figsize=(24, 8),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=False,
        gridspec_kw={'width_ratios': [1, 1, 1]}
    )

    for ax, var in zip(axes, variables):
        data = ds[var].isel(time=fecha_idx)

        # Plot con Cartopy
        im = data.plot(
            ax=ax,
            cmap="viridis",
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax
        )

        # A�adir detalles geogr�ficos
        ax.coastlines(resolution='110m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.set_title(f"{var} - {fecha}")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

    # Barra de color
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Valor")

    plt.subplots_adjust(wspace=0.05)
    plt.show()

#metricas()
plott_triple(300)