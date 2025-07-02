import os
import torch
from rich.progress import Progress


import torch
import os
from rich.progress import Progress
import torch.nn.functional as F

def test_model(model, test_loader, device, y_mean, y_std):
    model.to(device)
    model.eval()
    preds_all = []
    targets_all = []
    lowres_all = []

    os.makedirs('outputs', exist_ok=True)

    with torch.no_grad(), Progress() as progress:
        task = progress.add_task("[cyan]Evaluando modelo...", total=len(test_loader))

        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_X)

            # Desnormalizar predicciones y targets
            preds_denorm = preds * y_std + y_mean
            targets_denorm = batch_y * y_std + y_mean

            # Interpolar PM2.5 (asumimos canal 0 es PM2.5) a alta resoluciï¿½n
            lowres_interp = F.interpolate(batch_X[:, 0:1], size=batch_y.shape[-2:], mode='bilinear', align_corners=False)
            lowres_interp_denorm = lowres_interp * y_std + y_mean

            preds_all.append(preds_denorm.cpu())
            targets_all.append(targets_denorm.cpu())
            lowres_all.append(lowres_interp_denorm.cpu())

            progress.update(task, advance=1)

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    lowres_all = torch.cat(lowres_all, dim=0)

    torch.save({'preds': preds_all, 'targets': targets_all, 'lowres': lowres_all}, 'outputs/preds_targets.pt')

    print(f"? Predicciones desnormalizadas: shape={preds_all.shape}, min={preds_all.min():.3f}, max={preds_all.max():.3f}")
    print(f"? Targets desnormalizados: shape={targets_all.shape}, min={targets_all.min():.3f}, max={targets_all.max():.3f}")
    print(f"? Low-res interpolado: shape={lowres_all.shape}, min={lowres_all.min():.3f}, max={lowres_all.max():.3f}")
    print("? Predicciones, targets y low-res guardados en 'outputs/preds_targets.pt'")


 


def test_model_old(model, test_loader, device, y_mean, y_std):
    model.to(device)
    model.eval()
    preds_all = []
    targets_all = []

    os.makedirs('outputs', exist_ok=True)  # Asegura que exista el directorio

    with torch.no_grad(), Progress() as progress:
        task = progress.add_task("[cyan]Evaluando modelo...", total=len(test_loader))

        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_X)

            # Desnormalizar predicciones y targets
            preds_denorm = preds * y_std + y_mean
            targets_denorm = batch_y * y_std + y_mean

            preds_all.append(preds_denorm.cpu())
            targets_all.append(targets_denorm.cpu())

            progress.update(task, advance=1)

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    torch.save({'preds': preds_all, 'targets': targets_all}, 'outputs/preds_targets.pt')

    print(f"? Predicciones desnormalizadas: shape={preds_all.shape}, min={preds_all.min():.3f}, max={preds_all.max():.3f}")
    print(f"? Targets desnormalizados: shape={targets_all.shape}, min={targets_all.min():.3f}, max={targets_all.max():.3f}")
    print("? Predicciones y targets guardados como 'outputs/preds_targets.pt'")
