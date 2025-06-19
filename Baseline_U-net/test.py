import torch

def test_model(model, test_loader, device, y_mean, y_std):
    model.to(device)
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_X)

            # Desnormalizar predicciones y targets
            preds_denorm = preds * y_std + y_mean
            targets_denorm = batch_y * y_std + y_mean

            preds_all.append(preds_denorm.cpu())
            targets_all.append(targets_denorm.cpu())

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    torch.save({'preds': preds_all, 'targets': targets_all}, 'output/preds_targets.pt')

    print(f"Predicciones desnormalizadas: shape={preds_all.shape}, min={preds_all.min():.3f}, max={preds_all.max():.3f}")
    print(f"Targets desnormalizados: shape={targets_all.shape}, min={targets_all.min():.3f}, max={targets_all.max():.3f}")
    print("Predicciones y targets guardados como 'preds_targets.pt'")

 