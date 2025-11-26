import os

import torch
import xarray as xr
import numpy as np
import hydra
from omegaconf import OmegaConf

from model.spatial_modafno import ModAFNO
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.distributed.manager import DistributedManager


from model import train
from data.data_factory_static import NetCDFDataset
from model.loss import GeneralPM25Loss
import torch.optim as optim
from torch.utils.data import DataLoader


@hydra.main(
    version_base=None, config_path="config", config_name="spatial_downscaling.yaml"
)
def main(cfg):
    test_diagnostic(**OmegaConf.to_container(cfg))


def test_diagnostic(**cfg):
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalization
    norm_cfg = cfg['sources']["dataset"]["normalization"]
    if norm_cfg["method"] != "none" and os.path.exists(norm_cfg['input_stats_path']):
        input_stats = np.load(norm_cfg['input_stats_path'])
        if norm_cfg["compute_target_stats"] and os.path.exists(norm_cfg['target_stats_path']):
            target_stats = np.load(norm_cfg['target_stats_path'])
        else:   
            target_stats = None
    else:
        input_stats = target_stats = None
        print("No normalization statistics found, using 'none' normalization.")

    static_schannels = len(cfg["sources"]["dataset"]["static_variables"])
    # Model setup
    model = ModAFNO(
        inp_shape=cfg["model"]["inp_shape"],
        out_shape=cfg["model"]["out_shape"],
        static_channels=static_schannels,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        embed_model=cfg["model"]["embed_model"],
        patch_size=cfg["model"]["patch_size"],
        embed_dim=cfg["model"]["embed_dim"],
        mod_dim=cfg["model"]["embed_model"]["dim"],
        depth=cfg["model"]["depth"],
        num_blocks=cfg["model"]["num_blocks"],
        modulate_mlp=cfg["model"]["modulate_mlp"],
        modulate_filter=cfg["model"]["modulate_filter"],
    ).to(device)

    # Dataset setup
    # Static high-res features
        # Reads all .nc files in the directory of the static features and merges them into a single xarray Dataset
    static_files = sorted([os.path.join(cfg["sources"]["dataset"]["static_features_path"], f) for f in os.listdir(cfg["sources"]["dataset"]["static_features_path"]) if f.endswith(".nc")])
    input_static_ds = xr.open_mfdataset(static_files,  combine='by_coords', join='outer')
        
    train_input_ds = xr.open_dataset(cfg["sources"]["dataset"]["train_input"])
    train_target_ds = xr.open_dataset(cfg["sources"]["dataset"]["train_target"])
    train_dataset = NetCDFDataset(
        input_static_ds,
        train_input_ds,
        train_target_ds,
        cfg["sources"]["dataset"]['high_res_low_res_ratio'],
        static_variables=cfg["sources"]["dataset"]['static_variables'],
        input_variables=cfg["sources"]["dataset"]['input_variables'],
        output_variables=cfg["sources"]["dataset"]['output_variables'],
        input_stats=input_stats,
        target_stats=target_stats,
        norm_method=norm_cfg["method"],
    )
    
    valid_input_ds = xr.open_dataset(cfg["inference"]["inference_input"])
    valid_target_ds = xr.open_dataset(cfg["inference"]["inference_target"])
    valid_dataset = NetCDFDataset(
        input_static_ds,
        valid_input_ds,
        valid_target_ds,
        cfg["sources"]["dataset"]['high_res_low_res_ratio'],
        static_variables=cfg["sources"]["dataset"]['static_variables'],
        input_variables=cfg["sources"]["dataset"]['input_variables'],
        output_variables=cfg["sources"]["dataset"]['output_variables'],
        input_stats=input_stats,
        target_stats=target_stats,
        norm_method=norm_cfg["method"],
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['training_args']['batch_size'], shuffle=True, num_workers=cfg['training_args']['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['training_args']['batch_size'], shuffle=False, num_workers=cfg['training_args']['num_workers'])

    
    # Distributed setup
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    print(f"Using device: {dist_manager.device}")

    # create callback for tracking error
    if norm_cfg["method"] == "z_score" or norm_cfg["method"] == "z_score_per_pixel":
        mean = target_stats['mean'] if target_stats is not None else None
        std = target_stats['std'] if target_stats is not None else None
        rmse_callback = RMSECallback(device=dist_manager.device, mean=mean, std=std)
    elif norm_cfg["method"] == "minmax" or norm_cfg["method"] == "minmax_per_pixel":
        mean = target_stats['min'] if target_stats is not None else None
        std = (target_stats['max'] - target_stats['min']) if target_stats is not None else None
        rmse_callback = RMSECallback(device=dist_manager.device, mean=mean, std=std)
    else:
        mean = std = None
        rmse_callback = RMSECallback(device=dist_manager.device, mean=mean, std=std)

    # Loss
    loss_func = GeneralPM25Loss(
        loss_type=cfg["training_args"]["loss"]["loss_type"],
        log_weight=cfg["training_args"]["loss"]["log_weight"],
        eps=cfg["training_args"]["loss"]["eps"]
    ).to(device)
 
    print('loss:', loss_func)

    # Optimizer
    optimizer_cls = getattr(optim, cfg["training_args"]["optimizer"]["optimizer_type"])
    optimizer_params = cfg["training_args"]["optimizer"].get("optimizer_params", {})

    # setup training loop
    trainer = train.Trainer(
        model,
        dist_manager=dist_manager,
        loss=loss_func,
        train_datapipe=train_loader,
        valid_datapipe=valid_loader,
        input_output_from_batch_data=lambda batch: (
            (batch[0][0].to(device), batch[0][1].to(device)),
            batch[1].to(device),
            batch[2]  # coords, not moved to device
        ),
        optimizer=optimizer_cls,
        optimizer_params=optimizer_params,
        validation_callbacks=[rmse_callback],
        **cfg["evaluation"],
    )

    # Load best model
    trainer.load_model_for_inference()
    # evaluate model
    os.makedirs(cfg['inference']['inference_results'], exist_ok=True)

    trainer.validate_on_epoch(
         perform_inference=True,
         save_dir= cfg['inference']['inference_maps_dir'], 
         prediction_var_name=cfg["sources"]["dataset"]['output_variables'],
         mean=mean,
         std=std,
    )

    # save results
    rmse = rmse_callback.value().cpu().numpy()
    np.save(f"{cfg['inference']['inference_results']}/rmse.npy", rmse)  # TODO: should be configurable


class RMSECallback:
    """Callable that keeps track of RMS error.
    Can be used in `Trainer.validation_callbacks`.
    """

    def __init__(self, device, mean=None, std=None):
        self.mse = None
        self.n_samples = 0
        self.mean = None if mean is None else torch.from_numpy(mean).to(device=device)
        self.std = None if std is None else torch.from_numpy(std).to(device=device)

    def __call__(self, outvar_true, outvar_pred, **kwargs):
        # reverse normalization
        if self.mean is not None:
            if self.mean.ndim == 1:  # (C,)
                mean = self.mean.view(1, -1, 1, 1)
                std = self.std.view(1, -1, 1, 1)
            elif self.mean.ndim == 3:  # (C, H, W)
                mean = self.mean.unsqueeze(0)
                std = self.std.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected shape for mean: {self.mean.shape}")
            outvar_true = outvar_true * std + mean
            outvar_pred = outvar_pred * std + mean

        # compute squared difference
        sqr_diff = torch.square(outvar_true - outvar_pred)
        batch_size = sqr_diff.shape[0]
        avg_axes = tuple(range(sqr_diff.ndim - 2))
        sqr_diff = torch.mean(sqr_diff, axis=avg_axes)

        # accumulate MSE
        if self.mse is None:
            self.mse = sqr_diff
        else:
            old_weight = self.n_samples / (self.n_samples + batch_size)
            new_weight = 1 - old_weight
            self.mse = old_weight * self.mse + new_weight * sqr_diff
        self.n_samples += batch_size

    def value(self):
        return torch.sqrt(self.mse)


if __name__ == "__main__":
    main()

