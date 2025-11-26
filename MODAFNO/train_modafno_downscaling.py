import os
import torch
import xarray as xr
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import numpy as np
import torch.nn as nn
from torchinfo import summary

from model.spatial_modafno import ModAFNO
from physicsnemo.launch.logging.mlflow import initialize_mlflow
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.distributed.manager import DistributedManager


from model import train
from data.data_factory_static import NetCDFDataset
from model.normalization import compute_stats, save_stats
from model.loss import GeneralPM25Loss
import torch.optim as optim
from torch.utils.data import DataLoader


@hydra.main(
    version_base=None, config_path="config", config_name="spatial_downscaling.yaml"
)
def main(cfg):
    train_downscaling(**OmegaConf.to_container(cfg))


def log_hyperparameters(cfg, client, run):
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_cfg = flatten_dict(cfg)
    for key, value in flat_cfg.items():
        try:
            client.log_param(run.info.run_id, key, value)
        except Exception as e:
            print(f"Could not log param {key}: {e}")


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_downscaling(**cfg):
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"Number of trainable parameters: {count_parameters(model):,}")
    summary(model, input_size=((1, cfg["model"]["in_channels"], cfg["model"]["inp_shape"][0], cfg["model"]["inp_shape"][1]),(1, static_schannels, cfg["model"]["out_shape"][0], cfg["model"]["out_shape"][1])), device=device.type)

    # Dataset setup
    # Static high-res features
        # Reads all .nc files in the directory of the static features and merges them into a single xarray Dataset
    static_files = sorted([os.path.join(cfg["sources"]["dataset"]["static_features_path"], f) for f in os.listdir(cfg["sources"]["dataset"]["static_features_path"]) if f.endswith(".nc")])
    input_static_ds = xr.open_mfdataset(static_files,  combine='by_coords', join='outer')
        
        # Input low-res features
    train_input_ds = xr.open_dataset(cfg["sources"]["dataset"]["train_input"])
        # Target high-res features
    train_target_ds = xr.open_dataset(cfg["sources"]["dataset"]["train_target"])

    # Normalization 
    # Compute and save normalization independent of whether stats exist or not
    norm_cfg = cfg['sources']["dataset"]["normalization"]
    # compute & persist input train‐set stats
    input_stats = compute_stats(train_input_ds,
                          cfg["sources"]["dataset"]["input_variables"],
                          norm_cfg["method"])
    os.makedirs(os.path.dirname(norm_cfg["input_stats_path"]) or ".", exist_ok=True)
    save_stats(input_stats, norm_cfg["input_stats_path"])
    print(f"→ saved normalization stats ({norm_cfg['method']}) to {norm_cfg['input_stats_path']}")

    if norm_cfg['compute_target_stats']:
        target_stats = compute_stats(train_target_ds,
                                      cfg["sources"]["dataset"]["output_variables"],
                                      norm_cfg["method"])
        save_stats(target_stats, norm_cfg["target_stats_path"])


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
    
    valid_input_ds = xr.open_dataset(cfg["sources"]["dataset"]["val_input"])
    valid_target_ds = xr.open_dataset(cfg["sources"]["dataset"]["val_target"])
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

    # Logging
    mlflow_cfg = cfg.get("logging", {}).get("mlflow", {}).copy()

    if mlflow_cfg.pop("use_mlflow", False):
        # Manually resolve any ${now:...} expressions in run_name
        run_name = mlflow_cfg.get("run_name", "")
        if "${now:" in run_name:
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            mlflow_cfg["run_name"] = run_name.replace("${now:%Y-%m-%d_%H-%M-%S}", now_str)

        client, run = initialize_mlflow(**mlflow_cfg)
        LaunchLogger.initialize(use_mlflow=True)

        log_hyperparameters(cfg, client, run)

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
            batch[2]
        ),
        optimizer=optimizer_cls,
        optimizer_params=optimizer_params,
        **cfg["training"],
    )

    # train model
    trainer.fit()


if __name__ == "__main__":
    main()

