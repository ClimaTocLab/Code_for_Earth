import os
import torch
#import wandb
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor

import yaml

torch.set_float32_matmul_precision('high')


def my_main():
    cli = MyLightningCLI(save_config_callback=None)

    # Save YAML configurationGuardar configuración completa en YAML
    with open("full_config.yaml", "w") as f:
        yaml.dump(cli.config, f, sort_keys=False)

class MyLightningCLI(LightningCLI):

    def before_instantiate_classes(self): 
        subcmd = self.config.subcommand

        if hasattr(self.config, subcmd):
            subcfg = getattr(self.config, subcmd)

            data_transform = subcfg['data']['init_args']['transform']
            data_standarized = subcfg['data']['init_args']['standarized']
            model_transform = subcfg['model']['init_args']['transform']
            model_standarized = subcfg['model']['init_args']['standarized']

            if data_transform != model_transform:
                self.config['model']['init_args']['transform'] = data_transform
            if data_standarized != model_standarized:
                self.config['model']['init_args']['standarized'] = data_standarized

            bs = subcfg['data']['init_args']['transform']
            lr = subcfg['model']['init_args']['learning_rate']
            embed_dim = subcfg['model']['init_args']['embed_dim']
            embed_dim_sidechannels = subcfg['model']['init_args']['embed_dim_sidechannels']

            lr_str = str(lr).replace('.', 'p')

            # Construir carpeta output dinámica
            results_dir = f"output_lr_{lr_str}_bs_{bs}_emb_{embed_dim}_emb_sd_{embed_dim_sidechannels}"

            # Sobrescribir el campo en la configuración
            subcfg['default_checkpoint']['dirpath'] = os.path.join(results_dir, 'models')
            subcfg['model']['init_args']['results_dir'] = results_dir
            print(f"Checkpoint dirpath set to: {results_dir}")

        else:
            raise ValueError(f'Warning: subcommand {subcmd} has no config to modify')       
                

    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "default_early_stopping")
        parser.add_lightning_class_args(ModelCheckpoint, "default_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "default_lr_monitor")

        parser.set_defaults({"default_checkpoint.monitor": "val_loss", "default_checkpoint.save_top_k": 1, "default_checkpoint.mode": "min", "default_checkpoint.dirpath": "output/models"})
        parser.set_defaults({"default_early_stopping.monitor": "val_loss", "default_early_stopping.patience": 3, "default_early_stopping.mode": "min"})
        parser.set_defaults({"default_lr_monitor.logging_interval": "epoch"})
        
        # Linking arguments between data and model
        parser.link_arguments("data.init_args.transform", "model.init_args.transform")
        parser.link_arguments("data.init_args.standarized", "model.init_args.standarized")
        parser.link_arguments("data.init_args.dataset_metrics_cerra", "model.init_args.dataset_metrics_cerra", apply_on="instantiate")
        parser.link_arguments("data.init_args.dataset_metrics_era", "model.init_args.dataset_metrics_era", apply_on="instantiate")
        parser.link_arguments("data.init_args.channel_names", "model.init_args.channel_names")
    

if __name__ == '__main__':
    print("Starting PyTorch Lightning CLI...")
    my_main()

