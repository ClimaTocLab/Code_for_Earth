import lightning as L
from lightning.pytorch.callbacks import Callback
import pandas as pd
import numpy as np
import plotly.express as px
import wandb
import os
import torch
import h5py as h5
from tqdm import tqdm

import matplotlib.pyplot as plt


class PlottingCallback(Callback):
    def __init__(self, spatial_split=None):
        self.global_bias = None
        self.batch_size = None

        self.count_map = {}
        self.channel_names = None

        self.spatial_split = spatial_split
        self.patch_size = None
        self.is_era_original = False

        self.is_sidechannel = None

        self.channel_units_map = {'si10': 'm/s', 'u10': 'm/s', 'v10': 'm/s', 'wdir10': 'degree', 't2m': 'K', 'sp': 'Pa', 'msl': 'Pa', 't850': 'K', 'u1000': 'm/s', 'v1000': 'm/s', 'z1000': '$m^2$/$s^2$', 'u850': 'm/s', 'v850': 'm/s', 'z850': '$m^2$/$s^2$', 'u500': 'm/s', 'v500': 'm/s', 'z500': '$m^2$/$s^2$', 't500': 'K', 'z50': '$m^2$/$s^2$', 'r500': '%', 'r850': '%', 'tcwv': 'kg/$m^2$'}




    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:

        if stage == 'test':
            self.number_of_crops_per_sample = trainer.datamodule.dataset_test.number_of_crops_per_sample
            self.number_of_crops_per_sample_x = trainer.datamodule.dataset_test.number_of_crops_per_sample_x
            self.number_of_crops_per_sample_y = trainer.datamodule.dataset_test.number_of_crops_per_sample_y
            self.out_channels = trainer.datamodule.dataset_test.out_channels

            self.variable_mean = trainer.datamodule.dataset_metrics['variable_mean'].reshape(1, -1, 1, 1)[:, self.out_channels]
            self.variable_std = trainer.datamodule.dataset_metrics['variable_std'].reshape(1, -1, 1, 1)[:, self.out_channels]


            assert trainer.datamodule.return_offset, "The datamodule must return the offset for the plotting callback to work"
            if trainer.datamodule.return_era_original:
                print("Datamodule returns the original ERA5 data: Plotting both predicted and ERA5 data")
                self.is_era_original = True

            self.dataset_length = len(trainer.datamodule.dataset_test)
            self.set_batch_size = trainer.datamodule.hparams.batch_size

            self.is_sidechannel = trainer.datamodule.hparams.constant_channels

            if self.set_batch_size != trainer.datamodule.dataset_test.number_of_crops_per_sample:
                print("Batch size is not equal to the number of crops per sample. This may lead to incorrect results for the error over time")

            #self.spatial_split = trainer.datamodule.dataset_test.spatial_split

            self.channel_names = np.array(trainer.datamodule.channel_names)[self.out_channels]

            self.num_bins = 100
            self.histogram_acumulator = torch.zeros((len(self.channel_names), self.num_bins), dtype=torch.float64)
            self.histogram_acumulator_cerra = torch.zeros_like(self.histogram_acumulator)
            self.histogram_acumulator_era = torch.zeros_like(self.histogram_acumulator)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        #Global Bias/Error
        self.count_map = {}

        #Histograms
        self.histogram_acumulator = torch.zeros_like(self.histogram_acumulator)
        self.histogram_acumulator_cerra = torch.zeros_like(self.histogram_acumulator)
        self.histogram_acumulator_era = torch.zeros_like(self.histogram_acumulator)
        self.histogram_count = 0

        self.error_over_time = torch.zeros((self.dataset_length//self.set_batch_size, len(self.out_channels)))

    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        offsets = batch[2]
        size = outputs.shape[2]
        batch_size = outputs.shape[0]


        if self.batch_size is None:
            self.batch_size = batch_size
            self.global_bias = torch.zeros((outputs.shape[1], self.number_of_crops_per_sample_x*size, self.number_of_crops_per_sample_y*size))
            self.global_squared_error = torch.zeros_like(self.global_bias)
            self.patch_size = size

            if self.is_era_original:
                self.global_bias_era = torch.zeros_like(self.global_bias)
                self.global_squared_error_era = torch.zeros_like(self.global_squared_error)

        if self.is_sidechannel:
            target = batch[1][0].cpu() * self.variable_std + self.variable_mean
        else:
            target = batch[1].cpu() * self.variable_std + self.variable_mean
        
        output = outputs.cpu() #already de-standardized in the model
        bias = output - target
        squared_error = (bias ** 2)


        if self.is_era_original:
            era_outputs = batch[3][:,:outputs.shape[1]].cpu() * self.variable_std + self.variable_mean
            era_bias = era_outputs - target
            era_squared_error = (era_bias ** 2)

        #keep track of the error over time
        self.error_over_time[self.histogram_count] = torch.mean(squared_error, dim=(0, 2, 3))

        ###Compute the histograms
        self.histogram_count += 1

        for channel in range(len(self.channel_names)):
            channel_mean = self.variable_mean[0, channel, 0, 0].item()
            channel_std = self.variable_std[0, channel, 0, 0].item()

            hist_min = channel_mean - 3 * channel_std
            hist_max = channel_mean + 3 * channel_std
            self.histogram_acumulator[channel] = (self.histogram_acumulator[channel] * (self.histogram_count - 1) + torch.histc(output[:, channel], bins=self.num_bins, min=hist_min, max=hist_max)) / self.histogram_count
            self.histogram_acumulator_cerra[channel] = (self.histogram_acumulator_cerra[channel]  * (self.histogram_count - 1) + torch.histc(target[:, channel], bins=self.num_bins, min=hist_min, max=hist_max)) / self.histogram_count

            if self.is_era_original:
                self.histogram_acumulator_era[channel] = (self.histogram_acumulator_era[channel] * (self.histogram_count - 1) + torch.histc(era_outputs[:, channel], bins=self.num_bins, min=hist_min, max=hist_max)) / self.histogram_count


        for i in range(batch_size):
            map_key = f"{offsets[0][i].item()}-{offsets[1][i].item()}"

            if map_key not in self.count_map:
                self.count_map[map_key] = 0

            self.count_map[map_key] += 1
            count = self.count_map[map_key]
            self.global_bias[:, offsets[0][i]:offsets[0][i]+size, offsets[1][i]:offsets[1][i] + size] = (self.global_bias[:, offsets[0][i]:offsets[0][i]+size, offsets[1][i]:offsets[1][i]+ size] * (count-1) + bias[i]) / count
            self.global_squared_error[:, offsets[0][i]:offsets[0][i]+size, offsets[1][i]:offsets[1][i] + size] = (self.global_squared_error[:, offsets[0][i]:offsets[0][i]+size, offsets[1][i]:offsets[1][i]+ size] * (count-1) + squared_error[i]) / count

            if self.is_era_original:
                self.global_bias_era[:, offsets[0][i]:offsets[0][i]+size, offsets[1][i]:offsets[1][i]+ size] = (self.global_bias_era[:, offsets[0][i]:offsets[0][i]+size, offsets[1][i]:offsets[1][i]+ size] * (count-1) + era_bias[i]) / count
                self.global_squared_error_era[:, offsets[0][i]:offsets[0][i]+size, offsets[1][i]:offsets[1][i] + size] = (self.global_squared_error_era[:, offsets[0][i]:offsets[0][i]+size, offsets[1][i]:offsets[1][i]+ size] * (count-1) + era_squared_error[i]) / count


    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:

        self.plot_histograms(trainer, pl_module, self.histogram_acumulator, self.histogram_acumulator_cerra, self.histogram_acumulator_era)

        self.plot_error_time(trainer, pl_module, self.error_over_time)

        self.plot_bias(trainer, pl_module, self.global_bias)
        self.plot_bias(trainer, pl_module, self.global_squared_error, figure_name='global_test_mse', z_min=0)
        self.plot_with_era(trainer, pl_module, self.global_bias, self.global_bias_era, figure_name='global_test_bias_era')
        self.plot_with_era(trainer, pl_module, self.global_squared_error, self.global_squared_error_era, figure_name='global_test_mse_era', z_min=0)
        
    def plot_histograms(self, trainer: L.Trainer, pl_module: L.LightningModule, hist_data: torch.Tensor, hist_data_cerra: torch.Tensor, hist_data_era: torch.Tensor = None, figure_name: str = 'histograms'):
        print("Plotting histograms")


        columns = int(min(hist_data.shape[0], 5))
        rows = int(np.ceil(hist_data.shape[0] / columns))
        fig, axs = plt.subplots(columns, rows, figsize=(20, 20))

        for ax, channel in zip(axs.flat, range(hist_data.shape[0])):
            channel_name = self.channel_names[channel]
            channel_mean = self.variable_mean[0, channel, 0, 0].item()
            channel_std = self.variable_std[0, channel, 0, 0].item()
            hist_min = channel_mean - 3 * channel_std
            hist_max = channel_mean + 3 * channel_std

            ax.plot(np.linspace(hist_min, hist_max, self.num_bins), hist_data[channel], color='blue') #title=f'{channel_name} - ML', color='blue')
            ax.plot(np.linspace(hist_min, hist_max, self.num_bins), hist_data_cerra[channel], color='red') # title=f'{channel_name} CERRA', color='red')
            if self.is_era_original:
                ax.plot(np.linspace(hist_min, hist_max, self.num_bins), hist_data_era[channel], color='green')
            
            ax.set_title(f'Channel: {channel_name}')
            ax.set_xlabel(f'[{self.channel_units_map[channel_name]}]')
            ax.set_ylabel('Count')
            

        fig.tight_layout()
        if self.is_era_original:
            fig.legend(['ML', 'CERRA', 'ERA5 (bicubic)'], loc='upper right')
        else:
            fig.legend(['ML', 'CERRA'], loc='upper right')

        #plt.savefig(f'{figure_name}.png')
        # wandb.log({figure_name: fig})
        wandb.log({figure_name: wandb.Image(fig)})
        

    
    def plot_error_time(self, trainer: L.Trainer, pl_module: L.LightningModule, data: torch.tensor, figure_name: str = 'error_over_time'):
        print("Plotting Error over time")
        
        columns = int(min(data.shape[1], 5))
        rows = int(np.ceil(data.shape[1] / columns))
        fig, axs = plt.subplots(columns, rows, figsize=(20, 20))

        x = np.arange(0, data.shape[0]//8, 1/8)


        for ax, channel in zip(axs.flat, range(data.shape[1])):
            channel_name = self.channel_names[channel]
            
            ax.plot(x, data[:, channel], color='blue') #title=f'{channel_name} - ML', color='blue')
            ax.set_title(f'Channel: {channel_name}')
            

        fig.tight_layout()
       

        #plt.savefig(f'{figure_name}.png')
        
        # wandb.log({figure_name: fig})
        wandb.log({figure_name: wandb.Image(fig)})

        


    def plot_bias(self, trainer: L.Trainer, pl_module: L.LightningModule, global_data: torch.Tensor = None, figure_name: str = 'global_test_bias', z_min=None):
        columns = int(min(global_data.shape[0], 5))
        rows = int(np.ceil(global_data.shape[0] / columns))

        fig, axs = plt.subplots(columns, rows, figsize=(20, 20))

        if self.spatial_split is not None:
            print(f"Plotting: Test split uses {len(self.spatial_split)}/{len(self.count_map)} of the data")
            split_area = []

            for index in self.spatial_split:
                y_start = index // self.number_of_crops_per_sample_x * self.patch_size
                x_start = index % self.number_of_crops_per_sample_x * self.patch_size
                split_area.append((y_start, x_start))



        for ax, i in zip(axs.flat, range(global_data.shape[0])):
            percentile_5 = global_data[i].quantile(0.05).item()
            percentile_95 = global_data[i].quantile(0.95).item()

            if self.spatial_split is not None:
                combined_tensor = torch.zeros(len(self.spatial_split),self.patch_size,self.patch_size)
                for index, pos in enumerate(split_area):
                    combined_tensor[index] = global_data[i, pos[0]:pos[0]+self.patch_size, pos[1]:pos[1]+self.patch_size]

                
                percentile_5_test = combined_tensor.quantile(0.05).item()
                percentile_95_test = combined_tensor.quantile(0.95).item()
                
                title = f'{self.channel_names[i]}: ({percentile_5:.2f}, {percentile_95:.2f}) [{percentile_5_test:.2f}, {percentile_95_test:.2f}]'

            else: 
                title = f'{self.channel_names[i]} ({percentile_5:.2f}, {percentile_95:.2f})'     

            ax.set_title(title)
            if z_min is not None:
                im = ax.imshow(global_data[i], cmap='inferno', vmin=z_min, vmax=percentile_95)
            else:
                im = ax.imshow(global_data[i], cmap='RdBu_r', vmin=percentile_5, vmax=percentile_95)
            
            fig.colorbar(im, ax=ax)

        # plt.savefig(f'{figure_name}.png')
        wandb.log({figure_name: fig})

    def plot_with_era(self, trainer: L.Trainer, pl_module: L.LightningModule,  global_data: torch.Tensor = None, global_data_era: torch.tensor = None, figure_name: str = 'global_test_bias_era', z_min=None):

        columns = 2 #int(min(global_data.shape[0], 5)) * 2
        rows = int(np.ceil(global_data.shape[0] / columns)) * 2

        fig, axs = plt.subplots(rows, columns, figsize=(20, 100))

        fig.set_tight_layout(True)

        if self.spatial_split is not None:
            #print(f"Test split uses only {len(self.spatial_split)}/{len(self.count_map)} of the data")
            split_area = []

            for index in self.spatial_split:
                y_start = index // self.number_of_crops_per_sample_x * self.patch_size
                x_start = index % self.number_of_crops_per_sample_x * self.patch_size
                split_area.append((y_start, x_start))


        left = True
        indices = [i for i in range(global_data.shape[0]) for _ in range(2)]
        for ax, i in zip(axs.flat, indices):
            
            if left:

                percentile_5 = global_data[i].quantile(0.05).item()
                percentile_95 = global_data[i].quantile(0.95).item()

                if self.spatial_split is not None:
                    combined_tensor = torch.zeros(len(self.spatial_split),self.patch_size,self.patch_size)
                    for index, pos in enumerate(split_area):
                        combined_tensor[index] = global_data[i, pos[0]:pos[0]+self.patch_size, pos[1]:pos[1]+self.patch_size]

                    
                    percentile_5_test = combined_tensor.quantile(0.05).item()
                    percentile_95_test = combined_tensor.quantile(0.95).item()
                    
                    title = f'{self.channel_names[i]}: ({percentile_5:.2f}, {percentile_95:.2f}) [{percentile_5_test:.2f}, {percentile_95_test:.2f}]'

                else: 
                    title = f'{self.channel_names[i]} ({percentile_5:.2f}, {percentile_95:.2f})'     

                ax.set_title(title)
                if z_min is not None:
                    im = ax.imshow(global_data[i], cmap='inferno', vmin=z_min, vmax=percentile_95)
                else:
                    im = ax.imshow(global_data[i], cmap='RdBu_r', vmin=percentile_5, vmax=percentile_95)
                #fig.colorbar(im, ax=ax)

                left = False
            
            else:

                percentile_5_era = global_data_era[i].quantile(0.05).item()
                percentile_95_era = global_data_era[i].quantile(0.95).item()

                if self.spatial_split is not None:
                    combined_tensor = torch.zeros(len(self.spatial_split),self.patch_size,self.patch_size)
                    for index, pos in enumerate(split_area):
                        combined_tensor[index] = global_data_era[i, pos[0]:pos[0]+self.patch_size, pos[1]:pos[1]+self.patch_size]

                    
                    percentile_5_test_era = combined_tensor.quantile(0.05).item()
                    percentile_95_test_era = combined_tensor.quantile(0.95).item()
                    
                    title = f'({percentile_5_era:.2f}, {percentile_95_era:.2f}) [{percentile_5_test_era:.2f}, {percentile_95_test_era:.2f}]'

                else: 
                    title = f'({percentile_5_era:.2f}, {percentile_95_era:.2f})'     

                ax.set_title(title)

                if z_min is not None:
                    im = ax.imshow(global_data_era[i], cmap='inferno', vmin=z_min, vmax=percentile_95)
                else:
                    im = ax.imshow(global_data_era[i], cmap='RdBu_r', vmin=percentile_5, vmax=percentile_95)
                # fig.colorbar(im, ax=ax)

                left = True
                

        # plt.savefig(f'{figure_name}.png')
        wandb.log({figure_name: fig})

        
    

