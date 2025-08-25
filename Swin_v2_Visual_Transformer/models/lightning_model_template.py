import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True) # To check possible errors during gradient calculation
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


import lightning as L
from torchmetrics.image import PeakSignalNoiseRatio
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn

from collections import defaultdict
import os
import glob

import xarray as xr

from metrics.metrics import metrics

class LightningModelTemplate(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch[0:2]
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0:2]
        y_hat = self(x)
        val_loss = self.loss_function(y_hat, y)

        y_hat_s = y_hat * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
        y_s = y * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]

        val_mse = self.loss_function(y_hat_s, y_s)

        channelwise_metrics = self.perform_channelwise_evaluation(y_hat_s, y_s, y_hat, y, self.channel_names[self.out_channels])

        self.log_dict(channelwise_metrics, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict({"val_loss": val_loss, "val_psnr": self.valid_psnr(y_hat, y), "val_mse": val_mse}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        test_loss = self.loss_function(y_hat, y)
        
        y_hat_s = y_hat * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]
        y_s = y * self.variable_std[:, self.out_channels] + self.variable_mean[:, self.out_channels]

        if self.hparams['loss_function'] == 'cyclic_loss':
            test_mse = self.mse_function(y_hat_s, y_s)
            self.log_dict({"test_psnr_cyclic": self.psnr_cyclic(y_hat, y)}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        else:
            test_mse = self.loss_function(y_hat_s, y_s)


        channelwise_metrics = self.perform_channelwise_evaluation(y_hat_s, y_s, y_hat, y, self.channel_names[self.out_channels])

        if self.seperate_dataset:
            si10 = torch.sqrt(y_s[:, 0]**2 + y_s[:, 1]**2)
            wdir10 = torch.arctan2(y_s[:, 0],y_s[:, 1])
            wdir10 = torch.remainder(180.0 + torch.rad2deg(wdir10), 360.0)

            si10_hat = torch.sqrt(y_hat_s[:, 0]**2 + y_hat_s[:, 1]**2)
            wdir10_hat = torch.arctan2(y_hat_s[:, 0],y_hat_s[:, 1])
            wdir10_hat = torch.remainder(180.0 + torch.rad2deg(wdir10_hat), 360.0)

            si10 = (si10 - 5.303656134765812) / 3.695860737416358
            wdir10 = (wdir10 - 183.88709676666727) / 107.8809532565079

            si10_hat = (si10_hat - 5.303656134765812) / 3.695860737416358
            wdir10_hat = (wdir10_hat - 183.88709676666727) / 107.8809532565079

            channelwise_metrics['mse_si10'] = F.mse_loss(si10_hat, si10).mean()
            channelwise_metrics['mse_wdir10'] = (torch.min(torch.abs(wdir10_hat-wdir10), (360 / 107.8809532565079) - torch.abs(wdir10_hat-wdir10)) ** 2).mean()



        self.log_dict(channelwise_metrics, on_step=False, on_epoch=True, sync_dist=self.sync_dist)

        self.log_dict({"test_loss": test_loss, "test_psnr": self.test_psnr(y_hat, y), "test_mse": test_mse}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)

        return y_hat_s

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    

    def perform_channelwise_evaluation(self, y_hat_s, y_s, y_hat, y, channel_names = [], loss_function = None):
        if loss_function is None:
            mse_channelwise = self.loss_function(y_hat, y, reduction='none').mean(dim=(0, 2, 3))
        else:
            mse_channelwise = loss_function(y_hat, y, reduction='none').mean(dim=(0, 2, 3))
        psnr_channelwise = torch.empty_like(mse_channelwise)

        for i in range(y_s.shape[1]):
            psnr_channelwise[i] = (self.valid_psnr(y_hat_s[:, i, :, :], y_s[:, i, :, :]))

        results = {}
        if len(channel_names) != 0:
            for i in range(len(channel_names)):
                results['mse_' + channel_names[i]] = mse_channelwise[i]
                results['psnr_' + channel_names[i]] = psnr_channelwise[i]            

        return results


class LightningModelTemplateSidechannel(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        x, x_sidechannel = x
        y, y_sidechannel = y

        y_hat, y_hat_sidechannel = self((x, y_sidechannel)) # x_sidechannel))

        loss = self.loss_function(y_hat, y)
        loss_sidechannel = F.mse_loss(y_hat_sidechannel, y_sidechannel)        
        combined_loss = loss + self.loss_beta * loss_sidechannel
        
        self.log_dict({'train_loss': loss, 'train_loss_sidechannel': loss_sidechannel, 'train_combined_loss': combined_loss})
        return combined_loss
    
    def on_validation_start(self):
        
        self.reconstructed_y = defaultdict(lambda: torch.zeros((self.out_chans, self.full_height_cerra, self.full_width_cerra), device='cpu'))
        self.reconstructed_y_hat = defaultdict(lambda: torch.zeros((self.out_chans, self.full_height_cerra, self.full_width_cerra), device='cpu'))
        self.reconstructed_y_hat_sidechannel = defaultdict(lambda: torch.zeros((self.num_sidechannels, self.full_height_cerra, self.full_width_cerra), device='cpu'))
        self.full_count = defaultdict(lambda: torch.zeros((1, 420, 700), device='cpu'))

    def validation_step(self, batch, batch_idx):
        x, y, pos, _ = batch
        x, x_sidechannel = x
        y, y_sidechannel = y

        y_pos_cerra, x_pos_cerra, y_pos_era, x_pos_era, local_idx = pos

        y_hat, y_hat_sidechannel = self((x, y_sidechannel))

        val_loss = self.loss_function(y_hat, y)     
        val_loss_sidechannel = F.mse_loss(y_hat_sidechannel, y_sidechannel)
        val_combined_loss = val_loss + self.loss_beta * val_loss_sidechannel

        # If using patch-wise processing, accumulate into full-sized tensor
        if self.transform:

            for i in range(len(local_idx)):
                key = local_idx[i].item()
                y0 = y_pos_cerra[i].item()
                x0 = x_pos_cerra[i].item()

                y_patch = y[i].detach().cpu()
                y_hat_patch = y_hat[i].detach().cpu()
                y_hat_sidechannel_patch = y_hat_sidechannel[i].detach().cpu()

                h, w = y_hat_patch.shape[-2:]

                self.reconstructed_y[key][:, y0:y0+h, x0:x0+w] += y_patch
                self.reconstructed_y_hat[key][:, y0:y0+h, x0:x0+w] += y_hat_patch
                self.reconstructed_y_hat_sidechannel[key][:, y0:y0+h, x0:x0+w] += y_hat_sidechannel_patch
                self.full_count[key][:, y0:y0+h, x0:x0+w] += 1 #just to check for overlapping pixels

        # Else, assign the full-sized image directly
        else:
            for i in range(len(local_idx)):
                key = local_idx[i].item()
                self.reconstructed_y[key] += y[i].detach().cpu()
                self.reconstructed_y_hat[key] += y_hat[i].detach().cpu()
                self.reconstructed_y_hat_sidechannel[key] += y_hat_sidechannel[i].detach().cpu()
                self.full_count[key] += 1 

        self.log_dict({'val_loss': val_loss, 'val_loss_sidechannel': val_loss_sidechannel, 'val_combined_loss': val_combined_loss}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        
        return val_loss
    
    def on_validation_epoch_end(self):

        for key in self.full_count:
            num_zeros = (self.full_count[key] == 0).sum().item()
            num_overlaps = (self.full_count[key] > 1).sum().item()

            if num_zeros>0:
                print(f"Warning: count has zeros (uncovered pixels) for key {key}")
            if num_overlaps > 0:
                print(f"Warning: count has overlaps (pixels covered more than once) for key {key}")

        if self.trainer.sanity_checking:
            return
        
        reconstructed_y = [self.reconstructed_y[key] for key in self.reconstructed_y]
        reconstructed_y_hat = [self.reconstructed_y_hat[key] for key in self.reconstructed_y_hat]
        reconstructed_y_hat_sidechannel = [self.reconstructed_y_hat_sidechannel[key] for key in self.reconstructed_y_hat_sidechannel]
        full_count = [self.full_count[key] for key in self.full_count]
        
        reconstructed_y = torch.stack(reconstructed_y, dim=0)
        reconstructed_y_hat = torch.stack(reconstructed_y_hat, dim=0)
        reconstructed_y_hat_sidechannel = torch.stack(reconstructed_y_hat_sidechannel, dim=0)
        full_count = torch.stack(full_count, dim=0)

        # Mask: 1 if count > 0, 0 if count == 0
        mask = full_count > 0

        # Normalization
        reconstructed_y = torch.where(mask, reconstructed_y / full_count, torch.full_like(reconstructed_y, torch.nan))
        reconstructed_y_hat = torch.where(mask, reconstructed_y_hat / full_count, torch.full_like(reconstructed_y_hat, torch.nan))
        reconstructed_y_hat_sidechannel = torch.where(mask, reconstructed_y_hat_sidechannel / full_count, torch.full_like(reconstructed_y_hat_sidechannel, torch.nan))

        if self.standarized:
            mean_cerra = self.variable_mean_cerra[:, self.out_channels].detach().cpu()
            std_cerra = self.variable_std_cerra[:, self.out_channels].detach().cpu()

            reconstructed_y_s = reconstructed_y * std_cerra + mean_cerra
            reconstructed_y_hat_s = reconstructed_y_hat * std_cerra + mean_cerra
        else:
            reconstructed_y_s = reconstructed_y
            reconstructed_y_hat_s = reconstructed_y_hat

        #### Calculation of metrics
        output_metrics_path = os.path.join(os.getcwd(), self.results_dir, 'metrics', 'valid')
        if not os.path.exists(output_metrics_path):
            os.makedirs(output_metrics_path)

        map_metric_dict = {}
        scalar_metric_dict = {}

        for i in range(self.out_chans):
            ch_name = self.channel_names[i]
            map_dict_i, scalar_dict_i = metrics(reconstructed_y_hat_s[:, i, :, :], reconstructed_y_s[:, i, :, :], output_metrics_path, self.land_mask_bool, self.metrics_names, self.current_epoch)

            map_metric_dict.update({f"{k}_{ch_name}": v for k, v in map_dict_i.items()})
            scalar_metric_dict.update({f"{k}_{ch_name}": v for k, v in scalar_dict_i.items()})
            
        self.log_dict(scalar_metric_dict, on_step=False, on_epoch=True, sync_dist=self.sync_dist)       

        # Paths to read coords, name vars and save data and metric maps
        cerra_data_dir = os.path.join(os.getcwd(), 'data', 'CERRA', 'preprocessed_separate')
        cerra_files = glob.glob(os.path.join(cerra_data_dir, 'val') + "/**/*.nc", recursive=True)
        ds_cerra = xr.open_dataset(os.path.join(cerra_data_dir, cerra_files[0]))
        ds_aux_file = xr.open_dataset(os.path.join(cerra_data_dir, 'aux_variables_standarized.nc'))

        output_data_path = os.path.join(os.getcwd(), self.results_dir, 'data', 'valid')
        if not os.path.exists(output_data_path):
            os.makedirs(output_data_path)

        coords = {dim: ds_cerra.coords[dim] for dim in ['time', 'latitude', 'longitude']}

        cerra_var_names = list(ds_cerra.data_vars) 
        aux_var_names = list(ds_aux_file.data_vars) 

        np_y_hat = reconstructed_y_hat_s.cpu().numpy()
        np_y_hat_sidechannel = reconstructed_y_hat_sidechannel.cpu().numpy()

        ds_y_hat = xr.Dataset()
        for i, var_name in enumerate(cerra_var_names):
            data_var = np_y_hat[:, i, :, :]  # shape (time, lat, lon)
            ds_y_hat[var_name] = (['time', 'latitude', 'longitude'], data_var)
        ds_y_hat = ds_y_hat.assign_coords(coords)
        if self.trainer.state.fn == TrainerFn.FITTING:
            ds_y_hat.to_netcdf(os.path.join(output_data_path, f'reconstructed_y_hat_s_{self.current_epoch:04d}.nc'))
        elif self.trainer.state.fn == TrainerFn.VALIDATING:
            ds_y_hat.to_netcdf(os.path.join(output_data_path, f'reconstructed_y_hat_s_validation.nc'))
        
        del cerra_var_names, data_var, np_y_hat, ds_y_hat

        ds_sidechannel = xr.Dataset()
        for i, var_name in enumerate(aux_var_names):
            data_var = np_y_hat_sidechannel[:, i, :, :]  # shape (time, lat, lon)
            ds_sidechannel[var_name] = (['time', 'latitude', 'longitude'], data_var)
        ds_sidechannel = ds_sidechannel.assign_coords(coords)
        if self.trainer.state.fn == TrainerFn.FITTING:
            ds_sidechannel.to_netcdf(os.path.join(output_data_path, f'reconstructed_y_hat_sidechannel_{self.current_epoch:04d}.nc'))
        elif self.trainer.state.fn == TrainerFn.VALIDATING:
            ds_sidechannel.to_netcdf(os.path.join(output_data_path, f'reconstructed_y_hat_sidechannel_validation.nc'))
        

        del aux_var_names, data_var, np_y_hat_sidechannel, ds_sidechannel, coords 

        coords_space = {dim: ds_cerra.coords[dim] for dim in ['latitude', 'longitude']}
        coords_time = {dim: ds_cerra.coords[dim] for dim in ['time']}

        ds_metrics_space = xr.Dataset()
        ds_metrics_time = xr.Dataset()
        for metric_name, metric_map in map_metric_dict.items():

            if isinstance(metric_map, torch.Tensor):
                metric_map = metric_map.numpy()

            if "time" in metric_name:
                dims = ['latitude', 'longitude']

                ds_metrics_time[metric_name] = (dims, metric_map)
                ds_metrics_time = ds_metrics_time.assign_coords(coords_space)
                if self.trainer.state.fn == TrainerFn.FITTING:
                    ds_metrics_time.to_netcdf(os.path.join(output_metrics_path, f'metric_time_maps_{self.current_epoch:04d}.nc'))
                elif self.trainer.state.fn == TrainerFn.VALIDATING:
                    ds_metrics_time.to_netcdf(os.path.join(output_metrics_path, f'metric_time_maps_validation.nc'))

            elif "space" in metric_name:
                dims = ['time']

                ds_metrics_space[metric_name] = (dims, metric_map)
                ds_metrics_space = ds_metrics_space.assign_coords(coords_time)
                if self.trainer.state.fn == TrainerFn.FITTING:
                    ds_metrics_space.to_netcdf(os.path.join(output_metrics_path, f'metric_space_maps_{self.current_epoch:04d}.nc'))
                elif self.trainer.state.fn == TrainerFn.VALIDATING:
                    ds_metrics_space.to_netcdf(os.path.join(output_metrics_path, f'metric_space_maps_validation.nc'))

            else:
                raise ValueError(f"No dimension found in {metric_name}")


    def on_test_start(self):
        
        self.reconstructed_y = defaultdict(lambda: torch.zeros((self.out_chans, self.full_height_cerra, self.full_width_cerra), device='cpu'))
        self.reconstructed_y_hat = defaultdict(lambda: torch.zeros((self.out_chans, self.full_height_cerra, self.full_width_cerra), device='cpu'))
        self.reconstructed_y_hat_sidechannel = defaultdict(lambda: torch.zeros((self.num_sidechannels, self.full_height_cerra, self.full_width_cerra), device='cpu'))
        self.full_count = defaultdict(lambda: torch.zeros((1, 420, 700), device='cpu'))        
    
    def test_step(self, batch, batch_idx):

        x, y, pos, _ = batch
        x, x_sidechannel = x
        y, y_sidechannel = y

        y_pos_cerra, x_pos_cerra, y_pos_era, x_pos_era, local_idx = pos

        y_hat, y_hat_sidechannel = self((x, y_sidechannel))

        test_loss = self.loss_function(y_hat, y)       
        test_loss_sidechannel = F.mse_loss(y_hat_sidechannel, y_sidechannel)
        test_combined_loss = test_loss + self.loss_beta * test_loss_sidechannel   

        # If using patch-wise processing, accumulate into full-sized tensor
        if self.transform:

            for i in range(len(local_idx)):
                key = local_idx[i].item()
                y0 = y_pos_cerra[i].item()
                x0 = x_pos_cerra[i].item()

                y_patch = y[i].detach().cpu()
                y_hat_patch = y_hat[i].detach().cpu()
                y_hat_sidechannel_patch = y_hat_sidechannel[i].detach().cpu()

                h, w = y_hat_patch.shape[-2:]

                self.reconstructed_y[key][:, y0:y0+h, x0:x0+w] += y_patch
                self.reconstructed_y_hat[key][:, y0:y0+h, x0:x0+w] += y_hat_patch
                self.reconstructed_y_hat_sidechannel[key][:, y0:y0+h, x0:x0+w] += y_hat_sidechannel_patch
                self.full_count[key][:, y0:y0+h, x0:x0+w] += 1 #just to check for overlapping pixels
        
         # Else, assign the full-sized image directly
        else:
            for i in range(len(local_idx)):
                key = local_idx[i].item()
                self.reconstructed_y[key] += y[i].detach().cpu()
                self.reconstructed_y_hat[key] += y_hat[i].detach().cpu()
                self.reconstructed_y_hat_sidechannel[key] += y_hat_sidechannel[i].detach().cpu()
                self.full_count[key] += 1 

        self.log_dict({'test_loss': test_loss, 'test_loss_sidechannel': test_loss_sidechannel, 'test_combined_loss': test_combined_loss}, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
            
        return test_loss
    
    def on_test_epoch_end(self):

        for key in self.full_count:
            num_zeros = (self.full_count[key] == 0).sum().item()
            num_overlaps = (self.full_count[key] > 1).sum().item()

            if num_zeros>0:
                print("Warning: count has zeros (uncovered pixels) for key {key}")
            if num_overlaps > 0:
                print("Warning: count has overlaps (pixels covered more than once) for key {key}")

        reconstructed_y = [self.reconstructed_y[key] for key in self.reconstructed_y]
        reconstructed_y_hat = [self.reconstructed_y_hat[key] for key in self.reconstructed_y_hat]
        reconstructed_y_hat_sidechannel = [self.reconstructed_y_hat_sidechannel[key] for key in self.reconstructed_y_hat_sidechannel]
        full_count = [self.full_count[key] for key in self.full_count]
        
        reconstructed_y = torch.stack(reconstructed_y, dim=0)
        reconstructed_y_hat = torch.stack(reconstructed_y_hat, dim=0)
        reconstructed_y_hat_sidechannel = torch.stack(reconstructed_y_hat_sidechannel, dim=0)
        full_count = torch.stack(full_count, dim=0)
        
        # Mask: 1 if count > 0, 0 if count == 0
        mask = full_count > 0

        # Normalization
        reconstructed_y = torch.where(mask, reconstructed_y / full_count, torch.full_like(reconstructed_y, torch.nan))
        reconstructed_y_hat = torch.where(mask, reconstructed_y_hat / full_count, torch.full_like(reconstructed_y_hat, torch.nan))
        reconstructed_y_hat_sidechannel = torch.where(mask, reconstructed_y_hat_sidechannel / full_count, torch.full_like(reconstructed_y_hat_sidechannel, torch.nan))

        if self.standarized:
            mean_cerra = self.variable_mean_cerra[:, self.out_channels].detach().cpu()
            std_cerra = self.variable_std_cerra[:, self.out_channels].detach().cpu()

            reconstructed_y_s = reconstructed_y * std_cerra + mean_cerra
            reconstructed_y_hat_s = reconstructed_y_hat * std_cerra + mean_cerra
        else:
            reconstructed_y_s = reconstructed_y
            reconstructed_y_hat_s = reconstructed_y_hat

        #### Calculation of metrics
        output_metrics_path = os.path.join(os.getcwd(), self.results_dir, 'metrics', 'test')
        if not os.path.exists(output_metrics_path):
            os.makedirs(output_metrics_path)

        map_metric_dict = {}
        scalar_metric_dict = {}

        for i in range(self.out_chans):
            ch_name = self.channel_names[i]
            map_dict_i, scalar_dict_i = metrics(reconstructed_y_hat_s[:, i, :, :], reconstructed_y_s[:, i, :, :], output_metrics_path, self.land_mask_bool, self.metrics_names, self.current_epoch)

            map_metric_dict.update({f"{k}_{ch_name}": v for k, v in map_dict_i.items()})
            scalar_metric_dict.update({f"{k}_{ch_name}": v for k, v in scalar_dict_i.items()})
            
        self.log_dict(scalar_metric_dict, on_step=False, on_epoch=True, sync_dist=self.sync_dist)


        # Paths to read coords, name vars and save data
        cerra_data_dir = os.path.join(os.getcwd(), 'data', 'CERRA', 'preprocessed_separate')
        cerra_files = glob.glob(os.path.join(cerra_data_dir, 'test') + "/**/*.nc", recursive=True)
        ds_cerra = xr.open_dataset(os.path.join(cerra_data_dir, cerra_files[0]))
        ds_aux_file = xr.open_dataset(os.path.join(cerra_data_dir, 'aux_variables_standarized.nc'))

        output_data_path = os.path.join(os.getcwd(), self.results_dir, 'data', 'test')
        if not os.path.exists(output_data_path):
            os.makedirs(output_data_path)

        coords = {dim: ds_cerra.coords[dim] for dim in ['time', 'latitude', 'longitude']}

        cerra_var_names = list(ds_cerra.data_vars) 
        aux_var_names = list(ds_aux_file.data_vars) 

        np_y_hat = reconstructed_y_hat_s.cpu().numpy()
        np_y_hat_sidechannel = reconstructed_y_hat_sidechannel.cpu().numpy()

        ds_y_hat = xr.Dataset()
        for i, var_name in enumerate(cerra_var_names):
            data_var = np_y_hat[:, i, :, :]  # shape (time, lat, lon)
            ds_y_hat[var_name] = (['time', 'latitude', 'longitude'], data_var)
        ds_y_hat = ds_y_hat.assign_coords(coords)
        ds_y_hat.to_netcdf(os.path.join(output_data_path, f'reconstructed_y_hat_s_test.nc'))

        del cerra_var_names, data_var, np_y_hat, ds_y_hat

        ds_sidechannel = xr.Dataset()
        for i, var_name in enumerate(aux_var_names):
            data_var = np_y_hat_sidechannel[:, i, :, :]  # shape (time, lat, lon)
            ds_sidechannel[var_name] = (['time', 'latitude', 'longitude'], data_var)
        ds_sidechannel = ds_sidechannel.assign_coords(coords)
        ds_sidechannel.to_netcdf(os.path.join(output_data_path, f'reconstructed_y_hat_sidechannel_test.nc'))

        del aux_var_names, data_var, np_y_hat_sidechannel, ds_sidechannel, coords 

        coords_space = {dim: ds_cerra.coords[dim] for dim in ['latitude', 'longitude']}
        coords_time = {dim: ds_cerra.coords[dim] for dim in ['time']}

        ds_metrics_space = xr.Dataset()
        ds_metrics_time = xr.Dataset()
        for metric_name, metric_map in map_metric_dict.items():

            if isinstance(metric_map, torch.Tensor):
                metric_map = metric_map.numpy()

            if "time" in metric_name:
                dims = ['latitude', 'longitude']

                ds_metrics_time[metric_name] = (dims, metric_map)
                ds_metrics_time = ds_metrics_time.assign_coords(coords_space)
                ds_metrics_time.to_netcdf(os.path.join(output_metrics_path, f'metric_time_maps_test.nc'))
                

            elif "space" in metric_name:
                dims = ['time']

                ds_metrics_space[metric_name] = (dims, metric_map)
                ds_metrics_space = ds_metrics_space.assign_coords(coords_time)
                ds_metrics_space.to_netcdf(os.path.join(output_metrics_path, f'metric_space_maps_test.nc'))

            else:
                raise ValueError(f"No dimension found in {metric_name}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lr_scheduling:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)
            return [optimizer], [scheduler]
        
        return [optimizer]
    

    def perform_channelwise_evaluation(self, y_hat_s, y_s, channel_names = [], loss_function = None):
        loss_fn = loss_function if loss_function is not None else self.loss_fuction 
        mse_channelwise = loss_fn(y_hat_s, y_s, reduction='none').mean(dim=(0, 2, 3))

        psnr_channelwise = torch.empty_like(mse_channelwise)

        for i in range(y_s.shape[1]):
            psnr_channelwise[i] = (self.valid_psnr(y_hat_s[:, i, :, :], y_s[:, i, :, :]))

        results = {}
        if len(channel_names) != 0:
            for i in range(len(channel_names)):
                results['mse_' + channel_names[i]] = mse_channelwise[i]
                results['psnr_' + channel_names[i]] = psnr_channelwise[i]            

        return results
    

def get_outnorm(x:torch.Tensor, out_norm:str='') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1]*img_shape[-2]

    return norm


class cyclicMSELoss(nn.Module):
    def __init__(self, cyclic_indices: list, output_length: int, reduction='mean', norm_data: bool = True):
        super(cyclicMSELoss, self).__init__()

        self.cyclic_indices = cyclic_indices
        self.cyclic_map = torch.zeros(output_length, dtype=torch.bool)
        self.cyclic_map[cyclic_indices] = True

        if norm_data:
            # self.converted_degrees = (360 - 183.88709676666727) / 107.8809532565079
            self.converted_degrees = 360 / 107.8809532565079
        else:
            self.converted_degrees = 360


        # self.cyclic_factor = len(cyclic_indices)
        # self.normal_factor = output_length - self.cyclic_factor

    def forward(self, predictions, targets, reduction: str = 'mean'):


        error = (predictions - targets)
        error[:, self.cyclic_map] = torch.min(torch.abs(error[:, self.cyclic_map]), self.converted_degrees - torch.abs(error[:, self.cyclic_map]))

        if reduction == 'mean':
            return torch.mean(error**2)
        elif reduction == 'none':
            return error**2
        else:
            raise ValueError("Invalid reduction type for cyclicMSELoss")
        
class cyclicPSNR(nn.Module):
    
    def __init__(self, cyclic_indices: list, norm_data: bool = True):
        super(cyclicPSNR, self).__init__()

        self.psnr_func = PeakSignalNoiseRatio()
        self.cyclic_indices = cyclic_indices
        
        if norm_data:
            # self.converted_degrees = (360 - 183.88709676666727) / 107.8809532565079
            self.converted_degrees = 360 / 107.8809532565079
        else:
            self.converted_degrees = 360


    def forward(self, predictions, target, reduction: str = 'mean'):

        modified_pred = predictions.clone()

        for i in self.cyclic_indices:
            error = torch.min(torch.abs(modified_pred[:, i] - target[:, i]), self.converted_degrees - torch.abs(modified_pred[:, i] - target[:, i]))
            modified_pred[:, i] = target[:, i] - error

        return self.psnr_func(modified_pred, target)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, out_norm:str='bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss*norm
  
def grad_loss(v, v_gt):
    # Gradient loss
    loss = torch.mean(torch.abs(v - v_gt), dim=[1,2,3])
    
    jy = v[:, :, 1:, :] - v[:, :, :-1, :]
    jy_ = v_gt[:, :, 1:, :] - v_gt[:, :, :-1, :]

    jx = v[:, :, :, 1:] - v[:, :, :, :-1]
    jx_ = v_gt[:, :, :, 1:] - v_gt[:, :, :, :-1]
    
    loss += torch.mean(torch.abs(jy - jy_), dim=[1,2,3])
    loss += torch.mean(torch.abs(jx - jx_), dim=[1,2,3])
    
    return loss.mean()

def ch_grad_loss(pred, target, alpha=1.0, gamma=0.2):
    charbonnier = CharbonnierLoss()
    return alpha * charbonnier(pred, target) + gamma * grad_loss(pred, target)

def rescale_01(x):
    min_val = x.amin(dim=[2,3], keepdim=True)
    max_val = x.amax(dim=[2,3], keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-8)

def ch_ssim_loss(pred, target, alpha=1.0, beta=0.5, data_range=1.0):
    charbonnier = CharbonnierLoss()
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
    pred_ssim = rescale_01(pred)
    target_ssim = rescale_01(target)
    ssim_val = ssim(pred_ssim, target_ssim)
    return alpha * charbonnier(pred, target) + beta * (1 - ssim_val)

def ch_ssim_grad_loss(pred, target, alpha=1.0, beta=0.5, gamma=0.2, data_range=1.0):
    charbonnier = CharbonnierLoss()
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    pred_ssim = rescale_01(pred)
    target_ssim = rescale_01(target)
    ssim_val = ssim(pred_ssim, target_ssim)

    return alpha * charbonnier(pred, target) + beta * (1 - ssim_val) + gamma * grad_loss(pred, target)

