import os
import lightning as L
from torch.utils.data import DataLoader
import torch
from typing import Tuple


from dataset_utils.ERA2CERRA import ERA2CERRA_Dataset, read_stats as read_stats_era2cerra
from dataset_utils.ERA_dataset import ERA_Dataset, read_stats as read_stats_era

import warnings



class Era2CerraDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, downscaling_factor: int = 4, num_workers: int = 4, batch_size: int = 32, crop_size_era: int = 35,
                 crop_size_cerra: int = 35, cropping: str = 'deterministic', masking_prob: float = 0.5, channel_names: list = [], local_testing=False, 
                 constant_channels: bool = False, lat_lon_const_channels: bool = False, return_era_original: bool = False, 
                 spatial_split_train: list = None, spatial_split_val: list = None, spatial_split_test: list = None,
                 return_offset: bool = False, out_channels: list = None, in_channels: list = None, transform: bool = True, standarized: bool = True,
                 use_added_variables_dataset: bool = False, use_separate_dataset: bool = False):
        super().__init__()
        print("Loading Era2Cerra DataModule \n")
        
        self.data_path = data_path
        self.transform = transform
        self.standarized = standarized

        if use_separate_dataset:
            self.mean_cerra, self.std_cerra, self.min_cerra, self.max_cerra = read_stats_era2cerra(os.path.join(data_path, 'CERRA', 'preprocessed_separate'))
            self.mean_era, self.std_era, self.min_era, self.max_era = read_stats_era2cerra(os.path.join(data_path, 'ERA5', 'preprocessed_separate'))
        else:
            self.mean_cerra, self.std_cerra, self.min_cerra, self.max_cerra = read_stats_era2cerra(os.path.join(data_path, 'CERRA', 'preprocessed'))
            self.mean_era, self.std_era, self.min_era, self.max_era = read_stats_era2cerra(os.path.join(data_path, 'ERA5', 'preprocessed'))
        
        self.dataset_metrics_cerra={"variable_mean_cerra": self.mean_cerra, "variable_std_cerra": self.std_cerra, "variable_min_cerra": self.min_cerra, "variable_max_cerra": self.max_cerra}
        self.dataset_metrics_era={"variable_mean_era": self.mean_era, "variable_std_era": self.std_era, "variable_min_era": self.min_era, "variable_max_era": self.max_era}
        self.channel_names = channel_names
        self.local_testing = local_testing
        self.constant_channels = constant_channels
        self.masking_prob = masking_prob

        self.lat_lon_const_channels = lat_lon_const_channels
        self.return_era_original = return_era_original
        self.return_offset = return_offset
        self.cropping = cropping
        self.crop_size_era = crop_size_era
        self.crop_size_cerra = crop_size_cerra

        self.spatial_split_train = spatial_split_train
        self.spatial_split_val = spatial_split_val
        self.spatial_split_test = spatial_split_test

        self.use_added_variables = use_added_variables_dataset
        self.use_separate_dataset = use_separate_dataset

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.save_hyperparameters()

    def setup(self, stage:str):

        if stage == 'fit':
            self.dataset_train = ERA2CERRA_Dataset(self.hparams.data_path, mode='train', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, spatial_split=self.spatial_split_train, return_era_original = self.return_era_original, return_offset=self.return_offset, use_separate_dataset=self.use_separate_dataset)
            self.dataset_val = ERA2CERRA_Dataset(self.hparams.data_path, mode='val', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, spatial_split=self.spatial_split_val, return_era_original = self.return_era_original, return_offset=self.return_offset, use_separate_dataset=self.use_separate_dataset)

        elif stage == 'test':
            self.dataset_test = ERA2CERRA_Dataset(self.hparams.data_path, mode='test', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, return_offset=self.return_offset, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, spatial_split=self.spatial_split_test, return_era_original = self.return_era_original, use_separate_dataset=self.use_separate_dataset)

        elif stage == 'predict':
            self.dataset_full = ERA2CERRA_Dataset(self.hparams.data_path, mode='all', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, return_offset=self.return_offset, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, return_era_original = self.return_era_original, use_separate_dataset=self.use_separate_dataset)

        elif stage == 'validate':
            self.dataset_val = ERA2CERRA_Dataset(self.hparams.data_path, mode='val', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, return_offset=self.return_offset, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, spatial_split=self.spatial_split_val, return_era_original = self.return_era_original, use_separate_dataset=self.use_separate_dataset)

        else:
            self.dataset_train = ERA2CERRA_Dataset(self.hparams.data_path, mode='train', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, return_offset=self.return_offset, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, spatial_split=self.spatial_split_train, return_era_original = self.return_era_original, use_separate_dataset=self.use_separate_dataset)
            self.dataset_val = ERA2CERRA_Dataset(self.hparams.data_path, mode='val', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, return_offset=self.return_offset, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, spatial_split=self.spatial_split_val, return_era_original = self.return_era_original, use_separate_dataset=self.use_separate_dataset)
            self.dataset_test = ERA2CERRA_Dataset(self.hparams.data_path, mode='test', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, return_offset=self.return_offset, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, spatial_split=self.spatial_split_test, return_era_original = self.return_era_original, use_separate_dataset=self.use_separate_dataset)
            self.dataset_full = ERA2CERRA_Dataset(self.hparams.data_path, mode='all', transform=self.transform, standarized=self.standarized, cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size_era=self.crop_size_era, crop_size_cerra=self.crop_size_cerra, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, return_offset=self.return_offset, out_channels = self.out_channels, in_channels = self.in_channels, use_added_variables=self.use_added_variables, masking_prob=self.masking_prob, return_era_original = self.return_era_original, use_separate_dataset=self.use_separate_dataset)
            

    def train_dataloader(self):
        return DataLoader(self.dataset_train, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset_full, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size)
    

class EraDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, downscaling_factor: int = 4, num_workers: int = 4, batch_size: int = 32, crop_size: int = 256, cropping: str = 'deterministic', channel_names: list = [], local_testing=False, constant_channels: bool = False,
                  lat_lon_const_channels: bool = False, return_era_original: bool = False, train_years: Tuple[int, int] = None, val_years: Tuple[int, int] = None, 
                  test_years: Tuple[int, int] = None, return_offset: bool = False,  use_separate_dataset: bool = False):
        super().__init__()
        print("Loading ERA5 DataModule \n")

        
        self.data_path = data_path

        if use_separate_dataset:
            self.mean, self.std, self.min, self.max = read_stats_era2cerra(os.path.join(data_path, 'ERA5', 'preprocessed_america_separate'))
        else:
            self.mean, self.std, self.min, self.max = read_stats_era2cerra(os.path.join(data_path, 'ERA5', 'america_preprocessed'))
        

        self.dataset_metrics={"variable_mean": self.mean, "variable_std": self.std, "variable_min": self.min, "variable_max": self.max}
        self.channel_names = channel_names
        self.local_testing = local_testing
        self.constant_channels = constant_channels
        self.lat_lon_const_channels = lat_lon_const_channels
        self.return_offset = return_offset
        self.cropping = cropping
        self.crop_size = crop_size

        self.use_separate_dataset = use_separate_dataset


        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years

        self.save_hyperparameters()

    def setup(self, stage:str):

        if stage == 'test':
            self.dataset_test = ERA_Dataset(self.hparams.data_path, mode='test', cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size=self.crop_size, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, years=self.test_years, return_offset=self.return_offset, use_separate_dataset=self.use_separate_dataset)

        elif stage == 'predict':
            self.dataset_full = ERA_Dataset(self.hparams.data_path, mode='all', cropping=self.cropping, downscaling_factor=self.hparams.downscaling_factor, crop_size=self.crop_size, constant_channels = self.constant_channels, lat_lon_const_channels = self.lat_lon_const_channels, use_separate_dataset=self.use_separate_dataset)

            
    def train_dataloader(self):
        return DataLoader(self.dataset_train, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset_full, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size)
    