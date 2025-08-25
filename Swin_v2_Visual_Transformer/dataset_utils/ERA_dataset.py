import os
import glob
from typing import Tuple
import warnings
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import avg_pool2d


import h5py

from tqdm import tqdm

from scipy import ndimage
import matplotlib.pyplot as plt

import time



class ERA_Dataset(Dataset):

    def __init__(self, data_dir, mode: str = 'train', transform: bool = True, downscaling_factor:int = 1, cropping: str = 'random',
                  output_dim: int = 3, crop_size:int = 256, dataset_stats: dict = None, constant_channels: bool = False,
                  lat_lon_const_channels: bool = False, masking_prob: float = 0.5, masking_size_ratio: int = 16, return_era_original: bool = False,
                  return_offset: bool = False, years: Tuple[int, int] = None, use_separate_dataset: bool = False,
                  in_channels: list = None, out_channels: list = None):
        """
        Initialize the ERA  dataset.

        Args:
            data_dir (str): The directory path where the dataset is located.
            mode (str, optional): The mode of the dataset. Possible values are 'train', 'val', 'test', or 'full'. Defaults to 'train'.
            transform (bool, optional): Whether to apply transformations to the data. Defaults to True.
            downscaling_factor (int, optional): The downscaling factor for generating low resolution data. Defaults to 1.
            cropping (str, optional): The cropping strategy. Possible values are 'random' or 'deterministic'. Defaults to 'random'.
            output_dim (int, optional): The output dimension of the dataset. Defaults to 3 (c,h,w).
            crop_size (int, optional): The size of the cropped images. Defaults to 256.
            dataset_stats (dict, optional): Statistics of the dataset. Defaults to None.
            constant_channels (bool, optional): Whether to use constant channels. Defaults to False.
            lat_lon_const_channels (bool, optional): Whether to also use latitude and longitude as constant channels. Defaults to False.
            masking_prob (float, optional): The probability of applying masking to the data. Defaults to 0.5.
            masking_size_ratio (int, optional): The ratio of the masking size to the crop size. Defaults to 16.
            return_era_original (bool, optional): If true, dataset also returns the unchanged original era5 data, in addition to the subsampled era5 data and the cerra target. Defaults to False.
            return_offset (bool, optional): If true, dataset also returns the offset of the crop. Defaults to False.
            years (Tuple[int, int], optional): The years to include in the dataset. Defaults to None.
            use_separate_dataset (bool, optional): Whether to use the dataset versions using seperate u10/v10. Defaults to False.
            in_channels (list, optional): The channels to include in the input. Defaults to None.
            out_channels (list, optional): The channels to include in the output. Defaults to None.
        """

        print(f"Initializing ERA5 Dataset for America")
        
        self.data_dir = data_dir
        self.era5_data_dir = os.path.join(data_dir, 'ERA5', 'america_preprocessed')
        if use_separate_dataset:
            print("Using ERA5 dataset with separate u10/v10")
            self.era5_data_dir = os.path.join(data_dir, 'ERA5', 'preprocessed_america_separate')
        
        self.returns_cerra = False
        self.return_era_original = True


        # Generate low resolution data
        self.downscaling_factor = downscaling_factor
        self.filter_type = 'mean'
        self.output_size = 'small'

        self.crop_size = crop_size
        self.original_shape = None

        self.transform = transform
        self.cropping = cropping

        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels is None:
            self.in_channels = list(range(20))
        else:
            print(f"Returning the following channels as LR-input: {self.in_channels}")
            assert len(self.in_channels) <= 20, "Too many input channels"

        if self.out_channels is None:
            self.out_channels = list(range(20))
        else:
            print(f"Returning only the following channels as targets: {self.out_channels}")
            assert len(self.out_channels) <= 20, "Too many output channels"

        #Masking parameters
        self.constant_channels = constant_channels
        self.lat_lon_const_channels = lat_lon_const_channels
        self.masking_prob = masking_prob
        self.masking_size = self.crop_size // masking_size_ratio

        self.years = years


        # The constrained downscaling project requires a 4D output, while the standard is 3D
        self.output_dim = output_dim
        self.return_offset = return_offset
        
        if self.years is None:
            if mode == 'train':
                print("Return dataset in train mode. Years 2016-2017")
                self.years = list(range(2016,2018))
            elif mode == 'val' or mode == 'valid' or mode == 'validation':
                print("Return dataset in validation mode. Year 2018")
                self.years = [2018]
            elif mode == 'test':
                print("Return dataset in test mode. Year 2019")
                self.years = [2019]
            else:
                print("Return dataset in full mode. All years")
                self.years = list(range(2015,2020))
        else:
            self.years = list(range(self.years[0], self.years[1] + 1))
            print(f"Return dataset in custom mode. Years {self.years}")


        # Filter files by extension
        self.era5_files = glob.glob(self.era5_data_dir + "/**/*.h5", recursive=True)

        # Filter files by year
        self.era5_files = sorted([file for file in self.era5_files if int(file[-9:-5]) in self.years])

        print(f"Number of Files {len(self.era5_files)}")

        self.file_lengths = []

        for file in self.era5_files:
            with h5py.File(file, 'r') as f:
                self.file_lengths.append(f['data'].shape[0])


        self.cum_file_length = np.cumsum(self.file_lengths)

        self.number_of_crops_per_sample_x, self.number_of_crops_per_sample_y = self.determine_number_of_crops(files=self.era5_files)
        self.number_of_crops_per_sample = self.number_of_crops_per_sample_x * self.number_of_crops_per_sample_y

        if dataset_stats is None:
            self.variable_mean, self.variable_std, self.variable_min, self.variable_max = read_stats(self.era5_data_dir)
        else:
            self.variable_mean = dataset_stats['variable_mean']
            self.variable_std = dataset_stats['variable_std']
            self.variable_min = dataset_stats['variable_min']
            self.variable_max = dataset_stats['variable_max']

        self.variable_mean = torch.tensor(self.variable_mean, dtype=torch.float32).reshape((-1,1,1))
        self.variable_std = torch.tensor(self.variable_std, dtype=torch.float32).reshape((-1,1,1))

        if cropping == 'deterministic':
            self.file_lengths = [self.file_lengths[i] * self.number_of_crops_per_sample for i in range(len(self.file_lengths))]
            self.cum_file_length = np.cumsum(self.file_lengths)
            print(f"Dataset running in deterministic cropping mode. Number of crops per sample: {self.number_of_crops_per_sample}. Total length: {len(self)}")
        
        if self.constant_channels:
            self.constant = get_constant_data(self.era5_data_dir, self.lat_lon_const_channels)
            self.constant_pooled = avg_pool2d(self.constant,
                                            kernel_size=self.masking_size,
                                            stride=1,
                                            padding=self.masking_size//2)[:,:self.constant.shape[1], :self.constant.shape[2]]
        
            


    def __len__(self):
        return sum(self.file_lengths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find the file that contains the index
        file_idx: int = np.searchsorted(self.cum_file_length, idx, side='right')
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cum_file_length[file_idx - 1] 
        
        if self.cropping == 'deterministic':
            local_idx = local_idx // self.number_of_crops_per_sample
            crop_idx = idx % self.number_of_crops_per_sample

        era_file = self.era5_files[file_idx]

        
        
        # Read the file
        if era_file.endswith('.h5'):
            with h5py.File(era_file, 'r') as f:
                if self.original_shape is None:
                    self.original_shape = torch.tensor(f['data'][local_idx], dtype=torch.float32).shape
                
                if self.transform:
                    if self.cropping == 'deterministic':
                        y_pos = (crop_idx // self.number_of_crops_per_sample_x) * self.crop_size
                        x_pos = (crop_idx % self.number_of_crops_per_sample_x) * self.crop_size
                    else:
                        x_pos = random.randint(0, self.original_shape[1] - self.crop_size)
                        y_pos = random.randint(0, self.original_shape[2] - self.crop_size)

                    era_data = torch.tensor(f['data'][local_idx, :, y_pos:y_pos + self.crop_size, x_pos:x_pos + self.crop_size], dtype=torch.float32)

                    if self.downscaling_factor != 1:
                        era_data_lr = self.compute_low_res(era_data, self.downscaling_factor)
                    else:
                        era_data_lr = era_data.clone()

                else:
                    era_data = torch.tensor(f['data'][local_idx], dtype=torch.float32)

            if len(era_data.shape) == 2:
                era_data = era_data.unsqueeze(0)
                era_data_lr = era_data_lr.unsqueeze(0)
            
        else:
            raise ValueError('Unknown file format')
        
        
        if self.constant_channels:
            constant = self.constant[:, y_pos:y_pos + self.crop_size, x_pos:x_pos + self.crop_size]
            constant_pooled = self.constant_pooled[:, y_pos:y_pos + self.crop_size, x_pos:x_pos + self.crop_size]

            mask = torch.bernoulli(torch.full((self.crop_size//self.masking_size,self.crop_size//self.masking_size), self.masking_prob)).int()
            mask = mask.repeat_interleave(self.masking_size, dim=0).repeat_interleave(self.masking_size, dim=1)
            constant_masked = constant * (1-mask) + mask * constant_pooled
            

        if era_data.min() <= -1e7 or era_data.max() >= 1e7 or torch.isnan(era_data).any() or torch.isinf(era_data).any():
            warnings.warn(f"Data out of bounds in file {era_file} at local index {local_idx}, min: {era_data.min()}, max: {era_data.max()} \n Returning random sample instead.")
            warnings.warn(f"Some Data is out of bounds (too small or too large). Returning random sample instead.")
            return self.__getitem__(random.randint(0, len(self) - 1))


        if self.transform:
            
            #Normalize the data
            
            era_data = (era_data - self.variable_mean) / self.variable_std
            era_data_lr = (era_data_lr - self.variable_mean) / self.variable_std

         
            if self.constant_channels:
                
                if self.return_offset:
                    return (era_data_lr, constant_masked), (era_data, constant), (y_pos, x_pos)
                
                return (era_data_lr, constant_masked), (era_data, constant)
            
            if self.return_offset:
                return era_data_lr, era_data, (y_pos, x_pos)
            
            if self.output_dim == 4:
                return era_data_lr.unsqueeze(0), era_data.unsqueeze(0)
            
            return era_data_lr, era_data
        
        return era_data_lr, era_data

    def compute_low_res(self, data, factor=4):
        filter_size = (1, factor, factor)
        if self.filter_type == 'mean':
            filtered_data = ndimage.uniform_filter(data, size=filter_size)
        elif self.filter_type == 'median':
            filtered_data = ndimage.median_filter(data, size=filter_size)
        else:
            raise ValueError('Unknown filter type')
        
        if self.output_size == 'same':
            return torch.tensor(filtered_data, dtype=torch.float32)
        
        return torch.tensor(filtered_data[:, ::factor, ::factor], dtype=torch.float32)
        
   
    def determine_number_of_crops(self, files):
        if self.original_shape is None:
            file = files[0]
        
            # Read the file
            if file.endswith('.h5'):
                with h5py.File(file, 'r') as f:
                    data = f['data'][0]
                    data = torch.tensor(data)

                if len(data.shape) == 2:
                    data = data.unsqueeze(0)

            self.original_shape = data.shape
        
        y_crops = self.original_shape[1] // self.crop_size
        x_crops = self.original_shape[2] // self.crop_size

        return x_crops, y_crops
    

### Utility functions ####

def read_stats(data_dir) -> tuple[float, float]:
    """
    Reads the statistical values (mean, standard deviation, minimum, maximum) from a file in the given data directory.

    Args:
        data_dir (str): The path to the directory containing the statistical file.

    Returns:
        tuple[float, float, float, float]: A tuple containing the mean, standard deviation, minimum, and maximum values.

    Raises:
        ValueError: If there is an error reading the statistical file, default values are used instead.
    """
    if "stats.pt" in os.listdir(data_dir):
        
        stats = torch.load(os.path.join(data_dir, 'stats.pt'), weights_only=False)
        variable_mean = stats['mean']
        variable_std = stats['std']
        min_val = stats['min']
        max_val = stats['max']
        print(f"Loaded Mean/Std from file")
    else:
        print("Error reading mean/std file. Using default values.")
        variable_mean = 0
        variable_std = 1
        min_val = -1e10
        max_val = 1e10

    return variable_mean, variable_std, min_val, max_val

def get_constant_data(data_dir, lat_lon: bool = False):
    """
    Reads the constant data (e.g. orography, land-sea-mask) from a file in the given data directory.
    Automatically normalizes the data.

    Args:
        data_dir (str): The path to the directory containing the constant data file.

    Returns:
        torch.Tensor: The constant data.

    Raises:
        ValueError: If there is an error reading the constant data file, an error is raised.
    """
    if "america_const_rough.pt" in os.listdir(data_dir):
        const: torch.tensor = torch.load(os.path.join(data_dir, 'america_const_rough.pt'), weights_only=False)
        print(f"Loaded constant data from file")

        print("Testing const normalization")
        const = (const - torch.tensor([0.5281, 236.8313]).view(2, 1, 1))/ torch.tensor([0.4922,444.1848]).view(2, 1, 1)
        # const = (const - torch.mean(const, dim=(1,2), keepdim=True)) / torch.std(const, dim=(1,2), keepdim=True)

    else:
        raise ValueError("Error reading constant data file.")
    
    if lat_lon:
        if "america_lat_lon.pt" in os.listdir(data_dir):
            lat_lon: torch.tensor = torch.load(os.path.join(data_dir, 'america_lat_lon.pt'), weights_only=False).to(torch.float32)
            print(f"Loaded lat/lon constant data from file")

            lat_lon = (lat_lon - torch.mean(lat_lon, dim=(1,2), keepdim=True)) / torch.std(lat_lon, dim=(1,2), keepdim=True)

            const = torch.cat((const, lat_lon), dim=0)
    
        else:
            raise ValueError("Error reading lat/lon constant data file.")

    return const
