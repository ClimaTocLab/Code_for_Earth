import os
import glob
from typing import Tuple
import warnings
import random
import xarray as xr

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import avg_pool2d


import h5py

from tqdm import tqdm

from scipy import ndimage
import matplotlib.pyplot as plt

import time



class ERA2CERRA_Dataset(Dataset):

    def __init__(self, data_dir, mode: str = 'train', transform: bool = True, standarized: bool = True, downscaling_factor:int = 1, cropping: str = 'random', 
                 output_dim: int = 3, crop_size_era:int = 35, crop_size_cerra:int = 140, dataset_stats_cerra: dict = None, dataset_stats_era: dict = None, constant_channels: bool = False,
                 lat_lon_const_channels: bool = False, masking_prob: float = 0.5, masking_size_ratio: int = 7, return_era_original: bool = False,
                 return_offset: bool = False, use_added_variables: bool = False, use_separate_dataset: bool = True,
                 in_channels: list = None, out_channels: list = None, spatial_split: list = None):
        """
        Initialize the ERA2CERRA dataset.

        Args:
            data_dir (str): The directory path where the dataset is located.
            mode (str, optional): The mode of the dataset. Possible values are 'train', 'val', 'test', or 'full'. Defaults to 'train'.
            transform (bool, optional): Whether to apply transformations to the data. Defaults to True.
            downscaling_factor (int, optional): The downscaling factor for generating low resolution data. Defaults to 1.
            cropping (str, optional): The cropping strategy. Possible values are 'random', 'deterministic' or 'augmented' (and 'germany'/'italy'). Defaults to 'random'.
            output_dim (int, optional): The output dimension of the dataset. Defaults to 3 (c,h,w).
            crop_size (int, optional): The size of the cropped images. Defaults to 256.
            dataset_stats (dict, optional): Statistics of the dataset. Defaults to None.
            constant_channels (bool, optional): Whether to use constant channels. Defaults to False.
            lat_lon_const_channels (bool, optional): Whether to also use latitude and longitude as constant channels. Defaults to False.
            masking_prob (float, optional): The probability of applying masking to the data. Defaults to 0.5.
            masking_size_ratio (int, optional): The ratio of the masking size to the crop size. Defaults to 16.
            return_era_original (bool, optional): If true, dataset also returns the unchanged original era5 data, in addition to the subsampled era5 data and the cerra target. Defaults to False.
            return_offset (bool, optional): If true, dataset also returns the offset of the crop. Defaults to False.
            use_separate_dataset (bool, optional): Whether to use the dataset versions using seperate u10/v10. Defaults to False.
            spatial_split (List[int], optional): A list of patch indices to include. Defaults to None.
            in_channels (list, optional): The channels to include in the input. Defaults to None.
            out_channels (list, optional): The channels to include in the output. Defaults to None.
        """
        
        self.data_dir = data_dir
        self.cerra_data_dir = os.path.join(data_dir, 'CERRA', 'preprocessed', mode)

        if use_added_variables:
            print("Using ERA5 dataset with added variables")
            self.era5_data_dir = os.path.join(data_dir, 'ERA5', 'added_variables', mode)
        elif use_separate_dataset:
            print("Using ERA5 and CERRA dataset with separate u10/v10")
            self.era5_data_dir = os.path.join(data_dir, 'ERA5', 'preprocessed_separate', mode)
            self.cerra_data_dir = os.path.join(data_dir, 'CERRA', 'preprocessed_separate', mode)
        else:
            self.era5_data_dir = os.path.join(data_dir, 'ERA5', 'preprocessed', mode)

        # Generate low resolution data
        self.downscaling_factor = downscaling_factor
        self.filter_type = 'mean'
        self.output_size = 'small'

        self.crop_size_era = crop_size_era
        self.crop_size_cerra = crop_size_cerra
        self.original_shape_cerra = None
        self.original_shape_era = None

        self.transform = transform
        self.standarized = standarized
        self.cropping = cropping

        self.use_added_variables_dataset = use_added_variables
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels is None:
            if self.use_added_variables_dataset:
                self.in_channels = list(range(8))
            else:
                self.in_channels = list(range(6))
        else:
            print(f"Returning the following channels as LR-input: {self.in_channels}")
            assert len(self.in_channels) <= 8, "Too many input channels"

        if self.out_channels is None:
            self.out_channels = list(range(1))
        else:
            print(f"Returning only the following channels as targets: {self.out_channels}")
            assert len(self.out_channels) <= 1, "Too many output channels"

        # if self.downscaling_factor == 1 and return_era_original:
        #     warnings.warn("Downscaling factor is 1, but return_era_original is set to True. This will not have any effect.")
            

        #Masking parameters
        self.constant_channels = constant_channels
        self.lat_lon_const_channels = lat_lon_const_channels
        self.masking_prob = masking_prob
        self.masking_size = self.crop_size_cerra // masking_size_ratio

        self.spatial_split = spatial_split

        # The constrained downscaling project requires a 4D output, while the standard is 3D
        self.output_dim = output_dim
        self.return_era_original = return_era_original
        self.returns_cerra = True
        self.return_offset = return_offset
        
        # Filter files by extension
        self.cerra_files = glob.glob(self.cerra_data_dir + "/**/*.nc", recursive=True)
        self.era5_files = glob.glob(self.era5_data_dir + "/**/*.nc", recursive=True)

        # Filter files by year
        self.cerra_files = sorted([file for file in self.cerra_files])
        self.era5_files = sorted([file for file in self.era5_files])

        assert len(self.cerra_files) == len(self.era5_files), "Number of CERRA and ERA5 files do not match"
        print(f"Number of Files {len(self.cerra_files)}")

        self.file_lengths = []

        for file in self.cerra_files:
            with xr.open_dataset(file) as ds:
                self.file_lengths.append(ds.time.shape[0])

        for idx, file in enumerate(self.era5_files):
            with xr.open_dataset(file) as ds:
                assert self.file_lengths[idx] == ds.time.shape[0], f"File {file} has different number of samples"
        
        assert len(self.cerra_files) > 0, "No files found in the dataset directory"

        self.cum_file_length = np.cumsum(self.file_lengths)

        self.number_of_crops_per_sample_x_cerra, self.number_of_crops_per_sample_y_cerra, self.original_shape_cerra = self.determine_number_of_crops(files=self.cerra_files, crop_size=self.crop_size_cerra)
        self.number_of_crops_per_sample_cerra = self.number_of_crops_per_sample_x_cerra * self.number_of_crops_per_sample_y_cerra
        self.number_of_crops_per_sample_x_era, self.number_of_crops_per_sample_y_era, self.original_shape_era = self.determine_number_of_crops(files=self.era5_files, crop_size=self.crop_size_era)
        self.number_of_crops_per_sample_era = self.number_of_crops_per_sample_x_era * self.number_of_crops_per_sample_y_era

        assert self.number_of_crops_per_sample_x_cerra == self.number_of_crops_per_sample_x_era, "Different number of x crops"
        assert self.number_of_crops_per_sample_y_cerra == self.number_of_crops_per_sample_y_era, "Different number of y crops"
        assert self.number_of_crops_per_sample_cerra == self.number_of_crops_per_sample_era, "Different number of total crops"

        self.number_of_crops_per_sample_x, self.number_of_crops_per_sample_y, self.number_of_crops_per_sample = self.number_of_crops_per_sample_x_cerra, self.number_of_crops_per_sample_y_cerra, self.number_of_crops_per_sample_cerra

        if (dataset_stats_cerra is None) and (dataset_stats_era is None):
            self.variable_mean_cerra, self.variable_std_cerra, self.variable_min_cerra, self.variable_max_cerra = read_stats(os.path.join(self.cerra_data_dir, '..'))
            self.variable_mean_era, self.variable_std_era, self.variable_min_era, self.variable_max_era = read_stats(os.path.join(self.era5_data_dir, '..'))
        else:
            self.variable_mean_cerra = dataset_stats_cerra['variable_mean_cerra']
            self.variable_std_cerra = dataset_stats_cerra['variable_std_cerra']
            self.variable_min_cerra = dataset_stats_cerra['variable_min_cerra']
            self.variable_max_cerra = dataset_stats_cerra['variable_max_cerra']

            self.variable_mean_era = dataset_stats_era['variable_mean_era']
            self.variable_std_era = dataset_stats_era['variable_std_era']
            self.variable_min_era = dataset_stats_era['variable_min_era']
            self.variable_max_era = dataset_stats_era['variable_max_era']

        self.num_cerra_channels = len(self.variable_mean_cerra)
        self.num_era_channels = len(self.variable_mean_era)
    
        self.variable_mean_cerra = torch.tensor(self.variable_mean_cerra, dtype=torch.float32).reshape((-1,1,1))
        self.variable_std_cerra = torch.tensor(self.variable_std_cerra, dtype=torch.float32).reshape((-1,1,1))

        self.variable_mean_era = torch.tensor(self.variable_mean_era, dtype=torch.float32).reshape((-1,1,1))
        self.variable_std_era = torch.tensor(self.variable_std_era, dtype=torch.float32).reshape((-1,1,1))

        if self.transform:
            if self.spatial_split is not None:
                self.file_lengths = [self.file_lengths[i] * len(self.spatial_split) for i in range(len(self.file_lengths))]
                self.cum_file_length = np.cumsum(self.file_lengths)
                self.number_of_crops_per_sample = len(self.spatial_split)
                print(f"Dataset running in spatial split mode. Number of crops per sample: {self.number_of_crops_per_sample}. Total length: {len(self)}")
            elif cropping == 'deterministic':
                self.file_lengths = [self.file_lengths[i] * self.number_of_crops_per_sample for i in range(len(self.file_lengths))]
                self.cum_file_length = np.cumsum(self.file_lengths)
                print(f"Dataset running in deterministic cropping mode. Number of crops per sample: {self.number_of_crops_per_sample}. Total length: {len(self)}")
            elif cropping == 'germany':
                self.number_of_crops_per_sample = 1
                print("Returns only a patch centered on Germany")

        if self.constant_channels:
            self.constant = get_constant_data(self.cerra_data_dir)
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

        if self.transform:
            if self.spatial_split is not None:
                local_idx = local_idx // len(self.spatial_split)
                crop_idx = self.spatial_split[idx % len(self.spatial_split)]

            elif self.cropping == 'deterministic' or self.cropping == 'augmented':
                local_idx = local_idx // self.number_of_crops_per_sample
                crop_idx = idx % self.number_of_crops_per_sample

        cerra_file = self.cerra_files[file_idx]
        era_file = self.era5_files[file_idx]
        
        # Read the file
        if cerra_file.endswith('.nc'):
            with xr.open_dataset(cerra_file) as ds:

                if self.original_shape_cerra is None:
                    self.original_shape_cerra = torch.tensor(ds[:, local_idx].to_array().values, dtype=torch.float32).shape
                
                if self.transform:
                    if self.cropping == 'deterministic':
                        y_pos_cerra = (crop_idx // self.number_of_crops_per_sample_x) * self.crop_size_cerra
                        x_pos_cerra = (crop_idx % self.number_of_crops_per_sample_x) * self.crop_size_cerra
                    elif self.cropping == 'germany':
                        y_pos_cerra = 400
                        x_pos_cerra = 440
                    else:
                        x_pos = random.randint(0, self.original_shape[1] - self.crop_size_cerra)
                        y_pos = random.randint(0, self.original_shape[2] - self.crop_size_cerra)

                cerra_data = torch.tensor(ds.to_array().values[:, local_idx], dtype=torch.float32)
                
                if self.standarized:
                    #Normalize the data
                    cerra_data = (cerra_data - self.variable_mean_cerra) / self.variable_std_cerra

                if self.transform:
                    cerra_data =cerra_data[:, y_pos_cerra:y_pos_cerra + self.crop_size_cerra, x_pos_cerra:x_pos_cerra + self.crop_size_cerra]

            if len(cerra_data.shape) == 2:
                cerra_data = cerra_data.unsqueeze(0)
            
        else:
            raise ValueError('Unknown file format')
        
        if era_file.endswith('.nc'):
            with xr.open_dataset(era_file) as ds:
                
                if self.transform:
                    if self.cropping == 'deterministic':
                        y_pos_era = (crop_idx // self.number_of_crops_per_sample_x) * self.crop_size_era
                        x_pos_era = (crop_idx % self.number_of_crops_per_sample_x) * self.crop_size_era
                    elif self.cropping == 'germany':
                        y_pos_era = 400
                        x_pos_era = 440
                    else:
                        x_pos_era = random.randint(0, self.original_shape[1] - self.crop_size_era)
                        y_pos_era = random.randint(0, self.original_shape[2] - self.crop_size_era)
                
                era_data = torch.tensor(ds.to_array().values[:, local_idx, :-1, :-1], dtype=torch.float32)
                
                if self.standarized:
                    #Normalize the data
                    era_data = (era_data - self.variable_mean_era) / self.variable_std_era
                    
                if self.transform:
                    era_data =era_data[:, y_pos_era:y_pos_era + self.crop_size_era, x_pos_era:x_pos_era + self.crop_size_era]

                if self.return_era_original:
                        era_original = era_data
                
                if self.downscaling_factor != 1:
                    era_data = self.compute_low_res(era_data, self.downscaling_factor)

            if len(era_data.shape) == 2:
                era_data = era_data.unsqueeze(0)
        
        if self.constant_channels:

            if self.transform:

                constant = self.constant[:, y_pos_cerra:y_pos_cerra + self.crop_size_cerra, x_pos_cerra:x_pos_cerra + self.crop_size_cerra]
                constant_pooled = self.constant_pooled[:, y_pos_cerra:y_pos_cerra + self.crop_size_cerra, x_pos_cerra:x_pos_cerra + self.crop_size_cerra]

                mask = torch.bernoulli(torch.full((self.crop_size_cerra//self.masking_size,self.crop_size_cerra//self.masking_size), self.masking_prob)).int()
                mask = mask.repeat_interleave(self.masking_size, dim=0).repeat_interleave(self.masking_size, dim=1)
                constant_masked = constant * (1-mask) + mask * constant_pooled

            else:
                constant = self.constant
                constant_masked = self.constant

        # Some files may contain data out of bounds, so let's check for that print a warning, and return a random sample instead
        if cerra_data.min() <= -1e7 or cerra_data.max() >= 1e7 or torch.isnan(cerra_data).any() or torch.isinf(cerra_data).any():
            warnings.warn(f"Data out of bounds in file {cerra_file} at local index {local_idx}, min: {cerra_data.min()}, max: {cerra_data.max()} \n Returning random sample instead.")
            warnings.warn(f"Some Data is out of bounds (too small or too large). Returning random sample instead.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        if era_data.min() <= -1e7 or era_data.max() >= 1e7 or torch.isnan(era_data).any() or torch.isinf(era_data).any():
            warnings.warn(f"Data out of bounds in file {era_file} at local index {local_idx}, min: {era_data.min()}, max: {era_data.max()} \n Returning random sample instead.")
            warnings.warn(f"Some Data is out of bounds (too small or too large). Returning random sample instead.")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.constant_channels:
            era_out = (era_data[self.in_channels], constant_masked)
            cerra_out = (cerra_data[self.out_channels], constant)
        elif self.output_dim == 4:
            era_out = (era_data.unsqueeze(0), torch.zeros_like(era_data.unsqueeze(0)))
            cerra_out = (cerra_data.unsqueeze(0), torch.zeros_like(cerra_data.unsqueeze(0)))
        else:
            era_out = (era_data[self.in_channels], torch.zeros_like(era_data[:2]))
            cerra_out = (cerra_data[self.out_channels], torch.zeros_like(cerra_data[:2]))

        # Posiciones
        if self.transform:
            positions = (y_pos_cerra, x_pos_cerra, y_pos_era, x_pos_era, local_idx)
        else:
            positions = (-1, -1, -1, -1, local_idx)

        # Era original o tensor cero
        if self.return_era_original:
            era_orig_out = era_original[self.in_channels]
        else:
            era_orig_out = torch.zeros_like(era_data[self.in_channels])

        return era_out, cerra_out, positions, era_orig_out
        
    def compute_low_res(self, data, factor=4):
        filter_size = (1, factor, factor)
        if self.filter_type == 'mean':
            filtered_data = ndimage.uniform_filter(data, size=filter_size)
        else:
            raise ValueError('Unknown filter type')
        
        if self.output_size == 'same':
            return torch.tensor(filtered_data, dtype=torch.float32)
        
        return torch.tensor(filtered_data[:, ::factor, ::factor], dtype=torch.float32)
        
   
    def determine_number_of_crops(self, files, crop_size):

        file = files[0]
        
        # Read the file
        if file.endswith('.nc'):
            with xr.open_dataset(file) as ds:
                data = ds.to_array()[0]                
                data = torch.tensor(data.values)

            if len(data.shape) == 2:
                data = data.unsqueeze(0)

        original_shape = data.shape
        
        y_crops = original_shape[1] // crop_size
        x_crops = original_shape[2] // crop_size

        return x_crops, y_crops, original_shape
    

### Utility functions ####

def read_stats(data_dir, len_on_error: int = 20) -> tuple[float, float]:
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
        variable_mean = np.zeros(len_on_error)
        variable_std = np.ones(len_on_error)
        min_val = np.full_like(variable_mean, -1e10) 
        max_val = np.full_like(variable_mean, 1e10) 

    return variable_mean, variable_std, min_val, max_val

def get_constant_data(data_dir):
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

    file_name = "aux_variables.nc"

    if file_name in os.listdir(f'{data_dir}/..'):

        # Abre el archivo con xarray
        with xr.open_dataset(os.path.join(data_dir, '..', file_name)) as ds:
            const = torch.tensor(ds.to_array().values, dtype=torch.float32)
            print(const.shape)
        
        print(f"Loaded constant data from file")

        const = (const - torch.mean(const, dim=(1,2), keepdim=True)) / torch.std(const, dim=(1,2), keepdim=True)
    else:
        raise ValueError("Error reading constant data file.")

    return const
