import torch
from torch.utils.data import Dataset
import xarray as xr

def ZScore(x, mean, std, eps=1e-8):
    """
    Apply pixel-wise z-score normalization.
    x: torch.Tensor of shape [C, H, W]
    mean, std: torch.Tensor of shape [C,]
    """
    return (x - mean[:, None, None]) / (std[:, None, None] + eps)

def ZScorePerPixel(x, mean, std, eps=1e-8):
    """
    Apply pixel-wise z-score normalization.
    x: torch.Tensor of shape [C, H, W]
    mean, std: torch.Tensor of shape [C, H, W]
    """
    
    return (x - mean) / (std + eps)

def MinMax(x, min_val, max_val, eps=1e-8):
    """
    Apply pixel-wise min-max normalization.
    x: torch.Tensor of shape [C, H, W]
    min_val, max_val: torch.Tensor of shape [C,]
    """
    return (x - min_val[:, None, None]) / (max_val[:, None, None] - min_val[:, None, None] + eps)

def MinMaxPerPixel(x, min_val, max_val, eps=1e-8):
    """
    Apply pixel-wise min-max normalization.
    x: torch.Tensor of shape [C, H, W]
    min_val, max_val: torch.Tensor of shape [C, H, W]
    """
    return (x - min_val) / (max_val - min_val + eps)


class NetCDFDataset(Dataset):
    def __init__(self, input_static_ds: xr.Dataset,  input_ds_1: xr.Dataset, target_ds: xr.Dataset, high_res_low_res_ratio: float, 
                 static_variables: list =['z', 'population'], input_variables: list = ['pm2p5'], output_variables: list = ['pm2p5_conc'],
                 input_stats = None, target_stats = None, norm_method="none"):

        self.input_ds_1 = input_ds_1
        self.target_ds = target_ds

        self.high_res_low_res_ratio = high_res_low_res_ratio
        self.input_static_ds = input_static_ds
        

        # Validate shape compatibility
        ### The following is for training, in inference it may be a problem if we do not have target data. 
        # assert len(input_ds_1.time) == len(target_ds.time), "Time dimensions must match"

        self.static_vars = static_variables
        self.input_vars = input_variables
        self.output_vars = output_variables
        available_static_vars = [v for v in self.static_vars if v in input_static_ds.data_vars]
        available_input_vars = [v for v in self.input_vars if v in input_ds_1.data_vars]
        available_output_vars = [v for v in self.output_vars if v in target_ds.data_vars]

        self.input_static_data = input_static_ds[available_static_vars].to_array().transpose("variable", "latitude", "longitude")
        self.input_data = input_ds_1[available_input_vars].to_array().transpose("time", "variable", "latitude", "longitude")
        self.target_data = target_ds[available_output_vars].to_array().transpose("time", "variable", "latitude", "longitude")    #.sel(level=0.0).squeeze("level")  # Drop level dim if needed

        self.times = target_ds.time.values
        self.lats = target_ds.latitude.values
        self.lons = target_ds.longitude.values

        # Normalization
        self.norm_method = norm_method
        if norm_method == "z_score" or norm_method == "z_score_per_pixel":
            if input_stats is not None:
                self.input_mean = torch.tensor(input_stats['mean'], dtype=torch.float32)
                self.input_std = torch.tensor(input_stats['std'], dtype=torch.float32)
            else:
                self.input_mean = self.input_std = None 
            if target_stats is not None:
                self.target_mean = torch.tensor(target_stats['mean'], dtype=torch.float32)
                self.target_std = torch.tensor(target_stats['std'], dtype=torch.float32)
            else:
                self.target_mean = self.target_std = None
        elif norm_method == "minmax" or norm_method == "minmax_per_pixel":
            if input_stats is not None:
                self.input_min = torch.tensor(input_stats['min'], dtype=torch.float32)
                self.input_max = torch.tensor(input_stats['max'], dtype=torch.float32)
            else:
                self.input_min = self.input_max = None 
            if target_stats is not None:
                self.target_min = torch.tensor(target_stats['min'], dtype=torch.float32)
                self.target_max = torch.tensor(target_stats['max'], dtype=torch.float32)
            else:
                self.target_min = self.target_max = None
        elif norm_method == "none":
            self.norm_method = "none"
            self.input_mean = self.input_std = None
            self.target_mean = self.target_std = None
            self.input_min = self.input_max = None
            self.target_min = self.target_max = None
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")

    def __len__(self):
        return len(self.input_ds_1.time)

    def __getitem__(self, idx):
        # Lazy load tensors using dask -> numpy -> torch
        x1 = torch.tensor(self.input_data[idx].values, dtype=torch.float32)  # (variables, lat, lon)
        # x2 = torch.full((1,), 1/self.high_res_low_res_ratio, dtype=torch.float32)  # (1,)
        x2 = torch.tensor(self.input_static_data.values, dtype=torch.float32)  # (variables, lat', lon')
        y = torch.tensor(self.target_data[idx].values, dtype=torch.float32)  # (lat', lon')

        # Normalization
        if self.norm_method == "z_score":
            if self.input_mean is not None and self.input_std is not None:
                x1 = ZScore(x1, self.input_mean, self.input_std)
                y = ZScore(y, self.target_mean, self.target_std)
        elif self.norm_method == "z_score_per_pixel":
            if self.input_mean is not None and self.input_std is not None:
                x1 = ZScorePerPixel(x1, self.input_mean, self.input_std)
                y = ZScorePerPixel(y, self.target_mean, self.target_std)
        elif self.norm_method == "minmax":
            if self.input_min is not None and self.input_max is not None:
                x1 = MinMax(x1, self.input_min, self.input_max)
                y = MinMax(y, self.target_min, self.target_max)
        elif self.norm_method == "minmax_per_pixel":
            if self.input_min is not None and self.input_max is not None:
                x1 = MinMaxPerPixel(x1, self.input_min, self.input_max)
                y = MinMaxPerPixel(y, self.target_min, self.target_max)
        elif self.norm_method == "none":
            print("No normalization applied to input data.")

        # Manually remove the first row and column so that the incoming lat and lot are multiples of the target resolution, i.e. (106, 176) -> (105, 175)
        x1 = x1[:, 1:, 1:]


        # High-res coords for this specific sample
        coords = {
            "time": str(self.times[idx]),    # scalar
            "lat": self.lats,                # (lat',)
            "lon": self.lons                 # (lon',)
        }


        return (x1, x2), y, coords