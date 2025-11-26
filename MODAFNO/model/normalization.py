import numpy as np
import xarray as xr

def compute_stats(ds: xr.Dataset,
                  variables: list,
                  method: str):
    """
    Compute normalization statistics on ds[variables].

    Returns a dict whose keys depend on `method`:
      - "z_score":           {"mean": (n_vars,), "std": (n_vars,)}
      - "z_score_per_pixel":{"mean": (n_vars, H, W), "std": (n_vars, H, W)}
      - "minmax":            {"min": (n_vars,),  "max": (n_vars,)}
      - "minmax_per_pixel":  {"min": (n_vars,H,W),"max":(n_vars,H,W)}
      - "none":              {}
    """
    arr = ds[variables] \
          .to_array() \
          .transpose("variable","time","latitude","longitude")
    stats = {}
    if method == "z_score":
        stats["mean"] = arr.mean(dim=("time","latitude","longitude")).values
        stats["std"]  = arr.std(dim=("time","latitude","longitude")).values

    elif method == "z_score_per_pixel":
        stats["mean"] = arr.mean(dim="time").values
        stats["std"]  = arr.std(dim="time").values

    elif method == "minmax":
        stats["min"] = arr.min(dim=("time","latitude","longitude")).values
        stats["max"] = arr.max(dim=("time","latitude","longitude")).values

    elif method == "minmax_per_pixel":
        stats["min"] = arr.min(dim="time").values
        stats["max"] = arr.max(dim="time").values

    elif method == "none":
        return {}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return stats


def save_stats(stats: dict, path: str):
    """Save all arrays in stats to a compressed NPZ."""
    np.savez_compressed(path, **stats)