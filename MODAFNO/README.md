# 🌍 Spatial ModAFNO for Pollutant Downscaling
This repository contains the code for implementing the Spatial ModAFNO architecture, a novel approach adapted for the downscaling of environmental variables, specifically pollutant concentrations.

The model is designed to perform super-resolution (downscaling) from low-resolution to high-resolution products. In this work, the architecture downscales the PM2.5 concentrations from the CAMS Global product (low-resolution) to the CAMS Europe product (high-resolution), but it can be applied to any other geophysical variable and product.

The core contribution is the introduction of Spatial Adaptation of ModAFNO (Spatial ModAFNO), which implements a novel Spatial Modulation technique within the Adaptive Fourier Neural Operator (AFNO) network. This modulation allows for the effective incorporation of static features (like orography and population density) into the deep learning model. Beyond the original ModAFNO architecture (designed for temporal interpolation), this repository provides the first downscaling-oriented adaptation.


## 📂 Repository Structure

```
├── config/
│   └── spatial_downscaling.yaml        # Main configuration file for training & eval
├── data/
│   ├── train/                          # Training datasets
│   ├── val/                            # Validation datasets
│   ├── test/                           # Testing datasets
│   ├── stats/                          # Normalization statistics
│   ├── statics/                        # Static features (e.g., orography, population)
│   └── data_factory_static.py          # Data loader for static + dynamic variables
├── model/
│   ├── spatial_modafno.py              # Core Spatial ModAFNO architecture
│   ├── modembed_spatial.py             # Spatial embedding layers
│   ├── normalization.py                # Normalization utilities
│   ├── loss.py                         # Loss functions
│   ├── checkpoint.py                   # Saving/loading models
│   └── train.py                        # Model training utilities
├── train_modafno_downscaling.py        # Main training script
├── eval_modafno_downscaling.py         # Run inference on new LR inputs
├── compute_metrics.py                  # Compute metrics on inference maps
├── requirements.txt                    # Package list (includes physics-nemo)
└── README.md
```

## 🚀 Getting Started
Follow these steps to set up the environment and run the code.

### Prerequisites
You need a Python environment and NVIDIA drivers for the GPU-accelerated dependencies.

### Installation
1. Clone the repository:

```bash
git clone [Your Repository URL]
cd [Your Repository Name]
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies: The installation process, particularly for nvidia-physicsnemo, may take some time.

```bash
pip install -r requirements.txt
```

## 🧠 Training the Model
The model is trained using the `train_modafno_downscaling.py` script, with configurations loaded from the YAML file in the `config/` directory.

### Configuration (`spatial_downscaling.yaml`)
Before training, you must configure the following settings in `config/spatial_downscaling.yaml`:

* Model Hyperparameters: Adjust parameters like layer count, hidden dimensions, etc.

* Modulation Flag: Set whether to use the novel spatial modulation with static features (orography, population density) to modulate the main network.

* Data Paths: Define the correct paths for the train, val, and test datasets within the `data/` folder.

* Variables: Specify the input (low-res pollutant) and output (high-res pollutant) variables.

* Normalization: Configure normalization settings.

* MLflow Info: Set up configuration for tracking training runs using MLflow.

### Execution
Run the training script from the root directory:

```Bash
python train_modafno_downscaling.py
```

## 📉 Evaluation and Inference
Once the model is trained, you can perform inference to obtain the downscaled high-resolution maps.

### Execution
Run the evaluation script:

```Bash
python eval_modafno_downscaling.py
```

The resulting high-resolution map for the target pollutant (e.g., PM2.5) will be saved in the following location:

```
inference_results/maps/
```

## 📊 Computing Metrics
To quantify the model's performance, you can compute metrics by comparing the inferred high-resolution map with a ground truth map.

1. Update Configuration: Ensure the `spatial_downscaling.yaml` file has the correct paths set to both the inferred map (saved in `inference_results/maps/`) and the ground truth map.

2. Run Metrics Script:

```Bash
python compute_metrics.py
```

This will calculate and report the relevant performance metrics in the metrics folder.