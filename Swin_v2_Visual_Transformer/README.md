# 🌍 Spatial Swin Visual Transformer for Pollutant Downscaling

This repository contains the code for implementing the Spatial Swin Visual Transformer architecture, based on SwinFIR model and adapted for the downscaling of environmental variables, specifically pollutant concentrations.

The model is designed to perform super-resolution (downscaling) from low-resolution to high-resolution products. In this work, the architecture downscales the PM2.5 concentrations from the CAMS Global product (low-resolution) to the CAMS Europe product (high-resolution), but it can be applied to any other geophysical variable and product.

## 📂 Repository Structure

```
├── config/
│   └── test.yaml        # Main configuration file for training & eval
├── data/
│   ├── CERRA/preprocessed_separate     # High Resolution Data
│   ├── ERA5/preprocessed_separate     # Low Resolution Data
├── dataset_utils/
│   ├── ERA2CERRA.py		        # Dataset Class
│   ├── LightningDataModules.py         # Dataset Class with PyTorch Lightning Functionalities
├── metrics/
│   ├── metrics.py		        # Metrics used during training
├── models/
│   ├── swin_fir	                # Core of Swin Visual Transformer architecture
│   ├── lightning_model_template.py     # Model Class with PyTorch Lightning Functionalities
├── Tests/Interpretability_study	# Jupiter Notebooks for evaluation
├── environment.yml		        # Conda environment
├── run_train_valid.sh		        # Bash script for multiple runs
├── train_cli.py                  	# Main script for PyTorch Lightning
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

2. Install and activate the conda environment:

```bash
conda activate
conda env create -f environment.yml
conda activate downscaling
```

## 🧠 Training the Model

The model is trained using the `train_cli.py` script, with configurations loaded from the YAML file in the `config/` directory.

### Configuration (`spatial_downscaling.yaml`)

Before training, you must configure the following settings in `config/spatial_downscaling.yaml`:

* Model Hyperparameters: Adjust parameters like layer count, hidden dimensions, etc.
* Data Paths: Define the correct paths for the train, val, and test datasets within the `data/` folder.
* Variables: Specify the input (low-res pollutant) and output (high-res pollutant) variables.
* Normalization: Configure normalization settings.
* Weight&Bias Info: Set up configuration for tracking training runs using Weight&Bias.

### Execution

Run the training script from the root directory:

```Bash
python train_cli.py fit --config=configs/test.yaml
```

## 📉 Evaluation and Inference

Once the model is trained, you can perform inference to obtain the downscaled high-resolution maps.

### Execution

Run the evaluation script:

```Bash
python3 train_cli.py test --config=configs/test.yaml --ckpt_path=path/to/checkpoint.ckpt
```

The resulting high-resolution map for the target pollutant (e.g., PM2.5) will be saved in the following location:

```
model_name/output/data/test
```
