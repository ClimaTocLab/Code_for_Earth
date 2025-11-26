# ML-Driven Downscaling of Atmospheric Pollution Fields (Code for Earth 2025 – Challenge 21)

This repository contains our complete solution for Challenge 21 – ML-Driven Downscaling of Global Air Pollution Fields in the Code for Earth 2025 program, organized by ECMWF and partners.

Our work focuses on learning high-resolution pollution fields (specifically PM2.5 concentrations) from coarse global simulations, using CAMS Global as input data and CAMS Europe as target high-resolution fields.

The project presents a three-model architecture suite, a causal inference analysis framework, and comprehensive evaluation across multiple regions and real observations.

Each model folder contains its own training, inference, dataset structure and configuration instructions.


## 🌍 Project Overview

The aim is to build ML models capable of:


✔️ Downscaling PM2.5 pollution fields from CAMS Global (0.4°) → CAMS Europe (0.1°) 

✔️ Leveraging dynamic predictors: wind speed, temperature, etc. 

✔️ Incorporating static geospatial features: orography, population density 

✔️ Validating results against independent observations across Europe and the US

✔️ Testing generalization beyond the training domain

✔️ Understanding feature importance through causal analysis and interpretability tools


## 🧠 Our Three-Model Solution

We developed three complementary architectures to address the downscaling problem, each with unique strengths in accuracy, efficiency, and generalization.

### 1️⃣ Baseline U-Net

A classical fully convolutional architecture used as a baseline.

Highlights:

* Simple and robust

* Provides consistent downscaling

* Best pattern reproduction of CAMS Global outside training domain

Used as a reference for evaluating the other deep-learning operators.

📂 Code: `Baseline_U-net/`

### 2️⃣ Spatial ModAFNO

We extend the Adaptive Fourier Neural Operator (AFNO) with:

✔️ Spatial Modulation

The AFNO latent fields are modulated using static geospatial features:

* Orography

* Population density


✔️ Downscaling Adaptation

Originally temporal, AFNO is modified into a purely spatial operator capable of handling low→high resolution mapping.

Strengths:

* Second-best performance overall

* Most computationally efficient model

* Best accuracy over the United States as compared with real observations


📂 Code: `MODAFNO/`

### 3️⃣ SwinFIR Transformer

A Swin-Transformer variant adapted for geospatial downscaling using FIR-based convolutional mixing.

Strengths:

* Best quantitative performance on the eval set (Europe)

* Superior spatial detail reconstruction

📂 Code: `Swin_v2_Visual_Transformer/`

## 📊 Dataset and Training Setup
Low-resolution input

+ CAMS Global data subset covering Europe

* PM2.5 concentrations

* Dynamic variables used as predictors (wind speed, temperature, etc.)

High-resolution target

* CAMS Europe PM2.5

* 0.1° resolution

Static fields

* Orography

* Population density

Training region

* Model trained on the Europe region

* Evaluated both in-domain (Europe) and out-of-domain (USA)

## 🧪 Evaluation and Cross-Region Testing

After training on European data:

1. Comparison against CAMS Europe

* Swin Transformer performs best

* Spatial ModAFNO close behind

* Both outperform U-Net

* All significantly outperform (traditional) linear interpolation of CAMS Global

2. Comparison to real observations in Europe

Performance comparable to CAMS Europe, validating model skill.

3. Testing on the United States

* Using real US PM2.5 observations, we test generalization:

* Spatial ModAFNO → best accuracy vs real observations

* U-Net → best “pattern preservation” w.r.t CAMS Global

* Swin Transformer → good but more domain sensitive

🔍 Causal Inference Analysis

We supplemented the modeling with a causal analysis module to:

* Understand structural dependencies between PM2.5 and predictors

* Identify the most meaningful drivers

Tools included:

* Causal graph estimation via PCMCI method


📂 Code: `Causal_Inference_for_Variables/`

## 🧠 Interpretability

We applied model explainability tools, including:

* Feature ablation techniques

* Gradient-based saliency

* Integrated gradients

* SHAP values

These tools were used primarily on the SwinFIR Transformer.
Spatial ModAFNO interpretability is ongoing work.

## 🚀 How to Train and Run the Models

Each model is fully self-contained.

To use a model:

1. Enter its associated folder

2. Follow the instructions in its README.md

3. Prepare the data following the folder’s structure

4. Use the training script to train

5. Run inference

6. Compute metrics

## 📚 Citation

A preprint will appear soon for citations, keep tuned!

## 👥 Team and Acknowledgements

This work was developed by the Code for Earth Team for Challenge 21.
We thank ECMWF for providing access to CAMS data and compute infrastructure.
We also thank our mentors Johannes Bieser, Johannes Flemming, Paula Harder, Martin Ramacher and Miha Razinger for all their feedback and support.   

## ❗ Contact

If you use this repo or want to collaborate, contact:

Kevin Monsálvez-Pozo, Marcos Martínez-Roig, Francisco Granell, Víctor Galván Fraile or Nuria P. Plaza-Martín. 
