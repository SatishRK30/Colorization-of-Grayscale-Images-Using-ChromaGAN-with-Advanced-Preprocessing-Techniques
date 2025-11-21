# 📘 Colorization of Grayscale Images Using ChromaGAN with Advanced Preprocessing Techniques

This project explores **automatic colorization of grayscale images** using a ChromaGAN-based neural network pipeline enhanced with **advanced preprocessing techniques** such as bilateral filtering, Gaussian blur, CLAHE, guided filtering, and horizontal flipping.

The project compares multiple preprocessing workflows using **PSNR** and **SSIM** to evaluate structural fidelity and colorization quality.

The full research paper is included in the repository:

📄 **Report:**
`Colorization_of_Grayscale_Images_Using_ChromaGAN_with_Advanced_Preprocessing_Techniques.pdf`

---

## 📑 Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Pipeline Overview](#pipeline-overview)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Usage](#usage)

  * [1. Preprocessing](#1-preprocessing)
  * [2. Augmentation](#2-augmentation)
  * [3. ChromaGAN Inference](#3-chromagan-inference)
  * [4. Evaluation](#4-evaluation)
* [Results](#results)
* [Future Work](#future-work)

---

## 🔍 Introduction

ChromaGAN is a GAN-based model designed to colorize grayscale images by learning **semantic class distributions** within images.
However, vintage images often suffer from **noise, fading, and low contrast**, reducing the quality of colorization.

To address this, our project integrates **image preprocessing techniques** that enhance structure, reduce noise, and preserve edges before feeding images to ChromaGAN.

We evaluate:

* Bilateral Filter
* Gaussian Blur
* CLAHE
* Guided Filter
* Combined Gaussian + Bilateral filtering
* Horizontal flipping (augmentation)

PSNR and SSIM metrics are used for analysis.

---

## ✨ Features

* Fully implemented preprocessing pipeline
* ChromaGAN-based generator & PatchGAN discriminator
* LAB color space post-processing
* PSNR & SSIM evaluation scripts
* Modular source code (`src/` folder)
* Ready-to-use inference pipeline for single images
* Complete research report included

---

## 🔧 Pipeline Overview

```
Grayscale Image
      ↓
Preprocessing (Gaussian / Bilateral / CLAHE / Guided)
      ↓
Augmentation (Horizontal Flip)
      ↓
ChromaGAN (Generator Network)
      ↓
ab-channel prediction + LAB → BGR conversion
      ↓
Colorized Output Image
      ↓
Metric Evaluation (PSNR, SSIM)
```

---

## 📂 Repository Structure

```
.
├── sample_images/
│   ├── color/
│   ├── grayscale/
│
├── src/
│   ├── preprocessing/
│   ├── augmentation/
│   ├── chromagan/
│   ├── evaluation/
│   └── main.py
│
├── Colorization_of_Grayscale_Images_Using_ChromaGAN_with_Advanced_Preprocessing_Techniques (1).pdf
├── README.md
└── requirements.txt (your choice to add)
```

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/KaushikNITK/Colorization_of_Grayscale_Images_Using_ChromaGAN_with_Advanced_Preprocessing_Techniques.git
cd Colorization_of_Grayscale_Images_Using_ChromaGAN_with_Advanced_Preprocessing_Techniques
```

### 2. Create environment (recommended)

```bash
conda create -n chromagan python=3.7
conda activate chromagan
```

### 3. Install dependencies

TensorFlow 1.x (as required by ChromaGAN code):

```bash
pip install tensorflow==1.15 opencv-python numpy scikit-image matplotlib seaborn
```

---

## 🚀 Usage

### 1. Preprocessing

Run any filter independently, e.g.:

```python
from src.preprocessing.bilateral_filter import apply_bilateral
img = apply_bilateral(gray_img)
```

Or apply all recommended filters:

```python
from src.preprocessing.apply_all_filters import process_all
processed = process_all(gray_img)
```

---

### 2. Augmentation

```python
from src.augmentation.horizontal_flip import flip_horizontal
flipped = flip_horizontal(processed_img)
```

---

### 3. ChromaGAN Inference

Colorize any grayscale image:

```python
from src.chromagan.inference import colorize_image

output = colorize_image("input.png", ckpt_path="checkpoints/model.ckpt")
```

Or save directly:

```python
from src.chromagan.inference import colorize_and_save

colorize_and_save("input.png", "checkpoints/model.ckpt", "results/output.png")
```

---

### 4. Evaluation

```python
from src.evaluation.comparison import evaluate

metrics = evaluate(original_rgb, generated_rgb)
print(metrics["psnr"], metrics["ssim"])
```

---

## 📊 Results

| Technique                   | PSNR  | SSIM   |
| --------------------------- | ----- | ------ |
| Baseline (ChromaGAN)        | 30.16 | 0.9319 |
| Bilateral Filter            | 30.11 | 0.9297 |
| Guided Filter               | 29.42 | 0.8205 |
| CLAHE                       | 28.36 | 0.7598 |
| Gaussian + Bilateral        | 29.61 | 0.8511 |
| Flip + Gaussian + Bilateral | 29.60 | 0.8498 |

Observations:

* **Bilateral filter** preserved structural detail best
* **CLAHE** introduced artifacts leading to lower SSIM
* **Gaussian + Bilateral** improved stability and edge clarity
* **Horizontal flipping** improved generalization

---

## 🔮 Future Work

Potential enhancements:

* Train full ChromaGAN model on your custom vintage dataset
* Move pipeline to TensorFlow 2.x or PyTorch
* Use frequency-domain processing (DFT, Wavelet)
* Adaptive filtering based on noise estimation
* Explore diffusion-based colorization models

---
