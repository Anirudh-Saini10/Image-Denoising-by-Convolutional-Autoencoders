
# Convolutional Autoencoder for Image Denoising

A PyTorch implementation of a Convolutional Autoencoder (CAE) trained to remove Gaussian noise from images using the MNIST dataset.

## Overview

This project trains a CAE to reconstruct clean images from noisy inputs. Gaussian noise (σ=0.5) is added to MNIST digits during training, and the model learns to recover the original image.

## Model Architecture

**Encoder**
- Conv2d(1 → 32, kernel=3, padding=1) + ReLU + MaxPool2d(2×2)
- Conv2d(32 → 64, kernel=3, padding=1) + ReLU + MaxPool2d(2×2)

**Decoder**
- ConvTranspose2d(64 → 32, kernel=2, stride=2) + ReLU
- ConvTranspose2d(32 → 1, kernel=2, stride=2) + Sigmoid

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST |
| Noise Type | Gaussian (σ = 0.5) |
| Optimizer | Adam (lr = 1e-3) |
| Loss Function | MSE Loss |
| Epochs | 20 |
| Final Loss | 0.0880 |

## Results

The model successfully reconstructs clean digit images from heavily noised inputs (σ=0.5), achieving a final MSE loss of **0.0880** over 20 epochs.

## Usage

1. Clone the repo
2. Open `CAE_Denoising.ipynb` in Jupyter or Google Colab
3. Run all cells sequentially

## References

- [Hinton & Salakhutdinov, 2006 - Reducing Dimensionality with Neural Networks](https://www.science.org/doi/10.1126/science.1127647)
- [MNIST Dataset - Yann LeCun](http://yann.lecun.com/exdb/mnist/)
