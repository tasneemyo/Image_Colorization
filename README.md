# Image Colorization Using GANs

*A U-Net Generator with a PatchGAN Discriminator*

This repository implements a **deep learning--based image colorization
system** using a **Conditional Generative Adversarial Network (cGAN)**.\
The model learns to predict the color channels (a/b) of an image from
its luminance (L-channel) using a U-Net generator and a PatchGAN
discriminator.\
The pipeline supports **training, validation, visualization, and
inference on external images**.

## 🔍 Overview

-   **Model Architecture**
    -   U-Net Generator\
    -   PatchGAN Discriminator\
-   Works in **LAB color space**
-   Supports: training, visualization, inference, dataset preprocessing

## 📁 Project Structure

    Image_Colorization/
    ├── Gans/
    │   ├── data.py
    │   ├── model.py
    │   ├── train.py
    │   ├── utils.py
    ├── gui.py
    ├── main.py
    └── README.md

## 🧠 Model Architecture

### Generator (U-Net)

-   Encoder-decoder with skip connections\
-   Outputs ab channels

### Discriminator (PatchGAN)

-   Classifies patches instead of whole images\
-   More stable GAN training

### Loss Functions

-   GAN loss\
-   L1 reconstruction loss

## 📦 Installation

    pip install torch torchvision matplotlib numpy opencv-python requests

## 🚀 Training

Run main script with arguments:

    python main.py --epochs <num_epochs> --data_path <path> --train True --save_path <output_path> --save_images True

## 🧪 Testing

    python main.py --test True --data_path <path_to_test_data>

## 🎨 Visualization

    python main.py

## 🖥 Command Line Arguments

    --epochs        (int)   Number of training epochs
    --data_path     (str)   Path to dataset
    --train         (bool)  Enable training mode
    --test          (bool)  Enable testing/inference mode
    --save_path     (str)   Directory to save model/logs (default: /kaggle/working/)
    --save_images   (bool)  Save generated images (default: False)

## ✔ Requirements

-   Python 3.7+
-   PyTorch
-   numpy
