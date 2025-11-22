'# 🌈 Image Colorization Using GANs
*A U-Net Generator with a PatchGAN Discriminator*

This repository implements a **deep learning–based image colorization system** using a **Conditional Generative Adversarial Network (cGAN)**.  
The model learns to predict the color channels (a/b) of an image from its luminance (L-channel) using a U-Net generator and a PatchGAN discriminator.  
The pipeline supports **training, validation, visualization, and inference on external images**.

## 🔍 Overview
- **Model Architecture**
  - U-Net Generator  
  - PatchGAN Discriminator  
- Works in **LAB color space**
- Supports: training, visualization, inference, dataset preprocessing

## 📁 Project Structure
```
Image_Colorization/
├── Gans/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
├── download_data.py
├── main.py
└── README.md
```

## 🧠 Model Architecture

### Generator (U-Net)
- Encoder-decoder with skip connections  
- Outputs ab channels

### Discriminator (PatchGAN)
- Classifies patches instead of whole images  
- More stable GAN training  

### Loss Functions
- GAN loss  
- L1 reconstruction loss  

## 📦 Installation
```
pip install torch torchvision pillow matplotlib numpy opencv-python requests
```

## 🚀 Training
```
python main.py
```

## 🎨 Visualization
Results include input grayscale, predicted color, and ground truth.

## 🖼 Inference
Colorize external images:
```
colorize_image(model, "Unet_colorization_model.pth", img_path)
```

## ✔ Requirements
- Python 3.7+
- PyTorch
- (Optional) CUDA GPU

## 📝 License
MIT License
'