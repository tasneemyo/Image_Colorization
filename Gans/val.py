import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from skimage.color import rgb2lab
from . import utils
import matplotlib.pyplot as plt

def download_image(url):
    """
    Download an image from a URL and return it as a PIL Image.
    """
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    else:
        print(f"Failed to download image from {url}")
        return None

def preprocess_image(img, size=256):
    """
    Preprocess the image for the model:
    - Resize to the required size.
    - Convert to L*a*b* and extract the L-channel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.resize((size, size), Image.BICUBIC)
    img_array = np.array(img)
    img_lab = rgb2lab(img_array).astype("float32")  # Convert to L*a*b
    L = img_lab[:, :, 0]  # Extract L channel
    L = (L / 50.0) - 1.0  # Normalize L to [-1, 1]
    L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return L_tensor.to(device)

def postprocess_and_display(L, ab, original_img):
    """
    Post-process the model's output:
    - Combine L and predicted ab channels.
    - Convert to RGB.
    - Display original grayscale and colorized images side by side.
    """
    L = L.cpu()
    ab = ab.cpu()
    fake_rgb = utils.lab_to_rgb(L, ab)[0]  # Convert first image in batch to RGB
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Display Original Image
    ax[0].imshow(original_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Display Grayscale (Input)
    ax[1].imshow(L[0, 0], cmap="gray")
    ax[1].set_title("Grayscale Input (L-channel)")
    ax[1].axis("off")

    # Display Colorized (Output)
    ax[2].imshow(fake_rgb)
    ax[2].set_title("Colorized Output")
    ax[2].axis("off")

    # plt.show()

def test_model_with_images(model, image_urls, size=256):
    """
    Test the trained model with a list of image URLs.
    Args:
        model: Trained PyTorch model.
        image_urls: List of image URLs.
        size: Size to which images will be resized.
    """
    model.net_G.eval()  # Set the model to evaluation mode
    for url in image_urls:
        print(f"Processing image from: {url}")
        img = download_image(url)  # Download the image
        if img is None:
            continue
        # print(img.shape)
        L_tensor = preprocess_image(img, size)  # Preprocess the image
        with torch.no_grad():
            fake_ab = model.net_G(L_tensor)  # Predict ab channels
        postprocess_and_display(L_tensor, fake_ab, img)  # Display results

# List of image URLs
image_urls = [
    "https://images.pexels.com/photos/346529/pexels-photo-346529.jpeg?cs=srgb&dl=pexels-bri-schneiter-28802-346529.jpg&fm=jpg",
    "https://media.istockphoto.com/id/517188688/photo/mountain-landscape.jpg?s=612x612&w=0&k=20&c=A63koPKaCyIwQWOTFBRWXj_PwCrR4cEoOw2S9Q7yVl8=",

]


def test_model_with_metrics(model, val_dl, num_samples=5):
    """
    Test the model on the test set, visualize results, and calculate PSNR/SSIM.
    Args:
        model: Trained PyTorch model.
        val_dl: DataLoader for the validation/test set.
        num_samples: Number of samples to evaluate and visualize.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.net_G.eval()

    psnr_values = []
    ssim_values = []
    samples_shown = 0

    for data in val_dl:
        if samples_shown >= num_samples:
            break

        # Extract inputs and ground truth
        L = data['L'].to(device)
        ab_real = data['ab'].to(device)

        with torch.no_grad():
            # Predict ab channels
            ab_fake = model.net_G(L)

        # Convert to RGB
        L_cpu = L.cpu()
        ab_real_cpu = ab_real.cpu()
        ab_fake_cpu = ab_fake.cpu()

        real_images = utils.lab_to_rgb(L_cpu, ab_real_cpu)  # Ground truth
        fake_images = utils.lab_to_rgb(L_cpu, ab_fake_cpu)  # Predicted

        # Calculate PSNR and SSIM
        for i in range(len(real_images)):
            psnr = utils.calculate_psnr(real_images[i], fake_images[i])
            ssim = utils.calculate_ssim(real_images[i], fake_images[i])
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if samples_shown < num_samples:
                # Visualize results
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                ax[0].imshow(L[i][0].cpu(), cmap="gray")
                ax[0].set_title("Grayscale Input (L-channel)")
                ax[0].axis("off")

                ax[1].imshow(fake_images[i])
                ax[1].set_title("Colorized Output")
                ax[1].axis("off")

                ax[2].imshow(real_images[i])
                ax[2].set_title("Ground Truth")
                ax[2].axis("off")

                # plt.show()

                samples_shown += 1

    # Log average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")
    utils.visualize(model,data)
