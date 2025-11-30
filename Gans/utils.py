import torch
import numpy as np
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images.
    Args:
        img1: Ground truth image (RGB, range [0, 1]).
        img2: Predicted image (RGB, range [0, 1]).
    Returns:
        PSNR value.
    """
    psnr_value = peak_signal_noise_ratio(img1, img2, data_range=1.0)
    return psnr_value

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.
    Args:
        img1: Ground truth image (RGB, range [0, 1]).
        img2: Predicted image (RGB, range [0, 1]).
    Returns:
        SSIM value.
    """
    # Explicitly set win_size and handle multichannel
    ssim_value = structural_similarity(
        img1, img2, win_size=7, channel_axis=-1, data_range=1.0
    )
    return ssim_value

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
def init_weights(net, init='norm', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model




def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    
    for i in range(5):
        # L channel
        axes[0, i].imshow(L[i][0].cpu(), cmap='gray')
        axes[0, i].axis("off")
        # Fake color
        axes[1, i].imshow(fake_imgs[i])
        axes[1, i].axis("off")
        # Real color
        axes[2, i].imshow(real_imgs[i])
        axes[2, i].axis("off")
    
    plt.tight_layout()
    
    if save:
        fig.savefig(f"colorization_{int(time.time())}.png")
    
    # --- This ensures Kaggle actually renders the figure ---
    # plt.show(block=True)
    # plt.close(fig)  # Close figure to prevent memory issues

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")