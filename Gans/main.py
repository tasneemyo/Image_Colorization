from model import MainModel
from train import train_model
import torch
from dataset import make_dataloaders
import numpy as np
import glob
from torchsummary import summary
import os
from val import test_model_with_images,image_urls,test_model_with_metrics
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MainModel()
    print("Generator Summary:")
    summary(model.net_G, input_size=(1, 256, 256))  # Input: Grayscale image (1 channel)

    print("\nDiscriminator Summary:")
    summary(model.net_D, input_size=(3, 256, 256))
    dataset_root = "/kaggle/input/image-colorization-dataset/data" 
    train_color_paths = glob.glob("/kaggle/input/image-colorization-dataset/data/train_color/*.*")
    test_color_paths  = glob.glob("/kaggle/input/image-colorization-dataset/data/test_color/*.*")

    print("Found:", len(train_color_paths), "train_color images")
    print("Found:", len(test_color_paths), "test_color images")
    paths_subset = np.random.choice(train_color_paths, len(train_color_paths), replace=False)

    rand_idxs = np.random.permutation(len(paths_subset))

    # 80% train, 20% val (same as your original idea)
    split_point = int(len(paths_subset) * 0.8)

    train_idxs = rand_idxs[:split_point]
    val_idxs   = rand_idxs[split_point:]
    train_paths = paths_subset[train_idxs]
    val_paths   = paths_subset[val_idxs]
    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    print("Train =", len(train_paths), "Val =", len(val_paths))
    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    print(Ls.shape, abs_.shape)
    print(len(train_dl), len(val_dl))  
    train_model(model, train_dl,val_dl,10)
    save_path = "/kaggle/working/"

# Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the model
    model_file_path = os.path.join(save_path, "Unet_colorization_model.pth")
    torch.save(model.state_dict(), model_file_path)

    print(f"Model saved at {model_file_path}")
    test_model_with_images(model, image_urls)
    test_model_with_metrics(model, val_dl, num_samples=10)
    