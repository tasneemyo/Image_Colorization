from . import model
from . import train
import torch
from . import dataset
import numpy as np
import glob
from torchsummary import summary
import os
import argparse
from . import val,utils
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--save_path",type=str,default="/kaggle/working/",required=False)
    parser.add_argument("--save_images",type=bool,default=False,required=False)

    args=parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.MainModel()
    print("Generator Summary:")
    summary(model.net_G, input_size=(1, 256, 256))  # Input: Grayscale image (1 channel)

    print("\nDiscriminator Summary:")
    summary(model.net_D, input_size=(3, 256, 256))
    dataset_root = args.data_path
    train_color_paths=dataset_root+"/train_color/*.*"
    test_color_paths=dataset_root+"/test_color/*.*"
    train_color_paths = glob.glob(train_color_paths)
    test_color_paths  = glob.glob(test_color_paths)

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
    print(val_paths[:2],test_color_paths[:2])
    train_dl = dataset.make_dataloaders(paths=train_paths, split='train')
    val_dl = dataset.make_dataloaders(paths=val_paths, split='val')
    test_dl = dataset.make_dataloaders(paths=test_color_paths, split='test')
    print(val_dl,test_dl)
    print("Train =", len(train_paths), "Val =", len(val_paths))
    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    # print(Ls.shape, abs_.shape)
    # print(len(train_dl), len(val_dl))  
    loss_G,loss_D=[],[]
    print(f"args {args.train} {args.test}")
    if args.train:
        train.train_model(model, train_dl,val_dl,args.epochs,loss_D=loss_D,loss_G=loss_G)
    save_path = args.save_path

# Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the model
    model_file_path = os.path.join(save_path, "Unet_colorization_model.pth")
    torch.save(model.state_dict(), model_file_path)

    print(f"Model saved at {model_file_path}")
    if args.test:
        val.test_model_with_images(model, val.image_urls)
        val.test_model_with_metrics(model, val_dl, num_samples=10)
    