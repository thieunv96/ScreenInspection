import cv2
import numpy as np
from downloader import *
from utils import *
from model import *
from glob import glob
import os

def download_dataset():
    url = 'https://drive.google.com/u/0/uc?id=1hKLC6rHtG9yv5mcFc_5Q-5-10Wo1zP8o&export=download'
    output = 'dataset.7z'
    download(url, output)


def make_mask():
    files = glob('mask/*.png')
    id = 0
    save_path = "sub_mask"
    
    
    for f in files:
        image = cv2.imread(f, 1)
        images, boxs = split_image(image)
        for im in images:
            cv2.imwrite(f"{save_path}/{id}.png", im)
            id += 1

def make_dataset():
    files = glob('dataset/pass/*.png')
    id = 0
    save_path = "sub_dataset"
    if not os.path.exists(save_path):
        os.makedirs(f"{save_path}")
    for f in files:
        image = cv2.imread(f, 1)
        images, boxs = split_image(image)
        for im in images:
            cv2.imwrite(f"{save_path}/{id}.png", im)
            id += 1


def train():
    dataset = DataScreen("sub_dataset", "sub_mask")
    dataloader = DataLoader(dataset, 16, shuffle=True)
    
    net = AEModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    EPOCHS = 100
    optimizer = Adam(net.parameters(), lr=3e-2)
    loss_fn = nn.MSELoss()
    total_loss = 100000000
    for e in range(1, EPOCHS + 1):
        loss_train = 0.0
        for org_images, error_images, mask_images in dataloader:
            print(error_images.shape)
            error_images = error_images.to(device)
            mask_images = mask_images.to(device)
            pred_images = net(error_images)
            loss = loss_fn(pred_images, mask_images)
            optimizer.zero_grad()     
            loss.backward() 
            optimizer.step()
            loss_train += loss.item()
        print(f"Epochs: {e}, Loss{loss_train}...")
        if loss_train < total_loss:
            total_loss = loss_train
            save_path = "checkpoints"
            if not os.path.exists(save_path):
                os.makedirs(f"{save_path}")
            torch.save(net, f"{save_path}/AE_Epoch{e}.pth")



if __name__ == "__main__":
    pass
    # download_dataset()
    # make_mask()
    # make_dataset()
    train()