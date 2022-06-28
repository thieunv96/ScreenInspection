from turtle import forward
import numpy as np
import cv2
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from glob import glob

class AEModel(nn.Module):
    def __init__(self, image_size=(224,224,3), latent_dim=128) -> None:
        super().__init__()
        num = 3
        self.encoder = self.get_encoder(num)
        self.decoder = self.get_decoder(num)
        last_dim = int(image_size[0] / 2 ** (num + 1)  *  image_size[1] / 2 ** (num + 1) * 16 * 2 ** (num))
        self.en_fc1 = nn.Linear(last_dim, 1024)
        self.en_fc2 = nn.Linear(1024, latent_dim)
        self.de_fc1 = nn.Linear(latent_dim, 1024)
        self.de_fc2 = nn.Linear(1024, last_dim)

    def get_encoder(self, num=3):
        encoder = nn.Sequential()
        encoder.add_module("init", nn.Conv2d(3, 16, (3,3), 2, 1) )
        encoder.add_module(f"batch_init", nn.BatchNorm2d(16))
        encoder.add_module(f"leakyrelu_init", nn.LeakyReLU())
        for i in range(num):
            encoder.add_module(f"conv{i+1}", nn.Conv2d(16 *  2**(i), 16 *  2**(i + 1), (3,3), 2, 1))
            encoder.add_module(f"batch{i+1}", nn.BatchNorm2d(16 * 2**(i + 1)))
            encoder.add_module(f"leakyrelu{i+1}", nn.LeakyReLU())
        return encoder

    def get_decoder(self, num=3):
        encoder = nn.Sequential()
        for ix in range(num):
            i = num - ix - 1
            encoder.add_module(f"re-conv{i+1}", nn.ConvTranspose2d(16 *  2**(i + 1), 16 *  2**(i), 4, 2, 1))
            encoder.add_module(f"re-batch{i+1}", nn.BatchNorm2d(16 * 2**(i)))
            encoder.add_module(f"re-leakyrelu{i+1}", nn.LeakyReLU())
        encoder.add_module("recontruction", nn.ConvTranspose2d(16, 3, 4, 2 , 1))
        encoder.add_module(f"recontructionSigmoid", nn.Sigmoid())
        return encoder


    def forward(self, x):
        N = x.shape[0]
        x = self.encoder(x)
        last_shape = x.shape
        x = x.view(N, -1)
        x = self.en_fc1(x)
        dim = self.en_fc2(x)
        x = self.de_fc1(dim)
        x = self.de_fc2(x)
        x =  torch.reshape(x, last_shape)
        x = self.decoder(x)
        return x


class DataScreen(Dataset):
    def __init__(self, image_path, mask_path) -> None:
        super().__init__()
        self.list_images = glob(image_path + "/*")
        self.list_masks = glob(mask_path + "/*")


    def add_mask(self, image, mask):
        
        bx = np.random.randint(1,10)
        by = np.random.randint(1,10)
        bx = bx + 1 if bx % 2 == 0 else bx
        by = by + 1 if by % 2 == 0 else by
        mask = cv2.blur(mask, (bx, by))
        mask[mask < 50] = 0
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_threshold = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        weight = np.random.uniform(0, 1, (1,))[0]
        image_addWeighted = cv2.addWeighted(image, 1 - weight, mask, weight, 0.0)
        final_mask = cv2.add(image, image_addWeighted, mask=mask_threshold)
        error_image = cv2.absdiff(final_mask, image)
        return error_image, final_mask

    def __len__(self):
        return len(self.list_images)



    def convert(self, image):
        image = image / 255.0
        image = np.transpose(image, (2,0,1))
        return image

    def __getitem__(self, idx):
        image = cv2.imread(self.list_images[idx], 1)
        id_mask = np.random.randint(0, len(self.list_masks))
        mask = cv2.imread(self.list_masks[id_mask], 1)
        error_image, final_mask = self.add_mask(image, mask)
        image = self.convert(image)
        error_image = self.convert(error_image)
        final_mask = self.convert(final_mask)
        image = torch.from_numpy(image).type(torch.float32)
        error_image = torch.from_numpy(error_image).type(torch.float32)
        final_mask = torch.from_numpy(final_mask).type(torch.float32)
        return image, error_image, final_mask



if __name__ == "__main__":
    ae = AEModel()
    print(ae)
    x = torch.zeros((10, 3, 224, 224))
    pred = ae(x)
    print(pred.shape)