from turtle import forward
import numpy as np
import cv2
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F


class AEModel(nn.Module):
    def __init__(self, image_size=(224,224,3), latent_dim=128) -> None:
        super().__init__()
        num = 3
        self.encoder = self.get_encoder(num)
        self.decoder = self.get_decoder(num)
        last_dim = int(image_size[0] / 2 ** (num + 1)  *  image_size[1] / 2 ** (num + 1) * 16 * 2 ** (num))
        print(last_dim)

    def get_encoder(self, num=3):
        encoder = nn.Sequential()
        encoder.add_module("init", nn.Conv2d(3, 16, (3,3), 2, 1) )
        for i in range(num):
            encoder.add_module(f"conv{i+1}", nn.Conv2d(16 *  2**(i), 16 *  2**(i + 1), (3,3), 2, 1))
            encoder.add_module(f"batch{i+1}", nn.BatchNorm2d(16 * 2**(i + 1)))
            encoder.add_module(f"leakyrelu{i+1}", nn.LeakyReLU())
        return encoder

    def get_decoder(self, num=3):
        encoder = nn.Sequential()
        for ix in range(num):
            i = num - ix - 1
            encoder.add_module(f"re-conv{i+1}", nn.ConvTranspose2d(16 *  2**(i + 1), 16 *  2**(i), (3,3), 2, 1))
            encoder.add_module(f"re-batch{i+1}", nn.BatchNorm2d(16 * 2**(i)))
            encoder.add_module(f"re-leakyrelu{i+1}", nn.LeakyReLU())
        encoder.add_module("recontruction", nn.Conv2d(16, 3, (3,3), 2, 1) )
        return encoder


    def forward(self, x):
        N = x.shape[0]
        x = self.encoder(x)
        last_shape = x.shape
        print(last_shape)
        x = x.view(N, -1)
        # last_dim = x.shape[1]
        return x


ae = AEModel()
print(ae)
x = torch.zeros((10, 3, 224, 224))
pred = ae(x)
print(pred.shape)