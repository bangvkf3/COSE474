import os
import numpy as np

import torch.nn as nn
import torch

##############################################################################################################
#                    TODO : X1 ~ X7에 올바른 숫자 또는 변수를 채워넣어 ResNet32 코드를 완성할 것                 #
##############################################################################################################


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes) # 각각의 label을 표현하는 features를 위한 weight matrix

        self.model = nn.Sequential(
            nn.Linear(X1 + X2, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((X3, X4), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape) # generation한 image를 reshape
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes) # 각각의 label을 표현하는 features를 위한 weight matrix

        self.model = nn.Sequential(
            nn.Linear(X5 + X6, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, X7),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

class Config():
    def __init__(self):
        self.batch_size = 100
        self.lr = 0.001
        self.num_epochs = 200
        self.latent_dim = 100 # noise vector size
        self.n_classes = 10
        self.img_size = 28
        self.channels = 1