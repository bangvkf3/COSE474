import os
import numpy as np

import torch.nn as nn
import torch

##############################################################################################################
#                    TODO : X1 ~ X7에 올바른 숫자 또는 변수를 채워넣어 Conditional GAN 코드를 완성할 것                 #
##############################################################################################################



class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes) # 각각의 label을 표현하는 features를 위한 weight matrix

        self.model_1_1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model_1_2 = nn.Sequential(
            nn.ConvTranspose2d(n_classes, 128, 7, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 14 14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),  # 1 28 28
            nn.Tanh()
        )

    def forward(self, noise, labels):
        noise = noise.view([-1, 100, 1, 1])
        labels = self.label_emb(labels).view([-1, 10, 1, 1])

        noise = self.model_1_1(noise)
        labels = self.model_1_2(labels)
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((noise, labels), 1)

        img = self.model_2(gen_input)
        img.view([-1, 784])
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model_1 = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + n_classes, 784),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.model_2 = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, 7, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)

        d_in = self.model_1(d_in)
        d_in = d_in.view([-1, 1, 28, 28])

        validity = self.model_2(d_in)
        return validity.view(-1, 1)

class Config():
    def __init__(self):
        self.batch_size = 100
        self.lr = 0.001
        self.num_epochs = 200
        self.latent_dim = 100  # noise vector size
        self.n_classes = 10
        self.img_size = 28
        self.channels = 1