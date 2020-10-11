import argparse
import os
import numpy as np
import math
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch
from CGAN_model_default import Generator, Discriminator, Config
import torch.nn as nn
import torch.nn.functional as F

cfg = Config()

def sample_image(n_row, batches_done):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, cfg.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "out/%d.png" % batches_done, nrow=n_row, normalize=True)


transforms_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# Fashion MNIST dataset 다운로드
train_data = datasets.FashionMNIST("./data/mnist", train=True, download=True,transform=transforms_train)

# 배치 단위로 데이터를 처리해주는 Data loader
dataloader = torch.utils.data.DataLoader(train_data,batch_size=cfg.batch_size,shuffle=True)

print('[CGAN_training]')
# GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
print("GPU Available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# data 개수 확인
print('The number of training data: ', len(train_data))

os.makedirs("out", exist_ok=True)

img_shape = (cfg.channels, cfg.img_size, cfg.img_size)


# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
# generator = Generator(cfg.n_classes, cfg.latent_dim, img_shape, cfg.img_size)
# discriminator = Discriminator(cfg.n_classes, img_shape, cfg.img_size)

# default
generator = Generator(cfg.n_classes, cfg.latent_dim, img_shape)
discriminator = Discriminator(cfg.n_classes, img_shape)

if torch.cuda.is_available():
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr)

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

start_time = time.time()
print('========================================')
print("Start training...")

for epoch in range(cfg.num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        #  Train Generator
        optimizer_G.zero_grad()

        # noise  생성
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, cfg.latent_dim))))
        # random 하게 생성할 이미지의 label 결정
        gen_labels = Variable(LongTensor(np.random.randint(0, cfg.n_classes, batch_size)))

        # Generater를 이용하여 noise로 부터 image 생성
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step() # Generator 내 파라미터를 update (discriminator는 freeze)

        #  Train Discriminator
        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels) # discriminator의 실제이미지 예측
        d_real_loss = adversarial_loss(validity_real, valid) # discriminator의 실제이미지 예측에 대한 loss

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels) # discriminator의 가짜이미지 예측
        d_fake_loss = adversarial_loss(validity_fake, fake) # discriminator의 가짜이미지 예측에 대한 loss

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step() # discriminator 내 파라미터를 update (Generator는 freeze)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, cfg.num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            sample_image(n_row=10, batches_done=batches_done)

training_time = (time.time() - start_time) / 60
print('========================================')
print("training_time: %.2f minutes" % training_time)




