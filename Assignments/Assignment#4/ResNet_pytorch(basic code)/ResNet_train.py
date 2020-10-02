import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.utils.data.sampler import  SubsetRandomSampler
from torch_basic.ResNet_model import ResNet32_model, Config
from torch.optim import lr_scheduler


def data_load(data_augmentation):
    # train data augmentation : 1) 데이터 좌우반전(2배). 2) size 4만큼 패딩 후 32의 크기로 random cropping
    transforms_train = transforms.Compose([ # training data를 위한 transforms
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # input image 정규화 (standardization)
    ])
    transforms_val = transforms.Compose([ # validation data를 위한 transforms
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR10 dataset 다운로드
    if data_augmentation:
        train_data = dsets.CIFAR10(root='./dataset/', train=True, transform=transforms_train, download=True)
    else:
        train_data = dsets.CIFAR10(root='./dataset/', train=True, transform=transforms_val, download=True)
    val_data = dsets.CIFAR10(root='./dataset/', train=True, transform=transforms_val, download=True)
    #split 후에 transform을 적용하기 어렵기 때문에 50000개의 data에 대해 각각 전처리
    return train_data, val_data  # training set, validation set

def generate_batch(train_data, val_data):
    indices = list(range(len(train_data)))
    np.random.shuffle(indices) # training, val을 random하게 sampling하기 위해 index들을 shuffling
    train_indices, val_indices = indices[5000:], indices[:5000] # training, val data에 대한 index들
    train_sampler = SubsetRandomSampler(train_indices) # DataLoader 과정에서 training과 validation을 sampling하기 위한 Sampler
    val_sampler = SubsetRandomSampler(val_indices)
    train_batch_loader = DataLoader(train_data, cfg.batch_size, sampler=train_sampler)
    val_batch_loader = DataLoader(val_data, cfg.batch_size, sampler=val_sampler)
    return train_batch_loader, val_batch_loader


if __name__ == '__main__':
    # configuration
    cfg = Config()

    print('[CIFAR10_training]')
    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 데이터 로드
    # CIFAR10 dataset: [3,32,32] 사이즈의 이미지들을 가진 dataset
    train_data, val_data = data_load(cfg.data_augmentation) # * 아직 training validation 분리되지 않았음

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # data 개수 확인
    print('The number of training data: ', len(train_data))
    print('The number of validation data: ', len(val_data))

    # 학습 모델 생성

    model = ResNet32_model()

    if torch.cuda.is_available():
        model = model.to(device)

    # 배치 생성
    train_batch_loader, val_batch_loader = generate_batch(train_data, val_data) # batch 생성하면서 training , val 분리
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay) # lamda 값과 함께 weight decay 적용
    decay_iter = [32000, 48000]
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_iter, gamma=0.1) # 32000, 48000 iteration 때 lr을 10%로 감소시킴

    # training 시작
    start_time = time.time()
    highest_val_acc = 0
    val_acc_list = []
    global_steps = 0
    epoch = 0
    print('========================================')
    print("Start training...")
    while True:
        train_loss = 0
        train_batch_cnt = 0
        model.train()
        for img, label in train_batch_loader:
            global_steps += 1
            # img.shape: [200,3,32,32]
            # label.shape: [200]

            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()# iteration 마다 gradient를 0으로 초기화
            outputs = model(img)
            loss = criterion(outputs, label)#cross entropy loss 계산
            loss.backward()# 가중치 w에 대해 loss를 미분
            optimizer.step()# 가중치들을 업데이트
            step_lr_scheduler.step() # learning rate 업데이트

            train_loss += loss

            train_batch_cnt += 1

            if global_steps >= 64000:
                print("Training finished.")
                break

        ave_loss = train_loss / train_batch_cnt # 학습 데이터의 평균 loss
        training_time = (time.time() - start_time) / 60
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % ave_loss)
        print("training_time: %.2f minutes" % training_time)
        print("learning rate: %.6f" % step_lr_scheduler.get_lr()[0])

        # validation (for early stopping)
        correct_cnt = 0
        model.eval()
        for img, label in val_batch_loader:
            img = img.to(device)
            label = label.to(device)
            pred = model.forward(img)
            _, top_pred = torch.topk(pred, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            correct_cnt += int(torch.sum(top_pred == label))# 맞춘 개수 카운트

        val_acc = correct_cnt / 5000 * 100
        print("validation dataset accuracy: %.2f" % val_acc)
        val_acc_list.append(val_acc)
        if val_acc > highest_val_acc:# validation accuracy가 경신될 때
            save_path = './saved_model/setting_1/epoch_' + str(epoch + 1) + '.pth'
            # 위와 같이 저장 위치를 바꾸어 가며 각 setting의 epoch마다의 state를 저장할 것.
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)# best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
            highest_val_acc = val_acc
        epoch += 1
        if global_steps >= cfg.finish_step:
            break

    epoch_list = [i for i in range(1, epoch + 1)]
    plt.title('Validation dataset accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epoch_list, val_acc_list)
    plt.show()
