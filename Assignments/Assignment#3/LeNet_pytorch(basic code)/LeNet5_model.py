import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_model(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################################################################
        #                         TODO : LeNet5 모델 생성                                                             #
        ##############################################################################################################
        # * hint
        # conv1:  5*5, input_channel: 3, output_channel(# of filters): 6
        # max pooling: size: 2, stride 2
        # conv2:  5*5, input_channel: 6, output_channel(# of filters): 16
        # fc1: (16 * 5 * 5, 120)
        # fc2: (120, 84)
        # fc3: (84, 10)

        # * hint he initialization: stddev = sqrt(2/n), filter에서 n 값은?
        pass

    def forward(self, x):
        ##############################################################################################################
        #                         TODO : forward path 수행, 결과를 x에 저장                                            #
        ##############################################################################################################
        # * hint
        # conv1
        # relu
        # max_pooling
        # conv2
        # relu
        # max_pooling
        # reshape
        # fully connected layer1
        # relu
        # fully connected layer2
        # relu
        # fully connected layer3
        pass
        return x

class Config():
    def __init__(self):
        self.batch_size = 128
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-04
        self.finish_step = 64000
        self.data_augmentation = True
