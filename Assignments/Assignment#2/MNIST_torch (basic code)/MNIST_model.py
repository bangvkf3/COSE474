import torch
import torch.nn as nn


class MNIST_model(nn.Module):
    def __init__(self): # *hint* dropout probability를 위한 파라미터 추가
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(28 * 28, 300), # 첫 번째 층 정의
            # *hint* torch.nn.init.xavier_normal_ 등을 통해 weight initialization 적용
            nn.ReLU(), # activation function
            # *hint* nn.Dropout(drop_prob) 등을 통해 dropout 적용
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.mlp(x)
        return x


class Config(): # 하이퍼파라미터들을 위한 configuration
    def __init__(self):
        self.batch_size = 200
        self.lr_adam = 0.001
        self.lr_adadelta = 0.1
        self.epoch = 30
        self.weight_decay = 1e-03
