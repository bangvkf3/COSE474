import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # [batch_size,1,28,28] -> [batch_size,32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch_size,32,28,28] -> [batch_size,32,14,14]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # [batch_size,32,14,14] -> [batch_size,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [batch_size,64,14,14] -> [batch_size,64,7,7]
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 10),  # [batch_size,64*7*7] -> [batch_size,10]
        )  # (1, 3136) (3136, 10)

    def forward(self, x):
        out = self.layer(x)  # self.layer에 정의한 Sequential의 연산을 차례대로 다 실행
        out = out.view(-1, 64 * 7 * 7)  # view 함수를 이용해 텐서의 형태를 [batch_size,3136]
        out = self.fc_layer(out)
        return out


class Config(): # 하이퍼파라미터들을 위한 configuration
    def __init__(self):
        self.batch_size = 200
        self.lr_adam = 0.001
        self.lr_adadelta = 0.1
        self.epoch = 30
        self.weight_decay = 1e-03
