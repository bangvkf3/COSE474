import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##############################################################################################################
#                    TODO : X1 ~ X10에 올바른 숫자 또는 변수를 채워넣어 ResNet32 코드를 완성할 것                 #
##############################################################################################################

class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        # 패딩전 x: torch.Size([200, 32, 16, 16])
        x = F.pad(x, [0, 0, 0, 0, 0, self.add_channels])
        # 패딩후 x: torch.Size([200, 64, 16, 16])
        x = self.pooling(x)
        # 풀링후 x: torch.Size([200, 64, 8, 8])
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=X1, stride=X2, padding=X3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=X4, stride=X5, padding=X6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.down_sample is not None:
            shortcut = self.down_sample(shortcut)

        x += shortcut
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, num_layers, block, num_classes=10):
        super().__init__()
        # input img : [3, 32, 32]
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=X7, out_channels=X8,
                               kernel_size=X9, padding=X10, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # feature map size = [16,32,32]
        self.layers_2n = self.get_layers(block, 16, 16, stride=1)
        # feature map size = [32,16,16]
        self.layers_4n = self.get_layers(block, 16, 32, stride=2)
        # feature map size = [64,8,8]
        self.layers_6n = self.get_layers(block, 32, 64, stride=2)

        # output layers
        self.pool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, in_channels, out_channels, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        layers_list = nn.ModuleList([block(in_channels, out_channels, stride, down_sample)])

        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers_2n(x)
        x = self.layers_4n(x)
        x = self.layers_6n(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x


def ResNet32_model():
    block = ResidualBlock
    model = ResNet(5, block)
    return model

class Config():
    def __init__(self):
        self.batch_size = 128
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-04
        self.finish_step = 64000
        self.data_augmentation = True