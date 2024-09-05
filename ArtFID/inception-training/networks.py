import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

import ops


class ResBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool=False):

        super(ResBlock, self).__init__()
        
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2 if self.downsample else 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if self.downsample:
            self.down = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(num_features=out_channels)
                        )
        
    def forward(self, x: Tensor):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            residual = self.down(residual)

        x += residual
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool=False,
                 stride: bool=True,
                 expansion: int=4,
                 ):

        super(Bottleneck, self).__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2 if self.downsample and stride else 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * expansion)

        if self.downsample:
            self.down = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * expansion, kernel_size=1, stride=2 if stride else 1, bias=False),
                            nn.BatchNorm2d(num_features=out_channels * expansion)
                        )

    def forward(self, x: Tensor):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            residual = self.down(residual)
        
        x += residual
        x = self.relu(x)
        return x


class ResNet18(nn.Module):

    def __init__(self, num_classes_head1, num_classes_head2, zero_init_residual=True):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(
                          ResBlock(in_channels=64, out_channels=64, downsample=False),
                          ResBlock(in_channels=64, out_channels=64, downsample=False)
                      )

        self.layer2 = nn.Sequential(
                          ResBlock(in_channels=64, out_channels=128, downsample=True),
                          ResBlock(in_channels=128, out_channels=128, downsample=False)
                      )

        self.layer3 = nn.Sequential(
                          ResBlock(in_channels=128, out_channels=256, downsample=True),
                          ResBlock(in_channels=256, out_channels=256, downsample=False)
                      )

        self.layer4 = nn.Sequential(
                          ResBlock(in_channels=256, out_channels=512, downsample=True),
                          ResBlock(in_channels=512, out_channels=512, downsample=False)
                      )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_head1 = nn.Linear(in_features=512, out_features=num_classes_head1)
        self.fc_head2 = nn.Linear(in_features=512, out_features=num_classes_head2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor, return_embedding=False, random_shuffle_once=False, random_shuffle=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer2(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer3(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer4(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)

        if random_shuffle_once:
            x = ops.random_spatial_shuffle(x)

        if return_embedding:
            embedding = x

        x = self.avgpool(x)
        x = torch.reshape(x, shape=(x.shape[0], -1))
        out_head1 = self.fc_head1(x)
        out_head2 = self.fc_head2(x)
        
        if return_embedding:
            return out_head1, out_head2, embedding
        return out_head1, out_head2

        
class ResNet34(nn.Module):

    def __init__(self, num_classes_head1, num_classes_head2, zero_init_residual=True):
        super(ResNet34, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(
                          ResBlock(in_channels=64, out_channels=64, downsample=False),
                          ResBlock(in_channels=64, out_channels=64, downsample=False),
                          ResBlock(in_channels=64, out_channels=64, downsample=False)
                      )

        self.layer2 = nn.Sequential(
                          ResBlock(in_channels=64, out_channels=128, downsample=True),
                          ResBlock(in_channels=128, out_channels=128, downsample=False),
                          ResBlock(in_channels=128, out_channels=128, downsample=False),
                          ResBlock(in_channels=128, out_channels=128, downsample=False)
                      )

        self.layer3 = nn.Sequential(
                          ResBlock(in_channels=128, out_channels=256, downsample=True),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False)
                      )

        self.layer4 = nn.Sequential(
                          ResBlock(in_channels=256, out_channels=512, downsample=True),
                          ResBlock(in_channels=512, out_channels=512, downsample=False),
                          ResBlock(in_channels=512, out_channels=512, downsample=False)
                      )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_head1 = nn.Linear(in_features=512, out_features=num_classes_head1)
        self.fc_head2 = nn.Linear(in_features=512, out_features=num_classes_head2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor, return_embedding=False, random_shuffle_once=False, random_shuffle=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer2(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer3(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer4(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)

        if random_shuffle_once:
            x = ops.random_spatial_shuffle(x)

        if return_embedding:
            embedding = x

        x = self.avgpool(x)
        x = torch.reshape(x, shape=(x.shape[0], -1))
        out_head1 = self.fc_head1(x)
        out_head2 = self.fc_head2(x)
        
        if return_embedding:
            return out_head1, out_head2, embedding
        return out_head1, out_head2


class ResNet50(nn.Module):

    def __init__(self, num_classes_head1, num_classes_head2, zero_init_residual=True):
        super(ResNet50, self).__init__()

        expansion = 4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
                          Bottleneck(in_channels=64, out_channels=64, downsample=True, stride=False, expansion=expansion),
                          Bottleneck(in_channels=64 * expansion, out_channels=64, downsample=False, expansion=expansion),
                          Bottleneck(in_channels=64 * expansion, out_channels=64, downsample=False, expansion=expansion)
                      )

        self.layer2 = nn.Sequential(
                          Bottleneck(in_channels=64 * expansion, out_channels=128, downsample=True, expansion=expansion),
                          Bottleneck(in_channels=128 * expansion, out_channels=128, downsample=False, expansion=expansion),
                          Bottleneck(in_channels=128 * expansion, out_channels=128, downsample=False, expansion=expansion),
                          Bottleneck(in_channels=128 * expansion, out_channels=128, downsample=False, expansion=expansion)
                      )

        self.layer3 = nn.Sequential(
                          Bottleneck(in_channels=128 * expansion, out_channels=256, downsample=True, expansion=expansion),
                          Bottleneck(in_channels=256 * expansion, out_channels=256, downsample=False, expansion=expansion),
                          Bottleneck(in_channels=256 * expansion, out_channels=256, downsample=False, expansion=expansion),
                          Bottleneck(in_channels=256 * expansion, out_channels=256, downsample=False, expansion=expansion),
                          Bottleneck(in_channels=256 * expansion, out_channels=256, downsample=False, expansion=expansion),
                          Bottleneck(in_channels=256 * expansion, out_channels=256, downsample=False, expansion=expansion),
                      )

        self.layer4 = nn.Sequential(
                          Bottleneck(in_channels=256 * expansion, out_channels=512, downsample=True, expansion=expansion),
                          Bottleneck(in_channels=512 * expansion, out_channels=512, downsample=False, expansion=expansion),
                          Bottleneck(in_channels=512 * expansion, out_channels=512, downsample=False, expansion=expansion),
                      )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_head1 = nn.Linear(in_features=512 * expansion, out_features=num_classes_head1)
        self.fc_head2 = nn.Linear(in_features=512 * expansion, out_features=num_classes_head2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor, return_embedding=False, random_shuffle_once=False, random_shuffle=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer2(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer3(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)
        x = self.layer4(x)
        if random_shuffle: x = ops.random_spatial_shuffle(x)

        if random_shuffle_once and not random_shuffle:
            x = ops.random_spatial_shuffle(x)

        if return_embedding:
            embedding = x

        x = self.avgpool(x)
        x = torch.reshape(x, shape=(x.shape[0], -1))
        out_head1 = self.fc_head1(x)
        out_head2 = self.fc_head2(x)
        
        if return_embedding:
            return out_head1, out_head2, embedding
        return out_head1, out_head2


class MyResNet34(nn.Module):

    def __init__(self, num_classes_head1, num_classes_head2, pooling='avg', down_resolution=4, zero_init_residual=True):
        super(MyResNet34, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(
                          ResBlock(in_channels=64, out_channels=64, downsample=False),
                          ResBlock(in_channels=64, out_channels=64, downsample=False),
                          ResBlock(in_channels=64, out_channels=64, downsample=False)
                      )

        self.layer2 = nn.Sequential(
                          ResBlock(in_channels=64, out_channels=128, downsample=True),
                          ResBlock(in_channels=128, out_channels=128, downsample=False),
                          ResBlock(in_channels=128, out_channels=128, downsample=False),
                          ResBlock(in_channels=128, out_channels=128, downsample=False)
                      )

        self.layer3 = nn.Sequential(
                          ResBlock(in_channels=128, out_channels=256, downsample=True),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False)
                      )

        self.layer4 = nn.Sequential(
                          ResBlock(in_channels=256, out_channels=512, downsample=True),
                          ResBlock(in_channels=512, out_channels=512, downsample=False),
                          ResBlock(in_channels=512, out_channels=512, downsample=False)
                      )
        
        if pooling == 'avg':
            self.adapool = nn.AdaptiveAvgPool2d(output_size=(down_resolution, down_resolution))
        else:
            self.adapool = nn.AdaptiveMaxPool2d(output_size=(down_resolution, down_resolution))

        self.fc_head1 = nn.Linear(in_features=512 * down_resolution * down_resolution, out_features=num_classes_head1)
        self.fc_head2 = nn.Linear(in_features=512 * down_resolution * down_resolution, out_features=num_classes_head2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor, embedding=False, pooled_embedding=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if embedding:
            return x

        x = self.adapool(x)
        
        if pooled_embedding:
            return x

        x = torch.reshape(x, shape=(x.shape[0], -1))
        out_head1 = self.fc_head1(x)
        out_head2 = self.fc_head2(x)
        
        return out_head1, out_head2

    def preprocess(self, x):
        # normalize range from [0, 1] to [-1, 1]
        x = 2 * x - 1
        x = np.transpose(x, axes=(0, 3, 1, 2)).astype(np.float32)
        x = torch.from_numpy(x)
        return x

    def get_embedding(self, x):
        x = self.preprocess(x)
        x = self.forward(x, embedding=True)
        x = torch.reshape(x, shape=(x.shape[0], -1))
        return x

    def get_pooled_embedding(self, x):
        x = self.preprocess(x)
        x = self.forward(x, pooled_embedding=True)
        x = torch.reshape(x, shape=(x.shape[0], -1))
        return x


class MyResNet(nn.Module):

    def __init__(self, num_classes_head1, num_classes_head2, pooling='avg', down_resolution=4, zero_init_residual=True):
        super(MyResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(
                          ResBlock(in_channels=64, out_channels=64, downsample=False),
                          ResBlock(in_channels=64, out_channels=64, downsample=False),
                          ResBlock(in_channels=64, out_channels=64, downsample=False)
                      )

        self.layer2 = nn.Sequential(
                          ResBlock(in_channels=64, out_channels=128, downsample=True),
                          ResBlock(in_channels=128, out_channels=128, downsample=False),
                          ResBlock(in_channels=128, out_channels=128, downsample=False),
                          ResBlock(in_channels=128, out_channels=128, downsample=False)
                      )

        self.layer3 = nn.Sequential(
                          ResBlock(in_channels=128, out_channels=256, downsample=True),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                      )

        self.layer4 = nn.Sequential(
                          ResBlock(in_channels=256, out_channels=256, downsample=True),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                          ResBlock(in_channels=256, out_channels=256, downsample=False),
                      )

        self.layer5 = nn.Sequential(
                          ResBlock(in_channels=256, out_channels=512, downsample=True),
                          ResBlock(in_channels=512, out_channels=512, downsample=False),
                          ResBlock(in_channels=512, out_channels=512, downsample=False)
                      )
        
        if pooling == 'avg':
            self.adapool = nn.AdaptiveAvgPool2d(output_size=(down_resolution, down_resolution))
        else:
            self.adapool = nn.AdaptiveMaxPool2d(output_size=(down_resolution, down_resolution))

        self.fc_head1 = nn.Linear(in_features=512 * down_resolution * down_resolution, out_features=num_classes_head1)
        self.fc_head2 = nn.Linear(in_features=512 * down_resolution * down_resolution, out_features=num_classes_head2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor, embedding=False, pooled_embedding=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        if embedding:
            return x

        x = self.adapool(x)

        if pooled_embedding:
            return x

        x = torch.reshape(x, shape=(x.shape[0], -1))
        out_head1 = self.fc_head1(x)
        out_head2 = self.fc_head2(x)
        
        return out_head1, out_head2

    def preprocess(self, x):
        # normalize range from [0, 1] to [-1, 1]
        x = 2 * x - 1
        x = np.transpose(x, axes=(0, 3, 1, 2)).astype(np.float32)
        x = torch.from_numpy(x)
        return x

    def get_embedding(self, x):
        x = self.preprocess(x)
        x = self.forward(x, embedding=True)
        x = torch.reshape(x, shape=(x.shape[0], -1))
        return x

    def get_pooled_embedding(self, x):
        x = self.preprocess(x)
        x = self.forward(x, pooled_embedding=True)
        x = torch.reshape(x, shape=(x.shape[0], -1))
        return x


