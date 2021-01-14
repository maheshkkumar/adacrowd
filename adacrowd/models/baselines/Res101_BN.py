import torch.nn.functional as F
from torch import nn
from torchvision import models

from misc.utils import initialize_weights


class Res101_BN(nn.Module):
    def __init__(self, pretrained=True, num_norm=6):
        super(Res101_BN, self).__init__()

        self.num_norm = num_norm
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = []
        self.backend = make_layers(
            self.backend_feat,
            in_channels=1024,
            dilation=True,
            batch_norm=True,
            num_norm=self.num_norm)
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1), nn.ReLU())
        initialize_weights(self.modules())

        res = models.resnet101(pretrained=pretrained)
        self.frontend = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
            res.layer1,
            res.layer2)
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 23, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)

        x = self.own_reslayer_3(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)

        return x


def make_layers(
        cfg,
        in_channels=3,
        batch_norm=False,
        dilation=False,
        num_norm=6):
    norm_counter = 0
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels,
                v,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate)
            if norm_counter < num_norm:
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                    norm_counter += 1
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_res_layer(block, planes, blocks, stride=1):
    downsample = None
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            planes *
            self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
