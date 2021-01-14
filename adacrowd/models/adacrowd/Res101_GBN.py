import torch.nn.functional as F
from torch import nn
from torchvision import models

from misc.utils import initialize_weights

from .blocks import (GuidedBatchNorm2d, GuidingEncoder, GuidingMLP,
                     get_num_adaptive_params)


class Res101_GBN(nn.Module):
    def __init__(self, num_gbnnorm=6):
        super(Res101_GBN, self).__init__()

        # boolean condition to check if SSIM vec is required
        self.num_gbnnorm = num_gbnnorm
        
        # crowd encoder consists of normalization layers
        # Params: pretrained, default_value=True
        self.crowd_encoder = CrowdEncoder()

        # guiding encoder does not consist of normalization layers
        self.crowd_encoder = GuidingEncoder(downs=4, ind_im=3, dim=64, latent_dim=64, norm='none', activ='relu',
                                                    pad_type='reflect', pool_type='adapt_avg_pool')

        # decoder to generate the output density map
        self.crowd_decoder = CrowdDecoder(norm='GBN', num_gbnnorm=self.num_gbnnorm)

        # MLP to generate the adaptive normalization parameters from the output of guiding decoder
        num_ada_params = get_num_adaptive_params(self.crowd_decoder)
        self.guiding_mlp = GuidingMLP(in_dim=64, out_dim=num_ada_params, dim=256, n_blk=3, norm='none',
                                            activ='relu')

    def forward(self, x):
        pass


class CrowdEncoder(nn.Module):
    """
    Class to extract features from the crowd image
    Input dim: B x 3 x H x W
    Output dim: B x 512 x H/8 x W/8

    Params: pretrained=True, makes use of pretrained weights from ImageNet for the encoder part of the encoder
    """

    def __init__(self, pretrained=True):
        super(CrowdEncoder, self).__init__()
        res = models.resnet101(pretrained=pretrained)
        self.frontend = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2)
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 23, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())
        
    def forward(self, x):
        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        return x


class CrowdDecoder(nn.Module):
    """
    Class to decode the output density map from encoded (crowd + guiding) features
    """

    def __init__(self, norm=None, num_gbnnorm=6):
        super(CrowdDecoder, self).__init__()

        self.norm = norm
        self.num_gbnnorm = num_gbnnorm
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True, norm=self.norm,
                                   num_gbnnorm=self.num_gbnnorm)
        self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU())

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)

        return x


def make_layers(cfg, in_channels=3, dilation=False, norm=None, num_gbnnorm=6):
    """
    Method to make layers for ResNet
    """
    gbn_counter = 0
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if norm == 'GBN':
                if gbn_counter < num_gbnnorm:
                    layers += [conv2d, GuidedBatchNorm2d(v), nn.ReLU(inplace=True)]
                    gbn_counter += 1
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_res_layer(block, planes, blocks, stride=1):
    """
    Method to make resnet layer for ResNet
    """
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
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
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
