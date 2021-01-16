import torch.nn.functional as F
from torch import nn
from torchvision import models

from misc.utils import initialize_weights

from .blocks import (GuidedBatchNorm2d, GuidingEncoder, GuidingMLP,
                     get_num_adaptive_params)


class CSRNet_GBN(nn.Module):
    def __init__(self, num_gbnnorm=6):
        super(CSRNet_GBN, self).__init__()

        self.num_gbnnorm = num_gbnnorm

        # crowd encoder consists of normalization layers
        # Params: pretrained, default_value=True
        self.crowd_encoder = CrowdEncoder()

        # guiding encoder does not consist of normalization layers
        self.guiding_encoder = GuidingEncoder(
            downs=4,
            ind_im=3,
            dim=64,
            latent_dim=64,
            norm='none',
            activ='relu',
            pad_type='reflect',
            pool_type='adapt_avg_pool')

        # decoder to generate the output density map
        self.crowd_decoder = CrowdDecoder(norm='GBN', num_gbnnorm=self.num_gbnnorm)

        # MLP to generate the adaptive normalization parameters from the output
        # of guiding decoder
        num_ada_params = get_num_adaptive_params(self.crowd_encoder)
        self.guiding_mlp = GuidingMLP(
            in_dim=64,
            out_dim=num_ada_params,
            dim=256,
            n_blk=3,
            norm='none',
            activ='relu')

    def forward(self, x):
        pass


class CrowdDecoder(nn.Module):
    """
    Class to decode the output density map from encoded (crowd + guiding) features
    """

    def __init__(self, norm=None, num_gbnnorm=6):
        super(CrowdDecoder, self).__init__()

        self.norm = norm
        self.num_gbnnorm = num_gbnnorm

        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(
            self.backend_feat,
            in_channels=512,
            dilation=True,
            norm=self.norm,
            num_gbnnorm=self.num_gbnnorm)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)

        return x


class CrowdEncoder(nn.Module):
    """
    CSRNet implementation
    Reference: https://github.com/leeyeehoo/CSRNet-pytorch
    Paper: https://arxiv.org/abs/1802.10062
    """

    def __init__(self, pretrained=True):
        super(CrowdEncoder, self).__init__()

        # VGG feature extractor
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)

        # Loading weights from the pre-trained VGG16 model
        if pretrained:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())

    def forward(self, x):
        x = self.frontend(x)  # output feature shape (B x 512 x H/8 x W/8)
        return x

    # Weight initialization as per the CSRNet paper
    # Mean=0, Standard deviation=0.01
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, dilation=False, norm=None, num_gbnnorm=6):
    """
    Method to make decoder layers
    """
    gbn_counter = 0
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
            if norm == 'GBN':
                if gbn_counter < num_gbnnorm:
                    layers += [conv2d,
                               GuidedBatchNorm2d(v),
                               nn.ReLU(inplace=True)]
                    gbn_counter += 1
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
