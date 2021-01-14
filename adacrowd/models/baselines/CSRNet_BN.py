import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CSRNet_BN(nn.Module):
    """
    CSRNet implementation
    Reference: https://github.com/leeyeehoo/CSRNet-pytorch
    Paper: https://arxiv.org/abs/1802.10062
    """

    def __init__(self, pretrained=True, num_norm=6):
        super(CSRNet_BN, self).__init__()

        # number of normalization layers
        self.num_norm = num_norm

        # VGG backbone
        self.frontend_feat = [64, 64, 'M', 128, 128,
                              'M', 256, 256, 256, 'M', 512, 512, 512]

        # Density map estimator
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat,
            in_channels=512,
            dilation=True,
            batch_norm=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Loading weights from the pre-trained VGG16 model
        if pretrained:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())

    def forward(self, x):
        x = self.frontend(x)  # output feature shape (B x 512 x H/8 x W/8)
        x = self.backend(x)  # output feature shape (B x 64 x H/8 x W/8)
        x = self.output_layer(x)  # output feature shape (B x 1 x H/8 x W/8)
        # upsampling the output 1/8th of the input image by 2^3
        x = F.upsample(x, scale_factor=8)

        return x

    # Weight initialization as per the CSRNet paper
    # Mean=0, Standard deviation=0.01
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif self.affine and isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


# Method to construct the layers for the backend / density map estimator
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
