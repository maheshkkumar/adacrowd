import torch
from torch import nn
from torch.nn import functional as F

from misc.utils import initialize_weights

__constants__ = ['GuidedBatchNorm2d']

def init_params(modules):
    for m in modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class GuidingEncoder(nn.Module):
    """
    Class to capture guiding from an image

    Params: downs, number of blocks to downsample the input image. Each block downsamples the image by a factor of 2
            ind_im, number of input dimension. Mostly it is the channels of the input image
            dim, number of dimensions in the intermediate features of the encoder
            latent_dim, number of dimensions in the feature output from the encoder
            norm, type of normalization applied on the data.
            activ, type of activation layer used in the network.
            pad_type, type of image padding used in the layers.


    Architecture:
            The network uses 'downs' number of downsampling blocks, followed by an adaptive pool layer, 1x1 conv layer to reduce the channel dimensions
    """

    def __init__(
            self,
            downs,
            ind_im,
            dim,
            latent_dim,
            norm,
            activ,
            pad_type,
            pool_type='adapt_avg_pool'):
        super(GuidingEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(ind_im, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2

        if pool_type == 'adapt_avg_pool':
            self.model += [nn.AdaptiveAvgPool2d(1)]
        elif pool_type == 'adapt_max_pool':
            self.model += [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*self.model)

        # default initialization of gaussian with mean 1 and std 0.01
        initialize_weights(self.modules())

    def forward(self, x):
        return self.model(x)


class GuidingMLP(nn.Module):
    """
    Class to convert the guiding encoder output (tensor) to a vector, where the length of the vector is equal to the total number of parameters in the Adaptive normalization layers in the decoder

    Params: in_dim, size of the input dimension
            out_dim, size of the output vector
            dim, size of the intermediate layer
            n_blk, the number of MLP blocks
            norm, type of normalization applied on the data
            activ, type of activation layer used in the network
    """

    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):
        super(GuidingMLP, self).__init__()
        self.model = []

        dim = 256
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

        # default initialization of gaussian with mean 1 and std 0.01
        initialize_weights(self.modules())

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            # x = self.conv(self.pad(x))
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
        else:
            # x = self.conv(self.pad(x))
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class GuidedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(GuidedBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
            self.bias is not None, "Please assign GBN weights first"
        running_mean = self.running_mean
        running_var = self.running_var
        out = F.batch_norm(
            x, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def assign_adaptive_params(gbn_params, model):
    if gbn_params.dim() == 1:
        gbn_params = gbn_params.unsqueeze(0)

    for m in model.modules():
        if m.__class__.__name__ in __constants__:
            mean = gbn_params[:, :m.num_features]
            std = gbn_params[:, m.num_features:2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if gbn_params.size(1) > 2 * m.num_features:
                gbn_params = gbn_params[:, 2 * m.num_features:]


def get_num_adaptive_params(model):
    # return the number of GBN parameters needed by the model
    num_gbn_params = 0
    for m in model.modules():
        if m.__class__.__name__ in __constants__:
            num_gbn_params += 2 * m.num_features
    return num_gbn_params


if __name__ == '__main__':
    layer = GuidedBatchNorm2d(16)
    model = nn.Sequential(layer)
    x = torch.rand(1, 1024)
    assign_adaptive_params(x, layer)
    a = torch.rand(1, 16, 32, 32)
    output = model(a)
    print(output.shape)
