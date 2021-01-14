import torch
import torch.nn as nn


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, num_norm=6):
        super(CrowdCounter, self).__init__()

        self.num_norm = num_norm

        if model_name == 'CSRNet_BN':
            from .baselines.CSRNet_BN import CSRNet_BN as net
        elif model_name == 'Res101_BN':
            from .baselines.Res101_BN import Res101_BN as net
        elif model_name == 'Res101_SFCN_BN':
            from .baselines.Res101_SFCN_BN import Res101_SFCN_BN as net
        self.CCN = net(num_norm=self.num_norm)
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.loss_mse = self.build_loss(
            density_map.squeeze(), gt_map.squeeze())
        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
