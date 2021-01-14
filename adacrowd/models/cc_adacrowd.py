# import ipdb
import numpy as np
import torch
import torch.nn as nn

from .adacrowd.blocks import assign_adaptive_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SINGLE_FOLDER_DATASETS = ["WE", "CityUHK"]


class CrowdCounterAdaCrowd(nn.Module):
    def __init__(self, gpus, model_name, num_gbnnorm=6):
        super(CrowdCounterAdaCrowd, self).__init__()
        self.gpus = gpus
        self.model_name = model_name
        self.num_gbnnorm = num_gbnnorm

        if self.model_name == 'CSRNet_GBN':
            from .adacrowd.CSRNet_GBN import CSRNet_GBN as net
        elif self.model_name == 'Res101_GBN':
            from .adacrowd.Res101_GBN import Res101_GBN as net
        elif self.model_name == 'Res101_SFCN_GBN':
            from .adacrowd.Res101_SFCN_GBN import Res101_SFCN_GBN as net
        self.CCN = net(num_gbnnorm=self.num_gbnnorm)

        if len(self.gpus) > 1:
            self.CCN = torch.nn.DataParallel(
                self.CCN, device_ids=self.gpus).to(device)
        else:
            self.CCN = self.CCN.to(device)
        self.loss_mse_fn = nn.MSELoss().to(device)

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, crow_imgs, gui_imgs, gt_map):

        app_mean_list = []
        for bid in range(gui_imgs.shape[0]):
            batch_gui_img = gui_imgs[bid, :, :, :]
            split_gui_imgs = batch_gui_img.split(dim=0, split_size=3)
            split_gui_imgs = [gui_img.to(device).unsqueeze(0)
                              for gui_img in split_gui_imgs]

            app_latent_enc = [self.CCN.guiding_encoder(
                img) for img in split_gui_imgs]
            app_latent = [self.CCN.guiding_mlp(
                img) for img in app_latent_enc]

            # mean the features of the unlabeled images
            app_mean = torch.mean(torch.cat(app_latent), dim=0).unsqueeze(0)
            app_mean_list.append(app_mean)

        app_mean_list = torch.mean(
            torch.cat(app_mean_list),
            dim=0).unsqueeze(0)
        app_latent_enc = torch.mean(
            torch.cat(app_latent_enc),
            dim=0).unsqueeze(0)

        # crowd encoder output
        crow_img = self.CCN.crowd_encoder(crow_imgs)
        assign_adaptive_params(app_mean_list, self.CCN.crowd_decoder)

        # generating the output image
        if len(self.gpus) > 1:
            density_map = self.CCN.module.crowd_decoder(crow_img)
        else:
            density_map = self.CCN.crowd_decoder(crow_img)

        self.loss_mse = self.build_loss(
            density_map.squeeze(), gt_map.squeeze())

        return density_map

    def test(self, crow_imgs, gui_imgs, gt_map):

        # set the model in eval mode
        self.eval()

        app_mean_list = []
        for bid in range(gui_imgs.shape[0]):
            batch_gui_img = gui_imgs[bid, :, :, :]
            split_gui_imgs = batch_gui_img.split(dim=0, split_size=3)
            split_gui_imgs = [gui_img.to(device).unsqueeze(0)
                              for gui_img in split_gui_imgs]
            app_latent = [
                self.CCN.guiding_mlp(
                    self.CCN.guiding_encoder(img)) for img in split_gui_imgs]

            # mean the features of the unlabeled images
            app_mean = torch.mean(torch.cat(app_latent), dim=0).unsqueeze(0)
            app_mean_list.append(app_mean)

        app_mean_list = torch.mean(
            torch.cat(app_mean_list),
            dim=0).unsqueeze(0)

        # crowd encoder output
        crow_img = self.CCN.crowd_encoder(crow_imgs)

        if len(self.gpus) > 1:
            assign_adaptive_params(app_mean_list, self.CCN.module.crowd_decoder)
        else:
            assign_adaptive_params(app_mean_list, self.CCN.crowd_decoder)

        # generating the output image
        if len(self.gpus) > 1:
            density_map = self.CCN.module.crowd_decoder(crow_img)
        else:
            density_map = self.CCN.crowd_decoder(crow_img)

        self.loss_mse = self.build_loss(
            density_map.squeeze(), gt_map.squeeze())

        # reset the model in train mode
        self.train()

        # returning the generated density map along with the loss
        return density_map

    def compute_k_mean(self, dataloader, dataset=None):
        self.eval()
        app_mean = None
        if dataset not in SINGLE_FOLDER_DATASETS:
            app_latent = []
            for idx, data in enumerate(dataloader):
                img, _, original_image = data
                img = img.to(device)
                img_latent = self.CCN.guiding_mlp(
                    self.CCN.guiding_encoder(img))
                app_latent.append(img_latent.detach().cpu().numpy())
            app_mean = torch.mean(
                torch.from_numpy(
                    np.array(app_latent)).squeeze(0),
                dim=0).to(device)

        else:
            # Here the dataloader contains only the unlabeled images: B x H x
            # H x W
            app_latent = []
            for data in dataloader:
                batch_gui_img = data
                split_gui_imgs = batch_gui_img.split(dim=0, split_size=3)
                split_gui_imgs = [gui_img.to(device).unsqueeze(
                    0) for gui_img in split_gui_imgs]

                app_latent = [self.CCN.guiding_mlp(self.CCN.guiding_encoder(
                    img)).detach().cpu().numpy() for img in split_gui_imgs]

                # mean the features of the unlabeled images
                app_mean = torch.mean(
                    torch.from_numpy(
                        np.array(app_latent)).squeeze(0),
                    dim=0).to(device)

        return app_mean

    def get_test_prediction(self, app_mean, crow_imgs, didx):
        crow_img = self.CCN.crowd_encoder(crow_imgs)
        if didx == 0:
            assign_adaptive_params(app_mean, self.CCN.crowd_decoder)
        density_map = self.CCN.crowd_decoder(crow_img)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
