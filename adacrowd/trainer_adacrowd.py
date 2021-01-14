import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

from config.adacrowd import cfg
from misc.utils import (AverageMeter, Timer, logger, print_summary,
                        update_model, vis_results)
from models.cc_adacrowd import CrowdCounterAdaCrowd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Trainer class to initiate the training
class Trainer_AdaCrowd():
    """
    Params:
    dataloader: The dataloader the of the specified dataset
    cfg_data: The configuration data for training the model
    pwd: The present working directory
    """

    def __init__(self, dataloader, cfg_data, pwd):
        self.cfg_data = cfg_data
        self.dataloader = dataloader
        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = CrowdCounterAdaCrowd(
            gpus=cfg.GPU_ID,
            model_name=self.net_name,
            num_gbnnorm=cfg.NUM_GBNNORM).to(device)

        # Optimizer (Adam with weight decay of 1e-4)
        self.optimizer = optim.Adam(
            self.net.CCN.parameters(),
            lr=cfg.LR,
            weight_decay=1e-4)

        # Learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=cfg.NUM_EPOCH_LR_DECAY,
            gamma=cfg.LR_DECAY)

        # Stats to determine the best performing model
        self.train_record = {
            'best_mae': 1e20,
            'best_mse': 1e20,
            'best_model_name': ''}
        self.timer = {
            'iter time': Timer(),
            'train time': Timer(),
            'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0

        # Moving the dataloader inside the epoch iteration, so the crowd
        # images change for each iteration for a particular scene
        self.train_loader, self.val_loader, self.restore_transform = self.dataloader(
            k_shot=cfg.K_SHOT, num_scenes=cfg.NUM_SCENES)

        # Use this incase you need to resume the training of already trained
        # partial model
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        # Logging all the training characteristics
        self.writer, self.log_txt = logger(
            self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):

        # Iterating through the epoch count
        for epoch in range(self.epoch, cfg.MAX_EPOCH):

            # Use the dataloader (shuffling the unlabeled images per scene in
            # each epoch)
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()

            # Tracking training time
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # Validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                self.validate()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self):
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map, gui_imgs = data

            img = Variable(img).to(device)
            gt_map = Variable(gt_map).to(device)

            self.optimizer.zero_grad()
            pred_map = self.net(img, gui_imgs, gt_map)
            loss = self.net.loss
            loss.backward()

            # clipping grad norm with clip value = 1
            clip_grad_norm_(self.net.parameters(), cfg.GRAD_CLIP)
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print(
                    '[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' %
                    (self.epoch + 1, i + 1, loss.item(),
                        self.optimizer.param_groups[0]['lr'] *
                        10000,
                        self.timer['iter time'].diff))
                print(
                    '        [cnt: gt: %.1f pred: %.2f]' %
                    (gt_map[0].sum().data /
                     self.cfg_data.LOG_PARA,
                     pred_map[0].sum().data /
                        self.cfg_data.LOG_PARA))

    def validate(self):

        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, gui_imgs = data

            with torch.no_grad():
                img = Variable(img).to(device)
                gt_map = Variable(gt_map).to(device)

                pred_map = self.net.test(img, gui_imgs, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                if vi == 0:
                    vis_results(
                        self.exp_name,
                        self.epoch,
                        self.writer,
                        self.restore_transform,
                        img,
                        pred_map,
                        gt_map)

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,
                                         self.optimizer,
                                         self.scheduler,
                                         self.epoch,
                                         self.i_tb,
                                         self.exp_path,
                                         self.exp_name,
                                         [mae,
                                          mse,
                                          loss],
                                         self.train_record,
                                         self.log_txt)
        print_summary(self.exp_name, [mae, mse, loss], self.train_record)
