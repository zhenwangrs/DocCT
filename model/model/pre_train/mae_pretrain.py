import argparse
import glob
import re

import numpy as np
import torch
import torch.nn as nn
import yaml
from munch import Munch
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoFeatureExtractor
from transformers import ViTMAEForPreTraining, ViTMAEConfig

from utils import *


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, feature_extractor):
        self.cfg = cfg
        self.feature_extractor = feature_extractor
        self.img_paths = []
        self.data_paths = cfg.data.img_dirs
        for data_path in self.data_paths:
            self.img_paths.extend(get_image_paths(data_path))
        if cfg.data.max_img_num > 0:
            self.img_paths = self.img_paths[:cfg.data.max_img_num]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path)
            image = image_random_crop(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(e)
            return Image.open(self.img_paths[0])

    def __len__(self):
        return len(self.img_paths)

    def collate_fn(self, batch):
        images = []
        for img in batch:
            images.append(img)
        image_feats = self.feature_extractor(images=images, return_tensors="pt")['pixel_values'].cuda(
            non_blocking=True)
        return image_feats


class Mae(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = ViTMAEConfig(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            mask_ratio=cfg.mask_ratio,
        )
        self.mae = ViTMAEForPreTraining(config)
        if cfg.use_mae:
            self.load_mae_from_pretrained()
        if cfg.use_dit:
            self.load_dit_from_pretrained()

    def load_mae_from_pretrained(self):
        mae_pretrained = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        mae_pretrained_state_dict = mae_pretrained.state_dict()
        mae_state_dict = self.mae.state_dict()
        for key in mae_state_dict.keys():
            if key in mae_pretrained_state_dict:
                if mae_state_dict[key].shape == mae_pretrained_state_dict[key].shape:
                    mae_state_dict[key] = mae_pretrained_state_dict[key]
        self.mae.load_state_dict(mae_state_dict)

    def load_dit_from_pretrained(self):
        dit_pretrained = AutoModel.from_pretrained("microsoft/dit-base")
        dit_pretrained_state_dict = dit_pretrained.state_dict()
        mae_state_dict = self.mae.state_dict()
        for key in mae_state_dict.keys():
            dit_key = key.replace('vit.', '')
            if dit_key in dit_pretrained_state_dict:
                if mae_state_dict[key].shape == dit_pretrained_state_dict[dit_key].shape:
                    mae_state_dict[key] = dit_pretrained_state_dict[dit_key]
        self.mae.load_state_dict(mae_state_dict)

    def forward(self, pixel_values):
        output = self.mae(pixel_values)
        return output.loss


def pretrain(model, cfg):
    img_dirs = cfg.data.img_dirs
    logger.info('start pre-training')
    if cfg.data.clean:
        for img_dir in img_dirs:
            clean_img(img_dir, logger)

    scaler = GradScaler()
    optim = AdamW(params=model.parameters(),
                  lr=cfg.optimizer.lr,
                  betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                  weight_decay=cfg.optimizer.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=cfg.training.epochs, eta_min=cfg.optimizer.eta_min)
    for epoch in range(cfg.training.start_epoch-1):
        scheduler.step()

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model.feature_extractor)
    feature_extractor.size = cfg.model.image_size

    train_dataset = PretrainDataset(cfg=cfg, feature_extractor=feature_extractor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        drop_last=cfg.dataloader.drop_last,
        num_workers=cfg.dataloader.num_workers,
        # pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        persistent_workers=cfg.dataloader.persistent_workers,
        collate_fn=train_dataset.collate_fn,
    )

    for epoch in range(cfg.training.start_epoch, cfg.training.epochs):
        logger.info('epoch {} start'.format(epoch))
        epoch_loss = []
        total_iters = len(train_loader)
        for step, pixel_values in enumerate(tqdm(train_loader), start=1):
            if cfg.training.use_amp:
                with autocast():
                    l = model(pixel_values)
                scaler.scale(l).backward()
                if step % cfg.dataloader.batch_accum == 0:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
            else:
                l = model(pixel_values)
                l.backward()
                if step % cfg.dataloader.batch_accum == 0:
                    optim.step()
                    optim.zero_grad()
            torch.cuda.empty_cache()
            loss_value = np.around(l.item(), 4)
            tensorboard_writer.add_scalar('loss', loss_value, epoch * len(train_loader) + step)
            epoch_loss.append(loss_value)
            if step % cfg.training.loss_print_interval == 0:
                avg_epoch_loss = np.around(np.mean(epoch_loss), 4)
                logger.info(f'epoch: [{epoch}/{cfg.training.epochs}], iter: [{step}/{total_iters}], batch_loss: {loss_value}, epoch_loss: {avg_epoch_loss}')
            if step % cfg.training.step_save_interval == 0:
                model_save_path = f'{cfg.pkl_dir}/{cfg.training.model_save_name}'.format(cfg.model.image_size, epoch)
                torch.save(model.state_dict(), model_save_path)

        if epoch % cfg.training.epoch_save_interval == 0:
            model_save_path = f'{cfg.pkl_dir}/{cfg.training.model_save_name}'.format(cfg.model.image_size, epoch)
            torch.save(model.state_dict(), model_save_path)

        avg_epoch_loss = np.around(np.mean(epoch_loss), 4)
        logger.info(f'epoch: {epoch}, epoch_avg_loss: {avg_epoch_loss}, lr: {np.around(scheduler.get_last_lr(), 6)}')
        tensorboard_writer.add_scalar('epoch_loss', avg_epoch_loss, epoch)
        scheduler.step()


def get_args():
    parser = argparse.ArgumentParser('DocMAE')
    parser.add_argument('--config', default='./mae_pretrain.yaml', type=str, help='path to config file')
    parser.add_argument('--work_dir', default='./work_dir/', type=str, help='path to config file')
    parser.add_argument('--resume', default=True, type=bool, help='if resume from last checkpoint')
    # parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    # parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, 'r', encoding='utf8').read()))

    work_dir = args.work_dir
    log_dir = os.path.join(work_dir, 'log')
    pkl_dir = os.path.join(work_dir, 'pkl')
    tensorboard_dir = os.path.join(work_dir, 'tensorboard')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    cfg.work_dir = work_dir
    cfg.log_dir = log_dir
    cfg.pkl_dir = pkl_dir
    cfg.tensorboard_dir = tensorboard_dir

    if not args.resume:
        model = Mae(cfg.model).to(cfg.training.device)
        model_save_path = f'{cfg.pkl_dir}/{cfg.training.model_save_name}'.format(cfg.model.image_size, 0)
        torch.save(model.state_dict(), model_save_path)
        cfg.start_epoch = 1
    else:
        r = re.compile(r'^.*_epoch(\d+)\.pkl$')
        pkl_files = [os.path.basename(f) for f in glob.glob(os.path.join(cfg.pkl_dir, '*.pkl'))]
        pkl_files = sorted(pkl_files, key=lambda x: int(r.match(x).group(1)))
        pkl_file = pkl_files[-1]
        model = Mae(cfg.model).to(cfg.training.device)
        model.load_state_dict(torch.load(os.path.join(cfg.pkl_dir, pkl_file)))
        # model = torch.load(os.path.join(cfg.pkl_dir, pkl_file))
        last_epoch = pkl_file.split('_')[-1].split('.')[0]
        last_epoch = int(last_epoch.replace('epoch', ''))
        cfg.training.start_epoch = last_epoch + 1

    logger = build_logger(save_path=os.path.join(cfg.log_dir, 'pretrain.log'), task='DocMAE_Pretrain')
    tensorboard_writer = SummaryWriter(cfg.tensorboard_dir)  # tensorboard --logdir=./path/to/log --port 8123
    logger.info(f'Starting from epoch {cfg.training.start_epoch}')

    model = model.to(cfg.training.device)
    pretrain(model, cfg)
