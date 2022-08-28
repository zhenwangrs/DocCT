import argparse
import glob
import logging
import os
import re
import warnings
from distutils import dist

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from PIL import Image
from munch import Munch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from mae import ViTMAEConfig, ViTMAEForPreTraining

warnings.filterwarnings("ignore")

CLASSES = [
    'artist_',
    'buildings_',
    'economy_',
    'education_',
    # 'industry', 'entertainment', 'environment',
    # 'fashion', 'food', 'geography', 'health', 'history',
    # 'law', 'marriage', 'politics', 'religion', 'sports', 'science',
    # 'transport'
]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, feature_extractor):
        self.data_path = data_path
        self.img_paths, self.labels = self.find_images()
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image, self.labels[idx], img_path

    def __len__(self):
        return len(self.labels)

    def find_images(self):
        img_dirs = os.listdir(self.data_path)
        img_paths = []
        labels = []
        for img_dir in img_dirs:
            if img_dir in CLASSES:
                for root, dirs, files in os.walk(os.path.join(self.data_path, img_dir)):
                    for file in files:
                        img_paths.append(os.path.join(root, file))
                        labels.append(CLASSES.index(img_dir))
        return img_paths, labels

    def collate_fn(self, batch):
        images = []
        labels = []
        image_paths = []
        for img, label, img_path in batch:
            images.append(img)
            labels.append(label)
            image_paths.append(img_path)
        image_feats = self.feature_extractor(images=images, return_tensors="pt")['pixel_values']
        labels = torch.tensor(labels)
        return image_feats, labels, image_paths


class Mae(nn.Module):
    def __init__(self, docmae_state_dict):
        super().__init__()
        self.docmae_state_dict = docmae_state_dict
        config = ViTMAEConfig(
            image_size=640,
            patch_size=20,
            mask_ratio=0,
        )
        self.mae = ViTMAEForPreTraining(config)
        self.load_from_DocMae()
        self.cls = nn.Linear(512, len(CLASSES))

    def forward(self, pixel_values):
        output = self.mae(pixel_values, return_dict=True)
        output = output.hidden_states
        output = torch.mean(output, dim=1)
        return self.cls(output)

    def load_from_DocMae(self):
        docmae_state_dict = self.docmae_state_dict
        mae_state_dict = self.mae.state_dict()
        for key in mae_state_dict:
            if f'module.mae.{key}' in docmae_state_dict.keys():
                mae_state_dict[key] = docmae_state_dict[f'module.mae.{key}']
            elif f'mae.{key}' in docmae_state_dict.keys():
                mae_state_dict[key] = docmae_state_dict[f'mae.{key}']
            else:
                raise KeyError(f'{key} not found in {self.ckp_pth}')
        self.mae.load_state_dict(mae_state_dict)


def train(model, cfg):
    logger = build_logger(save_path=os.path.join(cfg.log_dir, 'mae_decoder_eval.log'))
    logger.info('categories: ' + str(CLASSES))
    logger.info('start fine-tuning')
    logger.info(cfg.pkl_file)

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
    feature_extractor.size = 640

    batch_accum = 8

    train_dataset = MyDataset(cfg.eval.train_dir, feature_extractor)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        # shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        sampler=train_sampler
    )

    val_dataset = MyDataset(cfg.eval.val_dir, feature_extractor)
    # val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=val_dataset.collate_fn,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
        # sampler=val_sampler
    )

    optim = AdamW(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = CosineAnnealingLR(optim, T_max=cfg.eval.epochs)
    # scheduler = ExponentialLR(optim, gamma=0.8)

    # optim = AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05)
    # scheduler = ConstantLR(optim, factor=1, total_iters=0)

    scaler = GradScaler()
    loss = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(cfg.eval.epochs):
        train_sampler.set_epoch(epoch)
        for index, (pixel_values, labels, _) in enumerate(tqdm(train_loader), start=1):
            pixel_values = pixel_values.to(cfg.training.device, non_blocking=True)
            labels = labels.to(cfg.training.device, non_blocking=True)
            with autocast():
                cls_output = model(pixel_values)
                l = loss(cls_output, labels)
            scaler.scale(l).backward()
            if index % batch_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
            torch.cuda.empty_cache()

        if cfg.distributed.local_rank == 0:
            torch.save(model, './ckp/mae640_cls.ckp')
            acc, f1, p, r = test(model, val_loader, cfg)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model, './ckp/mae640_cls_best.ckp')
            logger.info(
                f'epoch {epoch}, val acc: {acc}, f1: {f1}, p: {p}, r: {r}, best_f1: {best_f1}, lr: {scheduler.get_last_lr()[0]}')

        scheduler.step()


def test(mae, test_loader, cfg):
    with torch.no_grad():
        mae.eval()
        gold_ans = []
        pred_ans = []

        for pixel_values, labels, image_paths in tqdm(test_loader):
            pixel_values = pixel_values.to(cfg.training.device, non_blocking=True)
            labels = labels.to(cfg.training.device, non_blocking=True)

            cls_output = mae(pixel_values)
            cls_output = torch.argmax(cls_output, dim=1)

            [gold_ans.append(label.item()) for label in labels]
            [pred_ans.append(label.item()) for label in cls_output]

        # print(gold_ans)
        # print(pred_ans)
        acc = np.around(accuracy_score(gold_ans, pred_ans), 4)
        f1 = np.around(f1_score(gold_ans, pred_ans, average='macro'), 4)
        p = np.around(precision_score(gold_ans, pred_ans, average='macro'), 4)
        r = np.around(recall_score(gold_ans, pred_ans, average='macro'), 4)

        mae.train()
        return acc, f1, p, r


def build_logger(save_path='DocBERT_eval.log'):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('DocBERT_eval')
    handler = logging.FileHandler(save_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def doc_bert_eval(rank, args):
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, 'r', encoding='utf8').read()))
    cfg.training.device = torch.device('cuda:{}'.format(rank))
    cfg.distributed.local_rank = rank
    cfg.training.model_save_name = "DocBERT{}_epoch{}.pkl"
    cfg.pkl_dir = './work_dir/pkl/'
    cfg.log_dir = './'

    init_distributed(rank, cfg)

    # r = re.compile(r'^.*DocMae640_epoch(\d+)\.pkl$')
    # pkl_files = [os.path.basename(f) for f in glob.glob(os.path.join(cfg.pkl_dir, '*.pkl'))]
    # pkl_files = [pkl_file for pkl_file in pkl_files if r.match(pkl_file) is not None]
    # pkl_files = sorted(pkl_files)
    # pkl_file = pkl_files[-1]
    
    cfg.pkl_file = 'DocMae640_epoch30.pkl'
    docmae_state_dict = torch.load(os.path.join(cfg.pkl_dir, cfg.pkl_file))
    model = Mae(docmae_state_dict=docmae_state_dict).to(cfg.training.device)

    model = DistributedDataParallel(model, device_ids=[cfg.distributed.local_rank], find_unused_parameters=True)
    train(model, cfg)


def get_args():
    parser = argparse.ArgumentParser('DocMAE')
    parser.add_argument('--config', default='./mae_pretrain.yaml', type=str, help='path to config file')
    args = parser.parse_args()
    return args


def init_distributed(rank, cfg):
    gpus = torch.cuda.device_count()
    dist.init_process_group(
        backend=cfg.distributed.backend,
        init_method=cfg.distributed.init_method,
        world_size=cfg.distributed.world_size * gpus,
        rank=rank)


def run_multi(fn, world_size, args):
    mp.spawn(fn, args=(args,), nprocs=world_size, join=True)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = get_args()
    run_multi(doc_bert_eval, torch.cuda.device_count(), args=args)
