import logging
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from munch import Munch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, ConstantLR, CosineAnnealingLR
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoFeatureExtractor, ViTMAEModel
from transformers import ViTMAEConfig

warnings.filterwarnings("ignore")
# torch.backends.cudnn.benchmark = True



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
        return image, self.labels[idx]

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
        for img, label in batch:
            images.append(img)
            labels.append(label)
        image_feats = self.feature_extractor(images=images, return_tensors="pt")['pixel_values']
        labels = torch.tensor(labels)
        return image_feats, labels


class Mae(nn.Module):
    def __init__(self, ckp_pth):
        super().__init__()
        self.ckp_pth = ckp_pth

        config = ViTMAEConfig(
            image_size=640,
            patch_size=20,
            mask_ratio=0,
        )
        self.mae = ViTMAEModel(config)
        self.load_from_DocMae()

        # docmae672_pretrained = torch.load(self.ckp_pth)
        # self.mae = docmae672_pretrained.mae.vit

        self.cls = nn.Linear(768, len(CLASSES))

    def forward(self, pixel_values):
        output = self.mae(pixel_values)
        output = output['last_hidden_state']
        output = torch.mean(output, dim=1)
        return self.cls(output)

    def load_from_DocMae(self):
        docmae_state_dict = torch.load(self.ckp_pth)
        mae_state_dict = self.mae.state_dict()
        for key in mae_state_dict:
            if f'module.mae.vit.{key}' in docmae_state_dict.keys():
                mae_state_dict[key] = docmae_state_dict[f'module.mae.vit.{key}']
            elif f'mae.vit.{key}' in docmae_state_dict.keys():
                mae_state_dict[key] = docmae_state_dict[f'mae.vit.{key}']
            else:
                raise KeyError(f'{key} not found in {self.ckp_pth}')
        self.mae.load_state_dict(mae_state_dict)


def train(model, total_epoch):
    logger.info('start fine-tuning')

    optim = AdamW(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = CosineAnnealingLR(optim, T_max=total_epoch)
    # scheduler = ExponentialLR(optim, gamma=0.8)

    # optim = AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05)
    # scheduler = ConstantLR(optim, factor=1, total_iters=0)

    scaler = GradScaler()
    loss = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(total_epoch):
        for index, (pixel_values, labels) in enumerate(tqdm(train_loader), start=1):
            pixel_values = pixel_values.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with autocast():
                cls_output = model(pixel_values)
                l = loss(cls_output, labels)
            scaler.scale(l).backward()
            if index % batch_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
            # cls_output = model(pixel_values)
            # l = loss(cls_output, labels)
            # l.backward()
            # optim.step()
            # optim.zero_grad()
            torch.cuda.empty_cache()
        torch.save(model, './ckp/mae640_cls.ckp')
        acc, f1, p, r = test(model, val_loader)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model, './ckp/mae640_cls_best.ckp')
        logger.info(f'epoch {epoch}, val acc: {acc}, f1: {f1}, p: {p}, r: {r}, best_f1: {best_f1}, lr: {scheduler.get_last_lr()[0]}')
        # acc, f1, p, r = test(model, test_loader)
        # logger.info(f'test acc: {acc}, f1: {f1}, p: {p}, r: {r}')
        scheduler.step()


def test(mae, test_loader):
    with torch.no_grad():
        mae.eval()
        gold_ans = []
        pred_ans = []

        for pixel_values, labels in tqdm(test_loader):
            pixel_values = pixel_values.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            cls_output = mae(pixel_values)
            cls_output = torch.argmax(cls_output, dim=1)

            [gold_ans.append(label.item()) for label in labels]
            [pred_ans.append(label.item()) for label in cls_output]

        print(gold_ans)
        print(pred_ans)
        acc = np.around(accuracy_score(gold_ans, pred_ans), 4)
        f1 = np.around(f1_score(gold_ans, pred_ans, average='macro'), 4)
        p = np.around(precision_score(gold_ans, pred_ans, average='macro'), 4)
        r = np.around(recall_score(gold_ans, pred_ans, average='macro'), 4)

        mae.train()
        return acc, f1, p, r


def build_logger(save_path='log.txt'):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('MAE')
    handler = logging.FileHandler(save_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


if __name__ == '__main__':
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

    logger = build_logger()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
    feature_extractor.size = 640

    batch_accum = 8
    train_dataset = MyDataset('E:/Research/DocCT/data/train', feature_extractor)
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=4,
                              prefetch_factor=2,
                              collate_fn=train_dataset.collate_fn)
    val_dataset = MyDataset('E:/Research/DocCT/data/val', feature_extractor)
    val_loader = DataLoader(val_dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=4,
                            prefetch_factor=2,
                            collate_fn=val_dataset.collate_fn)
    test_dataset = MyDataset('E:/Research/DocCT/data/test', feature_extractor)
    test_loader = DataLoader(test_dataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=4,
                             prefetch_factor=2,
                             collate_fn=test_dataset.collate_fn)

    logger.info('categories: ' + str(CLASSES))
    cfg = Munch.fromDict(yaml.safe_load(open('mae_pretrain.yaml', 'r', encoding='utf8').read()))

    ckp_pth = './work_dir/pkl/DocMae640_epoch28.pkl'
    model = Mae(ckp_pth=ckp_pth).to(device)
    logger.info(f'ckp_path: {ckp_pth}')
    train(model, total_epoch=15)
    model = torch.load('./ckp/mae640_cls_best.ckp').to(device)

    # model = torch.load('./ckp/mae640_cls.ckp').to(device)

    acc, f1, p, r = test(model, test_loader)
    logger.info(f'test acc: {acc}, f1: {f1}, p: {p}, r: {r}')
