import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import random
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime


# ==================== 4. 数据处理与损失函数 (原样引用自原文件) ====================
class CityscapesDataset(Dataset):
    """Cityscapes数据集加载器"""
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    CLASS_COLORS = {0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156],
                    4: [190, 153, 153], 5: [153, 153, 153], 6: [250, 170, 30], 7: [220, 220, 0],
                    8: [107, 142, 35], 9: [152, 251, 152], 10: [70, 130, 180], 11: [220, 20, 60],
                    12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70], 15: [0, 60, 100],
                    16: [0, 80, 100], 17: [0, 0, 230], 18: [119, 11, 32], 255: [0, 0, 0]}

    LABEL_ID_MAPPING = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
                        23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def __init__(self, root, split='train', transform=None, target_size=(1024, 512)):
        self.root, self.split, self.transform, self.target_size = root, split, transform, target_size
        self.images = self._get_image_paths()
        self.targets = self._get_target_paths()

    def _get_image_paths(self):
        image_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        images = []
        for city in os.listdir(image_dir):
            city_dir = os.path.join(image_dir, city)
            for img_name in os.listdir(city_dir):
                if img_name.endswith('_leftImg8bit.png'): images.append(os.path.join(city_dir, img_name))
        return sorted(images)

    def _get_target_paths(self):
        target_dir = os.path.join(self.root, 'gtFine', self.split)
        targets = []
        for img_path in self.images:
            img_name = os.path.basename(img_path)
            base_name = img_name.replace('_leftImg8bit.png', '')
            city = os.path.basename(os.path.dirname(img_path))
            target_name = f'{base_name}_gtFine_labelIds.png'
            targets.append(os.path.join(target_dir, city, target_name))
        return sorted(targets)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.targets[idx])
        orig_size = image.size[::-1]
        image = image.resize(self.target_size, Image.BILINEAR)
        target = target.resize(self.target_size, Image.NEAREST)
        if self.transform: image, target = self.transform(image, target)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        target = self._remap_labels(np.array(target))
        return image, target, orig_size, self.images[idx]

    def _remap_labels(self, target):
        remapped = np.full_like(target, 255)
        for old, new in self.LABEL_ID_MAPPING.items(): remapped[target == old] = new
        return torch.from_numpy(remapped).long()

class SimpleTransform:
    def __init__(self, crop_size=(512, 512), flip_prob=0.5):
        self.crop_size, self.flip_prob = crop_size, flip_prob
    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = image.size
        cw, ch = self.crop_size
        left = random.randint(0, w - cw)
        top = random.randint(0, h - ch)
        image = image.crop((left, top, left + cw, top + ch))
        target = target.crop((left, top, left + cw, top + ch))
        return image, target