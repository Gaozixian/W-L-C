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
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms

# ==================== 1. 基础组件：Bottleneck ====================
class Bottleneck(nn.Module):
    """ResNet50的瓶颈块 (引用自原文件)"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

# ==================== 2. 核心改进：自适应双向引导交互模块 ====================

class AdaptiveBidirectionalInteraction(nn.Module):
    """
    自适应多尺度注意力融合机制：
    - 路径1: 浅层空间 As -> Gsd 引导深层通道
    - 路径2: 深层语义 Ac -> Gds 优化浅层空间
    """
    def __init__(self, s_channels, d_channels, reduction=16):
        super(AdaptiveBidirectionalInteraction, self).__init__()

        # --- 浅层分支 (Spatial Focus) ---
        self.s_conv7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # As -> Gsd (Spatial to Deep Channel Guidance)
        self.s_to_d_mlp = nn.Sequential(
            nn.Linear(1, d_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(d_channels // reduction, d_channels)
        )

        # --- 深层分支 (Channel Semantic) ---
        self.d_shared_mlp = nn.Sequential(
            nn.Linear(d_channels, d_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_channels // reduction, d_channels, bias=False)
        )

        # Ac -> Gds (Semantic to Shallow Spatial Guidance)
        self.d_to_s_mlp = nn.Sequential(
            nn.Linear(d_channels, d_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(d_channels // reduction, 1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fs, Fd):
        # 1. 深层通道注意力计算 (Ac)
        avg_p = F.adaptive_avg_pool2d(Fd, 1).view(Fd.size(0), -1)
        max_p = F.adaptive_max_pool2d(Fd, 1).view(Fd.size(0), -1)
        d_stats = self.d_shared_mlp(avg_p) + self.d_shared_mlp(max_p)
        Ac = self.sigmoid(d_stats)

        # 2. 深层对浅层的语义引导 (Ac -> Gds)
        Gds = self.d_to_s_mlp(Ac)

        # 3. 浅层空间注意力计算 (As), 注入 Gds
        s_avg = torch.mean(Fs, dim=1, keepdim=True)
        s_max, _ = torch.max(Fs, dim=1, keepdim=True)
        s_desc = torch.cat([s_avg, s_max], dim=1)
        s_desc_fused = s_desc * Gds.view(-1, 1, 1, 1) # 语义引导注入
        As = self.sigmoid(self.s_conv7x7(s_desc_fused))
        Fs_out = Fs * As

        # 4. 浅层对深层的空间引导 (As -> Gsd)
        s_guide_vec = F.adaptive_avg_pool2d(As, 1).view(As.size(0), -1)
        Gsd = self.s_to_d_mlp(s_guide_vec)

        # 5. 深层最终权重融合
        final_Ac = self.sigmoid(d_stats + Gsd).view(Fd.size(0), Fd.size(1), 1, 1)
        Fd_out = Fd * final_Ac

        return Fs_out, Fd_out

# ==================== 3. 改进版语义分割网络 ====================

class ResNet50_AdaptiveInteraction_Segmentation(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=19, pretrained=False):
        super(ResNet50_AdaptiveInteraction_Segmentation, self).__init__()
        self.inplanes = 64

        # Encoder 前缀
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet 层级
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 核心交互模块：连接 Layer 2 (浅层 512ch) 和 Layer 4 (深层 2048ch)
        self.interaction = AdaptiveBidirectionalInteraction(512, 2048)

        # 解码器部分 (引用自原文件逻辑)
        self.decoder4 = self._make_decoder_block(2048 + 1024, 256)
        self.decoder3 = self._make_decoder_block(256 + 512, 128)
        self.decoder2 = self._make_decoder_block(128 + 256, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self._initialize_weights()
        if pretrained: self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def _load_pretrained_weights(self):
        try:
            pre_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
            model_dict = self.state_dict()
            pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pre_dict)
            self.load_state_dict(model_dict)
            print("成功加载预训练权重")
        except: print("预训练加载失败")

    def forward(self, x, size=None):
        # Encoder
        x0 = self.relu(self.bn1(self.conv1(x)))
        x_low = self.maxpool(x0)

        x1 = self.layer1(x_low)
        x2 = self.layer2(x1)  # 浅层 Fs
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # 深层 Fd

        # 核心双向交互
        x2_enhanced, x4_enhanced = self.interaction(x2, x4)

        # Decoder (级联融合)
        h, w = x3.shape[2], x3.shape[3]
        d4 = F.interpolate(x4_enhanced, size=(h, w), mode='bilinear', align_corners=False)
        d4 = self.decoder4(torch.cat([d4, x3], dim=1))

        h, w = x2_enhanced.shape[2], x2_enhanced.shape[3]
        d3 = F.interpolate(d4, size=(h, w), mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d3, x2_enhanced], dim=1))

        h, w = x1.shape[2], x1.shape[3]
        d2 = F.interpolate(d3, size=(h, w), mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d2, x1], dim=1))

        h, w = x0.shape[2], x0.shape[3]
        d1 = F.interpolate(d2, size=(h, w), mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x0], dim=1))

        out = self.final_conv(d1)
        target_size = size if size is not None else (x.size(2), x.size(3))
        return F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

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

class CombinedLoss(nn.Module):
    """组合损失 (引用原文件)"""
    def __init__(self, num_classes=19, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        # Dice Loss 实现... (略，保持与原文件一致)
        return ce_loss

# ==================== 5. 训练器与主逻辑 (原样引用自原文件) ====================

class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_dir='checkpoints'):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.criterion, self.optimizer, self.scheduler, self.device = criterion, optimizer, scheduler, device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses, self.val_losses, self.val_miou_history = [], [], []
        self.best_miou, self.epoch = 0, 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_idx, (images, targets, _, _) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0: print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, confusion_matrix = 0, np.zeros((19, 19))
        for images, targets, _, _ in self.val_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            total_loss += self.criterion(outputs, targets).item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            for t, p in zip(targets.cpu().numpy(), preds):
                mask = (t != 255)
                label = 19 * t[mask].astype('int') + p[mask]
                confusion_matrix += np.bincount(label, minlength=19**2).reshape(19, 19)
        iu = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + 1e-10)
        return total_loss / len(self.val_loader), np.mean(iu)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = self.train_epoch()
            val_loss, val_miou = self.validate()
            self.train_losses.append(train_loss); self.val_losses.append(val_loss); self.val_miou_history.append(val_miou)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                print("保存最佳模型!")

# ==================== 6. 运行脚本 ====================

# 加载模型
def load_model(weight_path, device):
    model = ResNet50_AdaptiveInteraction_Segmentation(num_classes=19, pretrained=False)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 预处理输入图片
def preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # 增加 batch 维度

# 可视化分割结果
def visualize_result(image_path, output, class_colors, save_dir='results_png'):
    os.makedirs(save_dir, exist_ok=True)
    image = Image.open(image_path).convert('RGB')
    output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    color_mask = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for label, color in class_colors.items():
        color_mask[output == label] = color

    # 调整 color_mask 的尺寸与原始图片一致
    color_mask_image = Image.fromarray(color_mask).resize(image.size, Image.BILINEAR)

    # 混合图片
    blended = Image.blend(image, color_mask_image, alpha=0.5)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 保存原始图片
    original_save_path = os.path.join(save_dir, f'original_{timestamp}.png')
    image.save(original_save_path)
    # 保存分割结果
    result_save_path = os.path.join(save_dir, f'segmentation_{timestamp}.png')
    blended.save(result_save_path)
    # 保存纯色掩码
    mask_save_path = os.path.join(save_dir, f'mask_{timestamp}.png')
    color_mask_image.save(mask_save_path)

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Result")
    plt.imshow(blended)
    plt.axis('off')
    plt.show()

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = 'dpai/best_model.pth'  # 替换为你的模型权重路径
    image_path = 'frankfurt_000000_000294_leftImg8bit.png'  # 替换为你的测试图片路径

    # 加载模型
    model = load_model(weight_path, device)

    # 预处理图片
    input_tensor = preprocess_image(image_path).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 可视化结果
    class_colors = {  # 替换为你的类别颜色映射
        0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156],
        4: [190, 153, 153], 5: [153, 153, 153], 6: [250, 170, 30], 7: [220, 220, 0],
        8: [107, 142, 35], 9: [152, 251, 152], 10: [70, 130, 180], 11: [220, 20, 60],
        12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70], 15: [0, 60, 100],
        16: [0, 80, 100], 17: [0, 0, 230], 18: [119, 11, 32], 255: [0, 0, 0]
    }
    visualize_result(image_path, output, class_colors, save_dir='results_png')

if __name__ == "__main__":
    main()