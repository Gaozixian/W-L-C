import math
import torch.utils.model_zoo as model_zoo
from backbone import Bottleneck, AdaptiveBidirectionalInteraction, CBAM
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # 核心交互模块：连接 Layer 2 (浅层 512ch) 和 Layer 4 (深层 2048ch)
        self.interaction_heavy = AdaptiveBidirectionalInteraction(512, 2048)
        self.interaction_light = AdaptiveBidirectionalInteraction(256, 1024)
        self.x0_enhance = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64)
        )

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
        x1 = self.cbam1(x1)
        x2 = self.layer2(x1)  # 浅层 Fs
        x2 = self.cbam2(x2)
        x3 = self.layer3(x2)
        x3 = self.cbam3(x3)
        x4 = self.layer4(x3)  # 深层 Fd
        x4 = self.cbam4(x4)

        # 核心双向交互
        x0_processed = self.x0_enhance(x0)
        x2_enhanced, x4_enhanced = self.interaction_heavy(x2, x4)
        x1_enhanced, x3_enhanced = self.interaction_light(x1, x3)
        x0_fused = self.relu(x0 + x0_processed)
        x1_fused = self.relu(x1 + x1_enhanced)
        x2_fused = self.relu(x2 + x2_enhanced)
        x3_fused = self.relu(x3 + x3_enhanced)
        x4_fused = self.relu(x4 + x4_enhanced)


        # Decoder (级联融合)
        h, w = x3.shape[2], x3.shape[3]
        d4 = F.interpolate(x4_fused, size=(h, w), mode='bilinear', align_corners=False)
        d4 = self.decoder4(torch.cat([d4, x3_fused], dim=1))

        h, w = x2_enhanced.shape[2], x2_enhanced.shape[3]
        d3 = F.interpolate(d4, size=(h, w), mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d3, x2_fused], dim=1))

        h, w = x1.shape[2], x1.shape[3]
        d2 = F.interpolate(d3, size=(h, w), mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d2, x1_fused], dim=1))

        h, w = x0.shape[2], x0.shape[3]
        d1 = F.interpolate(d2, size=(h, w), mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x0], dim=1))

        out = self.final_conv(d1)
        target_size = size if size is not None else (x.size(2), x.size(3))
        return F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)



class DiceLoss(nn.Module):
    """
    Dice Loss: 直接优化模型的 IoU/区域重合度，极大地缓解类别不平衡影响。
    """
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    def forward(self, pred, target):
        # 对模型的输出进行 Softmax 得到预测概率 (B, C, H, W)
        pred = F.softmax(pred, dim=1)
        
        # 1. 创建 Ignore Mask (去除 Ignore Index 边界，不让其参与计算)
        valid_mask = (target != self.ignore_index)
        target = target * valid_mask.long()
        
        # 2. 将真实的 Target 转换成 One-hot 编码格式 (B, C, H, W)
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # 3. 把无效的像素点剔除
        target_one_hot = target_one_hot * valid_mask.unsqueeze(1).float()
        pred = pred * valid_mask.unsqueeze(1).float()
        # 4. 计算交集和并集
        intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))  # 交集
        cardinality = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3)) # 并集
        
        # 5. 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Dice 的计算结果是预测与实际的重合度，因此 Loss 为 (1 - 重合度)
        return 1.0 - dice.mean()
class FocalLoss(nn.Module):
    """
    Focal Loss: 让模型在训练后期专注“难样本”（比如交通信号灯，细小的行人），不再过度更新简单的大背景样本。
    """
    def __init__(self, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    def forward(self, pred, target):
        # 1. 计算原始的交叉熵损失 
        logpt = -self.ce_loss(pred, target)
        
        # 2. 还原出概率值 pt
        pt = torch.exp(logpt)
        
        # 3. 施加 Focal 权重: (1 - pt)^gamma
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        
        # 过滤计算均值
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """
    高能效组合损失： Focal Loss 捕捉小目标困难样本 + Dice Loss 极大拉高网络对形态(轮廓)的拟合能力。
    这是在 Cityscapes 和医学分割上经常拿第一的标准化组合。
    """
    def __init__(self, num_classes=19, ignore_index=255, focal_w=1.0, dice_w=1.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal_w = focal_w
        self.dice_w = dice_w
    def forward(self, pred, target):
        f_loss = self.focal(pred, target)
        d_loss = self.dice(pred, target)
        
        # 你可以根据训练中后期的表现调整权重比例，最经典的起步是 1:1 双管齐下
        total_loss = (self.focal_w * f_loss) + (self.dice_w * d_loss)
        return total_loss


# ==================== 1. 空洞空间金字塔池化 (ASPP) 实现 ====================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积分支
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 三个空洞卷积分支
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
            
        # 全局池化分支
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # 结果拼接并用 1x1 最后融合
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# ==================== 2. DeepLabV3+ 解码器模块 ====================
class DeepLabV3PDecoder(nn.Module):
    def __init__(self, num_classes=19, low_level_channels=256, aspp_out_channels=256):
        super(DeepLabV3PDecoder, self).__init__()
        
        # 第一步：把极厚度（如256通道）的底级特征降维到48层，防止底层细节信息在Concat时“淹没”高层语义信息
        self.reduce_low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 第二步：将合并后的高级(256)和低级(48)特征通道(共304)进行卷积并输出类数
        self.classifier = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, high_level_features, low_level_features):
        low_level_features = self.reduce_low_level(low_level_features)
        
        # 直接上采样 ASPP 的输出尺寸到原本 x1 低级特征的长宽上
        high_level_features = F.interpolate(
            high_level_features, 
            size=low_level_features.shape[2:], 
            mode='bilinear', align_corners=False
        )
        
        # 通道级别合并并做最后的 3x3 解码推理
        concat_features = torch.cat([high_level_features, low_level_features], dim=1)
        return self.classifier(concat_features)

# ==================== 3. 组装成 DPAI 体系下的网络主体 ====================
class DPAI_DeepLabV3P_Segmentation(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=19, pretrained=False):
        super(DPAI_DeepLabV3P_Segmentation, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])     # x1 (通常提取自这里的 256ch 送出残差去当 low_level)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)    
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)    
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)    # x4 (这里的 2048ch 送向 ASPP)

        # 挂载了你核心创新点的特征引导
        self.interaction_heavy = AdaptiveBidirectionalInteraction(512, 2048)
        self.interaction_light = AdaptiveBidirectionalInteraction(256, 1024)

        # ASPP 采用标准膨胀率 [6, 12, 18]，如果你算力或者显存有溢出，可以换成小的 [4, 8, 12]
        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18])
        
        # 我们用经过 DPAI 强化过后的 x1_fused (256通道) 作为浅层边缘信息的补充传递给 Decoder
        self.decoder = DeepLabV3PDecoder(num_classes=num_classes, low_level_channels=256, aspp_out_channels=256)

        self._initialize_weights()
        # 这里预训练权重的加载逻辑同样按照之前 _load_pretrained_weights 直接粘过来即可使用
        # if pretrained: self._load_pretrained_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x, size=None):
        input_shape = x.shape[-2:]
        # ------ Encoder ------
        x0 = self.relu(self.bn1(self.conv1(x)))
        x_low = self.maxpool(x0)

        x1 = self.layer1(x_low)
        x2 = self.layer2(x1)  
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  

        # ------ DPAI 模块交互 ------
        x2_enhanced, x4_enhanced = self.interaction_heavy(x2, x4)
        x1_enhanced, x3_enhanced = self.interaction_light(x1, x3)

        x1_fused = self.relu(x1 + x1_enhanced) 
        x4_fused = self.relu(x4 + x4_enhanced)

        # ------ ASPP 模块获取大感受野高级别语义 ------
        # 使用 DPAI 强化后的 x4_fused （或者如果出于对比试验的目的，你可以在这里对比传直连的 x4 后的指标跌幅）
        aspp_features = self.aspp(x4_fused)

        # ------ DeepLabV3+ 解码拼接 ------
        # 使用 DPAI 强化后的浅层 x1_fused 当做补偿轮廓
        out = self.decoder(high_level_features=aspp_features, low_level_features=x1_fused)
        
        # ------ 最后一次性暴力插值回原始画面分辨率大小以算 Loss ------
        target_size = size if size is not None else input_shape
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

        return out

# ==================== 4. 【消融专用基线】CBAM + DeepLabV3+ ====================
class CBAM_DeepLabV3P_Segmentation(nn.Module):
    """
    用来对比剥离 DPAI 交互机制后效果如何。
    在 ResNet 每层之后加上经典的序列注意力 CBAM，随后仅交由标准的 ASPP 和 Decoder 进行重建。
    """
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=19, pretrained=False):
        super(CBAM_DeepLabV3P_Segmentation, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])     
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)    
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)    
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)    

        # 挂载老牌对比基线注意力机制 CBAM
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # 沿用之前的 ASPP 和 DeepLabV3PDecoder 类（不用重写）
        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18])
        self.decoder = DeepLabV3PDecoder(num_classes=num_classes, low_level_channels=256, aspp_out_channels=256)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x, size=None):
        input_shape = x.shape[-2:]
        # ------ Encoder ------
        x0 = self.relu(self.bn1(self.conv1(x)))
        x_low = self.maxpool(x0)

        # ------ 每层提取完毕立刻套拉满 CBAM 串联送向下一层 ------
        x1 = self.layer1(x_low)
        x1_cbam = self.cbam1(x1)
        
        x2 = self.layer2(x1_cbam)  
        x2_cbam = self.cbam2(x2)
        
        x3 = self.layer3(x2_cbam)
        x3_cbam = self.cbam3(x3)
        
        x4 = self.layer4(x3_cbam)  
        x4_cbam = self.cbam4(x4)

        # ------ ASPP 模块获取大感受野高级别语义 ------
        # 使用 CBAM 强化后的 x4_cbam
        aspp_features = self.aspp(x4_cbam)

        # ------ DeepLabV3+ 解码拼接 ------
        # 因为没有双向加强补偿，x2_cbam 与 x3_cbam 的信息就像经典的 DeepLabV3+ 模型一样被丢弃
        # 在这里原封不动将 x1_cbam 和 aspp 送给解码器聚合
        out = self.decoder(high_level_features=aspp_features, low_level_features=x1_cbam)
        
        # 将模糊后的画质扯平强制恢复原尺寸分辨率 (8倍双线性拉伸)
        target_size = size if size is not None else input_shape
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

        return out
