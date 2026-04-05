import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import timm

# ==========================================
# 模块 1：面积 × 重要性的语义下采样函数
# ==========================================
def importance_area_downsample(mask, num_classes, weights_dict, patch_size=4):
    """
    将 224x224 的 Mask 下采样到 56x56，结合类别优先级
    """
    B, _, H, W = mask.shape

    # 初始化权重张量 (默认全为1.0)
    weight_tensor = torch.ones(num_classes, device=mask.device)
    for class_id, weight in weights_dict.items():
        weight_tensor[class_id] = weight
    weight_tensor = weight_tensor.view(1, num_classes, 1, 1)

    # 转换为 One-Hot 编码: [B, C, H, W]
    mask_onehot = F.one_hot(mask.squeeze(1).long(), num_classes=num_classes).float()
    mask_onehot = mask_onehot.permute(0, 3, 1, 2)

    # 计算 4x4 区域内的面积 (频次)
    area_count = F.avg_pool2d(mask_onehot, kernel_size=patch_size, stride=patch_size) * (patch_size ** 2)

    # 乘以重要性权重，并取最大值对应的类别
    score = area_count * weight_tensor
    downsampled_mask = torch.argmax(score, dim=1) # 形状: [B, H/4, W/4]

    return downsampled_mask

# ==========================================
# 模块 2：带有语义注入的 PatchEmbedding 层
# ==========================================
class SemanticInjectingPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, num_classes=20, weights_dict=None):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.weights_dict = weights_dict if weights_dict else {}

        # A. 修改卷积层接收 4 通道 (3 RGB + 1 Mask)
        self.proj = nn.Conv2d(in_chans + 1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

        # B. 语义嵌入查表 (96 维)
        self.semantic_embedding = nn.Embedding(num_classes, embed_dim)
        nn.init.normal_(self.semantic_embedding.weight, std=0.02) # 标准初始化

    def forward(self, rgb, mask):
        # 🌟 新增：动态 Resize 到 224x224 以适配不同尺度的输入
        # RGB 使用双线性插值，Mask 使用最近邻插值（保持类别索引不被破坏）
        if rgb.shape[-2:] != (224, 224):
            rgb = F.interpolate(rgb, size=(224, 224), mode='bilinear', align_corners=False)
        if mask.shape[-2:] != (224, 224):
            mask = F.interpolate(mask, size=(224, 224), mode='nearest')

        B, C, H, W = rgb.shape

        # 1. 通道拼接: [B, 4, 224, 224]
        x_concat = torch.cat([rgb, mask.float()], dim=1)

        # 2. 图像切块与线性映射: [B, 96, 56, 56]
        patch_tokens = self.proj(x_concat)

        # 展平为 Transformer 序列格式: [B, 3136, 96]
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        patch_tokens = self.norm(patch_tokens)

        # 3. 语义下采样: [B, 56, 56]
        down_mask = importance_area_downsample(mask, self.num_classes, self.weights_dict, self.patch_size)
        down_mask_flat = down_mask.flatten(1) # [B, 3136]

        # 4. 查表获取语义向量: [B, 3136, 96]
        sem_embeds = self.semantic_embedding(down_mask_flat)

        # 5. 核心：语义注入 (逐元素相加)
        output_tokens = patch_tokens + sem_embeds

        # 🌟 修复关键：新版 timm Swin 需要 4D 输入 [B, H, W, C]
        H_p, W_p = H // self.patch_size, W // self.patch_size
        output_tokens = output_tokens.reshape(B, H_p, W_p, -1)

        # 返回融合后的 tokens，以及当前特征图的宽高
        resolution = (H_p, W_p)
        return output_tokens, resolution

# ==========================================
# 模块 3：前向视角主干网络 (抽取至 Stage 3)
# ==========================================
class FrontViewSwinTiny(nn.Module):
    def __init__(self, num_classes=20, weights_dict=None):
        super().__init__()
        # 1. 初始化自定义的语义注入 PatchEmbed
        self.patch_embed = SemanticInjectingPatchEmbed(
            img_size=224, patch_size=4, in_chans=3, embed_dim=96,
            num_classes=num_classes, weights_dict=weights_dict
        )

        # 2. 从 timm 加载 Swin-Tiny 的注意力层 (Layers)
        base_swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)

        # 提取 Stage 1 到 Stage 4 的层
        self.layers = base_swin.layers

    # def load_pretrained_swin(self):
    #     """
    #     核心逻辑：加载 Swin-Tiny 预训练权重并适配 4 通道 PatchEmbed
    #     """
    #     print(">>> 正在加载前向视角 Swin-Tiny 预训练权重...")
    #     # 1. 获取官方预训练模型
    #     official_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    #     official_dict = official_model.state_dict()
    #     model_dict = self.state_dict()

    #     # 2. 特殊处理 patch_embed.proj.weight (4 通道适配)
    #     # 官方是 [96, 3, 4, 4], 我们需要 [96, 4, 4, 4]
    #     if 'patch_embed.proj.weight' in official_dict:
    #         old_weight = official_dict['patch_embed.proj.weight']
    #         new_weight = torch.zeros(96, 4, 4, 4)
    #         new_weight[:, :3, :, :] = old_weight
    #         new_weight[:, 3, :, :] = old_weight.mean(dim=1) # Mask 通道初始化
    #         official_dict['patch_embed.proj.weight'] = new_weight

    #     # 3. 匹配并加载其余层
    #     updated_dict = {k: v for k, v in official_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    #     self.load_state_dict(updated_dict, strict=False)
    #     print(">>> 前向视角权重适配完成！")
    def load_pretrained_swin(self, local_path=r'E:\Laboratory files\code_project\D2D\weights\swin_tiny_patch4_window7_224.pth'):
        """
        核心逻辑：加载本地 Swin-Tiny 预训练权重并适配 4 通道 PatchEmbed
        """
        if not os.path.exists(local_path):
            print(f"❌ 找不到本地权重文件: {local_path}，加载失败！")
            return
        print(f">>> 正在从本地加载前向视角 Swin-Tiny 权重: {local_path}")
        
        # 1. 直接从本地文件加载权重字典
        checkpoint = torch.load(local_path, map_location='cpu')
        # 兼容处理：有些 .pth 文件带有 'model' 键，有些直接是字典
        official_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        model_dict = self.state_dict()
        # 2. 特殊处理 patch_embed.proj.weight (4 通道适配，这部分逻辑保留你原来的)
        if 'patch_embed.proj.weight' in official_dict:
            old_weight = official_dict['patch_embed.proj.weight']
            # 官方是 [96, 3, 4, 4], 我们需要 [96, 4, 4, 4]
            if old_weight.shape[1] == 3:
                new_weight = torch.zeros(96, 4, 4, 4)
                new_weight[:, :3, :, :] = old_weight
                new_weight[:, 3, :, :] = old_weight.mean(dim=1) # Mask 通道初始化
                official_dict['patch_embed.proj.weight'] = new_weight
                print(">>> 已完成 4 通道 PatchEmbed 权重映射适配。")
        # 3. 匹配并加载其余层
        updated_dict = {k: v for k, v in official_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        self.load_state_dict(updated_dict, strict=False)
        print(">>> 前向视角权重适配完成！")
  

    def forward(self, rgb, mask):
        # 1. 语义注入阶段
        # x 形状: [Batch, 56, 56, 96] (4D)
        x, resolution = self.patch_embed(rgb, mask)
        # 2. 依次通过 Swin 的各个 Stage
        stage3_feat = None
        for i, layer in enumerate(self.layers):
            # 新版 timm 的 Stage 接收并输出 4D 张量 [B, H, W, C]
            x = layer(x)
            if i == 2:  # 当完成 Stage 3 (索引为 2) 时，提取特征并退出循环
                stage3_feat = x
                break

        # 最终输出转换回序列格式: [Batch, L, C]
        # Stage 3 完成后，空间分辨率为 14x14，通道为 384
        # stage3_feat 形状为 [B, 14, 14, 384]
        b, h_f, w_f, c_f = stage3_feat.shape
        stage_feat = stage3_feat.reshape(b, -1, c_f)
        print(f"[Debug Swin] FrontView Swin Output (Flattened): {stage_feat.shape}")
        
        return stage_feat
