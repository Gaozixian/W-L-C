import torch
import torch.nn as nn
from SwinTiny import FrontViewSwinTiny
from side_extractor import SideViewDPAIExtractor
from CrosAttention import TemporalCrossAttention
from seg_model import DPAI_UNet_Segmentation, load_unet_model_weight


class MainDrivingModel(nn.Module):
    """
    你最原始的需求：这是主体框架，独立接收并剥离 (RGB, Mask) 后推入对应机制并只训练该网络。
    """
    def __init__(self, num_classes=19, weights_dict=None, embed_dim=384, output_dim=2):
        super(MainDrivingModel, self).__init__()
        
        # 1. 前视分支 (Swin-Tiny 机制)
        self.front_extractor = FrontViewSwinTiny(num_classes=num_classes, weights_dict=weights_dict)
        
        # 2. 时空融合模块 (Cross-Attention 机制)
        self.temporal_fusion = TemporalCrossAttention(
            embed_dim=embed_dim, 
            num_heads=12, 
            num_patches=196, 
            ffn_dim=1536
        )
        
        # 3. 侧视分支 (DPAI侧边机制)
        self.side_extractor = SideViewDPAIExtractor(out_dim=embed_dim)

        # 4. ==== 你的物理状态序列特征提取 (LSTM) ====
        # 对应你 Dataset 取出的 10 个物理量，传入LSTM处理时间序列信息
        self.lstm_hidden_dim = 128
        self.state_lstm = nn.LSTM(input_size=10, hidden_size=self.lstm_hidden_dim, num_layers=1, batch_first=True)
        
        # 5. 最终融合层与回归头（视觉 + LSTM状态 一并回归）
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        # 维度对齐部分
        combined_dim = embed_dim*4 +self.lstm_hidden_dim
        self.regressor = nn.Sequential(
            # 将视觉特征池化后的 embed_dim (384) 与 LSTM 输出 (128) 拼接
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )

        # 🌟 自动执行底层通道适配预训练权重加载
        self._init_weights()

    def _init_weights(self):
        try:
            self.front_extractor.load_pretrained_swin()
            self.side_extractor.load_pretrained_resnet()
        except Exception as e:
            print(f">>> Backbone预训练权重加载失败(跳过): {e}")

    def forward(self, front_images_with_masks, side_image_with_mask, state_seq, return_attn=False):
        """
        front_images_with_masks: [(rgb_t2, m_t2), (rgb_t1, m_t1), (rgb_t, m_t)]
        side_data_list: [(L_rgb, L_m), (R_rgb, R_m), (B_rgb, B_m)]    
        """
        (f_rgb_t2, f_mask_t2), (f_rgb_t1, f_mask_t1), (f_rgb_t, f_mask_t) = front_images_with_masks
        
        # 分别送入 Swin-Tiny 提取机制
        feat_t2 = self.front_extractor(f_rgb_t2, f_mask_t2)
        feat_t1 = self.front_extractor(f_rgb_t1, f_mask_t1)
        feat_t  = self.front_extractor(f_rgb_t, f_mask_t)     # [B, 196, 384]
        
        # 前视流汇入时空机制
        if return_attn:
            f_combined, attn_weights = self.temporal_fusion(feat_t, feat_t1, feat_t2, return_attn=True)
        else:
            f_combined = self.temporal_fusion(feat_t, feat_t1, feat_t2) # [B, 196, 384]
        f_vec = self.final_pool(f_combined.transpose(1, 2)).squeeze(-1)
        
        # 分别送入 DPAI 侧视机制
        side_vectors = []
        for s_rgb, s_mask in side_image_with_mask:
            s_feat = self.side_extractor(s_rgb, s_mask) # [B, 49, 384]
            s_vec = self.final_pool(s_feat.transpose(1, 2)).squeeze(-1)
            side_vectors.append(s_vec)        
        # ==================================
        # D. LSTM 时序提取与最终多模态联合 Regression
        # ==================================
        # 1. 提取物理历史序列最后一部的隐藏状态作为全局车辆动态特征
        # state_seq: [B, 8, 10] (8 帧，每帧 10 个速度/偏航等特征)
        lstm_out, (hn, cn) = self.state_lstm(state_seq)
        state_feat = hn[-1] # 提取LSTM最后一层的输出: [B, 128]
        # 2. 视觉特征拼接并池化
        fused_ultimate = torch.cat([f_vec]+side_vectors+[state_feat], dim=1)        
        # 3. 动态状态 (LSTM) 与 外部视觉 (Swin+ResNet) 的顶级融合！
        out = self.regressor(fused_ultimate)
        if return_attn:
            return out, attn_weights        
        return out


class EndToEndDrivingPipeline(nn.Module):
    """
    顶层端到端流水线（专门负责组装）：
    完全符合你要求的：“先输出掩码 -> 组装原始图像 -> 分别送入 SwinTiny 机制”。
    整个过程只有 MainDrivingModel 里的机制是被训练的。
    """
    def __init__(self, unet_weight_path='', device='cpu', output_dim=2):
        super().__init__()
        # ==========================================
        # 1. 语义分割模型 (只做感知提取，不训练)
        # ==========================================
        self.segmentor = DPAI_UNet_Segmentation(num_classes=19, pretrained=False)
        # 如果给了权重路径，就使用你在 seg_model.py 里写好的智能多卡挂载函数完美加载！
        if unet_weight_path:
            self.segmentor = load_unet_model_weight(self.segmentor, unet_weight_path, device)
        # 冻结语义分割的所有参数！彻底阻断反向传播，不需要再训练语义分割了
        for param in self.segmentor.parameters():
            param.requires_grad = False
        self.segmentor.eval()# 强制锁死在 eval 模式

        # ==========================================
        # 2. 端到端驾驶主脑 (需要基于你的 Dataset 继续被训练)
        # ==========================================
        # 注意: Cityscapes 分割是19类，所以传 num_classes=19
        self.driving_model = MainDrivingModel(num_classes=19, output_dim=output_dim)

    def forward(self, front_images, side_images, state_seq, return_attn=False):
        """
        front_images: List of 3 frames [rgb_t2, rgb_t1, rgb_t], 每个形状 [B, 3, 224, 224]
        side_image: 当前帧三个侧向图像
        state_seq:  历史车辆物理状态 [B, seq_length=8, num_features=10]
        """
        # ==================================
        # 步骤 1：先通过语义分割模型输出全角度掩码
        # ==================================
        self.segmentor.eval()  # 再次保险
        with torch.no_grad():  # 省下大笔显存内存
            f_rgb_t2, f_rgb_t1, f_rgb_t = front_images
            # UNet 分割图(19通道)->Argmax变成(1通道)->升维处理
            f_mask_t2 = torch.argmax(self.segmentor(f_rgb_t2), dim=1).unsqueeze(1).float()
            f_mask_t1 = torch.argmax(self.segmentor(f_rgb_t1), dim=1).unsqueeze(1).float()
            f_mask_t = torch.argmax(self.segmentor(f_rgb_t), dim=1).unsqueeze(1).float()
        # ==================================
        # 步骤 2：将掩码和原始图像拼接/元组化
        # ==================================
            front_seq_input = [(f_rgb_t2, f_mask_t2), (f_rgb_t1, f_mask_t1), (f_rgb_t, f_mask_t)]
            side_rgb_and_masks = []
            for s_rgb in side_images:
                s_mask = torch.argmax(self.segmentor(s_rgb), dim=1).unsqueeze(1).float()
                side_rgb_and_masks.append((s_rgb, s_mask))
        # ==================================
        # 步骤 3：分别送入后续主体网络机制！
        # ==================================
        prediction = self.driving_model(front_seq_input, side_rgb_and_masks, state_seq, return_attn=return_attn)
        return prediction

# 测试验证代码块
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 你存放训练好 UNet 下来的权重（路径示例，如果你有真实的按真实写）
    # 在这里初始化之后，网络就加载成功了！
    unet_pth = r'E:/Laboratory files/code_project/D2D/visual/Segmentation/checkpoints_unet/dpai_unet/dpai_unet.pth'
    
    pipeline = EndToEndDrivingPipeline(unet_weight_path=unet_pth, device=device, output_dim=2).to(device)
    
    # 模拟外部传入的数据流: RGB 纯图像流 + JSON 读取出来的十维物理序列
    B = 2
    dummy_rgb = torch.randn(B, 3, 224, 224).to(device)
    front_seq = [dummy_rgb, dummy_rgb, dummy_rgb]  # [rgb_t2, rgb_t1, rgb_t]
    side_input = dummy_rgb
    
    # 模拟 Dataset 第 281 行传入的 state_seq_tensor (8 帧历史, 10 个速度加速度特征)
    dummy_state_seq = torch.randn(B, 8, 10).to(device)
    
    # 一键畅通！融合视觉感知池与LSTM物理动力学池
    output = pipeline(front_seq, side_input, dummy_state_seq)
    print(f"顶层端到端流水线 输出预测形状: {output.shape}") 
