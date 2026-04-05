import torch
import torch.nn as nn

class TemporalCrossAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, num_patches=196, ffn_dim=1536):
        super().__init__()
        self.embed_dim = embed_dim

        # ==========================================
        # 1. 时空位置编码 (Spatiotemporal Positional Encoding)
        # ==========================================
        # 空间位置编码：告诉模型这 196 个 Patch 的空间排列关系 (14x14)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=.02)

        # 时间位置编码：告诉模型哪一帧是 t (当前), 哪一帧是 t-1, 哪一帧是 t-2
        # 我们定义 3 个时间步的 Embedding
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=.02)

        # ==========================================
        # 2. 交叉注意力机制核心
        # ==========================================
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True # 确保输入形状是 [Batch, Seq_Len, Dim]
        )

        # 标准 Transformer Decoder 里的 LayerNorm 和 前馈神经网络 (FFN)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, feat_t, feat_t_minus_1, feat_t_minus_2, return_attn=False):
        """
        Args:
            feat_t: 当前帧特征 [B, 196, 384]
            feat_t_minus_1: 过去第 1 帧特征 [B, 196, 384]
            feat_t_minus_2: 过去第 2 帧特征 [B, 196, 384]
        """
        B = feat_t.shape[0]

        # --- A. 注入空间位置编码 ---
        # 广播机制会自动应用到 Batch 的每一条数据上
        feat_t = feat_t + self.spatial_pos_embed
        feat_t_minus_1 = feat_t_minus_1 + self.spatial_pos_embed
        feat_t_minus_2 = feat_t_minus_2 + self.spatial_pos_embed

        # --- B. 注入时间位置编码 ---
        # self.temporal_pos_embed[0, 0, :] 代表当前帧 t
        # self.temporal_pos_embed[0, 1, :] 代表 t-1
        # self.temporal_pos_embed[0, 2, :] 代表 t-2
        feat_t = feat_t + self.temporal_pos_embed[:, 0, :].unsqueeze(1)
        feat_t_minus_1 = feat_t_minus_1 + self.temporal_pos_embed[:, 1, :].unsqueeze(1)
        feat_t_minus_2 = feat_t_minus_2 + self.temporal_pos_embed[:, 2, :].unsqueeze(1)

        # --- C. 准备 Q 和 K, V ---
        # Query: 当前帧
        # 形状: [B, 196, 384]
        q = feat_t
        # Key & Value: 前两帧在序列维度上拼接
        # 形状: [B, 196*2, 384] -> [B, 392, 384]
        kv = torch.cat([feat_t_minus_1, feat_t_minus_2], dim=1)

        # --- D. 交叉注意力计算 ---
        # 当前帧去“询问”过去两帧，寻找动态变化线索（如车辆速度、行人轨迹）
        attn_output, _ = self.cross_attn(query=q, key=kv, value=kv)

        # --- E. 残差连接与前馈网络 (Transformer 标准操作) ---
        # 1. 融合历史信息并保持当前帧的底色
        x = self.norm1(q + attn_output)
        # 2. 特征非线性映射
        out = self.norm2(x + self.ffn(x))

        # 输出形状依然是 [B, 196, 384]
        if return_attn:
            return out, attn_output
        
        return out