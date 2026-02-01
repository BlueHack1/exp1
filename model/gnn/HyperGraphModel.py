import torch
import torch.nn.functional as F
from exp.model.gnn.HyperGraphConv import HypergraphConv


class HyperGraphModel(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, all_nodes, init_weights=None):
        super().__init__()
        if init_weights is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(init_weights, freeze=False)
        else:
            self.embedding = torch.nn.Embedding(num_nodes, emb_dim)

        self.node_types = ['user_', 'poi_', 'attr_', 'time_', 'zone_']
        # Point A: 这里的投影层保留，用于初始空间的对齐
        self.type_projectors = torch.nn.ModuleDict({
            t: torch.nn.Linear(emb_dim, emb_dim) for t in self.node_types
        })

        for t in self.node_types:
            mask = torch.tensor([n.startswith(t) for n in all_nodes])
            self.register_buffer(f'mask_{t}', mask)
        # --- 【新增】可学习的融合权重 ---
        # 我们有 3 层 (X0, X1, X2)，所以初始化为长度为 3 的全 1 向量
        self.fusion_weights = torch.nn.Parameter(torch.ones(3))
        # Point B: 使用轻量化的卷积
        self.hgc1 = HypergraphConv(emb_dim, emb_dim)
        self.hgc2 = HypergraphConv(emb_dim, emb_dim)

    def get_masks(self):
        return {t: getattr(self, f'mask_{t}') for t in self.node_types}

    def forward(self, H, Dv_inv_sqrt, De_inv):
        X_raw = self.embedding.weight
        masks = self.get_masks()

        # 1. 初始空间对齐 (保留 Linear，这是 RAG 的灵魂)
        X0 = torch.zeros_like(X_raw)
        for t, mask in masks.items():
            if mask.any():
                X0[mask] = self.type_projectors[t](X_raw[mask])

        # 2. 迭代卷积 (传播结构信息)
        X1 = self.hgc1(X0, H, Dv_inv_sqrt, De_inv)
        X2 = self.hgc2(X1, H, Dv_inv_sqrt, De_inv)

        # 3. 核心改进：Softmax 动态加权
        # 移除了之前的 0.5 强行保底，改用纯动态权重，但在训练脚本里控制它
        w = torch.nn.functional.softmax(self.fusion_weights, dim=0)
        X_final = w[0] * X0 + w[1] * X1 + w[2] * X2
        return X_final