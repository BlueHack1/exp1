import torch
import torch.nn.functional as F
from exp.model.gnn.HyperGraphConv import HypergraphConv

class HyperGraphModel(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, all_nodes, init_weights=None):
        super().__init__()
        # --- 修改点 1: 支持外部语义权重注入 ---
        if init_weights is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(init_weights, freeze=False)
        else:
            self.embedding = torch.nn.Embedding(num_nodes, emb_dim)

        # 预先计算节点类型掩码
        self.node_types = ['user_', 'poi_', 'attr_', 'time_', 'zone_']
        self.node_masks = {}
        for t in self.node_types:
            mask = torch.tensor([n.startswith(t) for n in all_nodes])
            self.register_buffer(f'mask_{t}', mask)

        self.hgc1 = HypergraphConv(emb_dim, emb_dim, self.node_types)
        self.hgc2 = HypergraphConv(emb_dim, emb_dim, self.node_types)

        # Point 6: 引入初始残差系数 alpha 和 恒等映射系数 beta
        self.alpha = 0.7
        self.beta = 0.5

    def get_masks(self):
        return {t: getattr(self, f'mask_{t}') for t in self.node_types}

    def forward(self, H, Dv_inv_sqrt, De_inv):
        X0 = self.embedding.weight
        masks = self.get_masks()

        # 第一层卷积
        X1 = self.hgc1(X0, H, Dv_inv_sqrt, De_inv, masks)
        # Point 6: GCNII 式残差连接: (1-alpha)*X_conv + alpha*X0
        X1 = F.relu((1 - self.alpha) * X1 + self.alpha * X0)

        # 第二层卷积
        X2 = self.hgc2(X1, H, Dv_inv_sqrt, De_inv, masks)
        X2 = F.relu((1 - self.alpha) * X2 + self.alpha * X0)

        return X2
