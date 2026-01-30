import torch


class HypergraphConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, node_types):
        super().__init__()
        # Point 3: 为不同类型节点设置独立的映射层
        self.type_projectors = torch.nn.ModuleDict({
            t: torch.nn.Linear(in_dim, out_dim) for t in node_types
        })
        # Point 4: 简单的注意力权重系数（也可以设为可学习参数）
        self.edge_weight = torch.nn.Parameter(torch.ones(len(node_types)))
        self.node_types = node_types
        # 新增：定义 Dropout 层
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, X_dict, H, Dv_inv_sqrt, De_inv, node_masks):
        """
        X_dict: 原始节点 Embedding 矩阵
        node_masks: 字典，包含每种类型节点在全量节点中的索引掩码
        """
        # 1. 类型感知投影 (Point 3)
        X_projected = torch.zeros_like(X_dict)
        for t, mask in node_masks.items():
            if mask.any():
                # 修改：投影后接激活函数和 Dropout
                feat = self.type_projectors[t](X_dict[mask])
                X_projected[mask] = self.dropout(torch.nn.functional.relu(feat))
        # 2. 节点度归一化
        X = Dv_inv_sqrt.unsqueeze(1) * X_projected

        # 3. 节点 -> 超边 (聚合)
        X = torch.sparse.mm(H.t(), X)

        # 4. 超边归一化 (Point 4: 这里可以根据超边类型进一步加权)
        X = De_inv.unsqueeze(1) * X

        # 5. 超边 -> 节点 (分发)
        X = torch.sparse.mm(H, X)

        # 6. 最终归一化
        X = Dv_inv_sqrt.unsqueeze(1) * X
        return X
