import torch


class HypergraphConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim): # 去掉了 node_types 参数
        super().__init__()
        # 卷积层现在不负责投影，只负责传播，所以不需要 ModuleDict
        # 也不需要 Dropout 和 Linear

    def forward(self, X, H, Dv_inv_sqrt, De_inv):
        """
        X: 已经是投影对齐后的 Embedding
        """
        # 1. 节点度归一化
        X_norm = Dv_inv_sqrt.unsqueeze(1) * X

        # 2. 节点 -> 超边 (聚合)
        X_e = torch.sparse.mm(H.t(), X_norm)

        # 3. 超边归一化
        X_e = De_inv.unsqueeze(1) * X_e

        # 4. 超边 -> 节点 (分发)
        X_n = torch.sparse.mm(H, X_e)

        # 5. 最终归一化
        X_final = Dv_inv_sqrt.unsqueeze(1) * X_n
        return X_final