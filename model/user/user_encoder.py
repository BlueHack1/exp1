import torch
import torch.nn as nn

class UserInterestEncoder(nn.Module):
    """
    基于 Attention 的用户兴趣聚合
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, poi_embeddings):
        """
        poi_embeddings: [N, D]
        """
        # 计算每个历史 POI 的重要性权重
        weights = torch.softmax(self.attn(poi_embeddings), dim=0)

        # 加权求和得到用户兴趣向量
        user_vec = torch.sum(weights * poi_embeddings, dim=0)
        return user_vec
