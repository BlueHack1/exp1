import torch
import torch.nn.functional as F

def compute_score(query_vec, user_vec, poi_vec, alpha=0.5, beta=0.5):
    """
    个性化排序函数
    """
    semantic_sim = F.cosine_similarity(query_vec, poi_vec, dim=0)
    interest_sim = F.cosine_similarity(user_vec, poi_vec, dim=0)
    return alpha * semantic_sim + beta * interest_sim


def bpr_loss(pos_score, neg_score):
    """
    Pairwise Ranking Loss
    """
    return -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
