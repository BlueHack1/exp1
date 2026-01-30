from exp.config import BERT_MODEL
from sentence_transformers import SentenceTransformer

# 全局变量，模块加载时只实例化一次
model = SentenceTransformer(BERT_MODEL)

def get_model():
    """
    获取模型实例
    """
    return model

