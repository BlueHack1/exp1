import torch
import json
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from exp.config import YELP_DATA_REVIEW, YELP_DATA_USER
from exp.data.yelp.user.hyperedge import HypergraphConstructor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def generate_semantic_embeddings(output_path='semantic_init.pt', embed_dim=128):
    """
    检索增强预处理：将所有异质节点转化为统一维度的语义向量
    """
    # 1. 加载模型
    logging.info("加载预训练语义模型...")
    # all-MiniLM-L6-v2 输出是 384 维
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_dim = sbert_model.get_sentence_embedding_dimension()

    # 2. 准备数据源
    logging.info("构建超图以获取全量节点...")
    hyperedgesCon = HypergraphConstructor(
        YELP_DATA_REVIEW / 'reviewed_business.jsonl',
        YELP_DATA_USER / 'user_interest.jsonl',
        'zone_node.json'
    )
    hyperedgesCon.build_hyperedges()
    poi_infos = hyperedgesCon.poi_infos

    all_nodes = set()
    for edge in (hyperedgesCon.train_hyperedges + hyperedgesCon.test_hyperedges):
        all_nodes.update(edge)
    all_nodes = sorted(list(all_nodes))
    num_nodes = len(all_nodes)
    logging.info(f"总节点数：{num_nodes}")

    # 3. 初始化结果矩阵
    init_weights = torch.zeros(num_nodes, embed_dim)

    # 定义一个简单的线性投影，将 384 维转为 128 维
    # 这样比 interpolate（插值）更符合 NLP 的特征变换逻辑
    projection = torch.nn.Linear(sbert_dim, embed_dim)
    torch.nn.init.xavier_uniform_(projection.weight)

    logging.info("开始跨模态向量化...")

    for i, node in enumerate(tqdm(all_nodes)):
        # --- 根据不同前缀构建描述文本 (RAG 核心) ---
        if node.startswith('poi_'):
            bid = node.replace('poi_', '')
            info = poi_infos.get(bid, {})
            text = f"Point of interest named {info.get('name', 'unknown')}, which belongs to {info.get('categories', 'various categories')}."

        elif node.startswith('attr_'):
            attr_name = node.replace('attr_', '').replace('_', ' ')
            text = f"This business has the attribute or feature of {attr_name}."

        elif node.startswith('zone_'):
            text = f"A specific geographical urban area or administrative zone in the city."

        elif node.startswith('time_'):
            time_info = node.replace('time_', '').replace('_', ' ')
            text = f"The specific time slot of {time_info} for visiting locations."

        elif node.startswith('user_'):
            text = f"An active urban consumer searching for personalized recommendations."

        else:
            text = "An auxiliary node in the urban hypergraph."

        # 4. 编码并降维
        with torch.no_grad():
            raw_vec = torch.tensor(sbert_model.encode(text))
            # 通过线性层将 384 -> 128
            refined_vec = projection(raw_vec)
            init_weights[i] = refined_vec

    # 5. 保存
    torch.save(init_weights, output_path)
    logging.info(f"成功保存全节点语义初始化权重至: {output_path}")


if __name__ == "__main__":
    # 注意：这里直接运行即可，不再需要传入 poi_path
    generate_semantic_embeddings(output_path='semantic_init.pt', embed_dim=128)