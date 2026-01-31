import json
import logging
import numbers
from datetime import datetime
import random
from typing import Dict, List, Tuple, Set

import torch
from torch import nn, optim

from exp.config import YELP_DATA_REVIEW, YELP_DATA_USER
from exp.model.gnn.HyperGraphConv import HypergraphConv
from exp.model.gnn.HyperGraphModel import HyperGraphModel

'''
    超边:
        hyperedges = [
        f"user_{user_id}",
        f"poi_{business_id}",
        f"time_{time_slot}",
        zone_node
        ] + attr_nodes

'''

def is_true(v):
    return v is True or (isinstance(v, str) and v.lower() == "true")

# 时间节点：日期离散化
def discretize_time(dt_str):
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    hour = dt.hour

    # weekend / weekday
    is_weekend = "weekend" if dt.weekday() >= 5 else "weekday"

    # time period
    if 6 <= hour < 11:
        period = "morning"
    elif 11 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 22:
        period = "evening"
    else:
        period = "night"

    return f"{is_weekend}_{period}"

# 空间节点
def build_spatial_node(poi_path, K=50, output_name='zone_node.json'):
    import numpy as np

    with open(poi_path, 'r', encoding='utf-8') as f:
        poi_ids = []
        coords = []
        for line in f:
            poi = json.loads(line)
            if poi["latitude"] is None or poi["longitude"] is None:
                continue
            poi_ids.append(poi["business_id"])
            coords.append([poi["latitude"], poi["longitude"]])
        # 转为 矩阵，后面 key-means 运算。
        coords = np.array(coords)

        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=K,
            random_state=42,
            n_init=10
        )

        zone_labels = kmeans.fit_predict(coords)
        zone_map = {}
        with open(output_name, "w") as f:
            for poi_id, z in zip(poi_ids, zone_labels):
                zone_map[poi_id] = f"zone_{z}"

            json.dump(zone_map, f)
        logging.info("构建空间节点完毕")

# 属性节点：返回有序的 attr，保证复现
'''
    嵌套属性：attr_{key}_{skey}
    普通属性：attr_{key}
    仅仅保留 True 的属性
    For POI attributes, we adopt a structure-aware encoding strategy. 
    Boolean attributes are modeled as atomic attribute nodes only when positively observed. 
    Nested dictionary attributes are decomposed into attribute groups and fine-grained sub-attributes, 
    preserving only true-valued facts under an open-world assumption. 
    This design avoids introducing noisy negative signals and enables high-order semantic aggregation in the hypergraph.
'''


def build_attr_node(poi_id: str, attributes: Dict
                    ) -> Tuple[
    List[str],  # Attr nodes
    List[Tuple[str, str]],  # POI -> Attr edges
]:
    attr_nodes = set()
    poi_attr_edges = []

    if attributes is not None:
        for key, value in attributes.items():
            if value is not None:
                #  bool 属性，保证数据集的 value 是 true 或者 false 对象，而不是字符串
                if isinstance(value, bool) and value:
                    attr = f"attr_bool_{key}"
                    attr_nodes.add(attr)
                    poi_attr_edges.append((poi_id, attr))

                # dict 嵌套属性，嵌套属性的嵌套 value 是 true 或者 false
                elif isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if is_true(sub_val):
                            attr = f"attr_sub_{key}_{sub_key}"
                            attr_nodes.add(attr)
                            poi_attr_edges.append((poi_id, attr))

                # 字符串
                elif isinstance(value, str):
                    value = value.strip()
                    attr = f"attr_val_{key}_{value.replace(' ', '_')}"
                    attr_nodes.add(attr)
                    poi_attr_edges.append((poi_id, attr))
                # 数值类型
                elif isinstance(value, numbers.Number):
                    attr = f"attr_val_{key}_{value}"
                    attr_nodes.add(attr)
                    poi_attr_edges.append((poi_id, attr))
        # return sorted(attr_nodes), poi_attr_edges
    return sorted(attr_nodes)


import numpy as np
from collections import Counter, defaultdict


def sanity_check_zones(poi_ids, coords, poi2zone, reviews, hyperedges, cold_threshold=5):
    """
    对空间节点 Zone 进行一次性 sanity check

    参数:
    - poi_ids: list, 所有 POI id
    - coords: np.array, shape=(num_POI, 2), 经纬度
    - poi2zone: dict, POI_id -> Zone_x
    - reviews: list of dict, 每条 review 中必须包含 'business_id' 和 'stars'
    - hyperedges: list of list, 每条超边包含节点字符串
    - cold_threshold: int, review_count <= cold_threshold 的 POI 判定为冷启动
    """

    logging.info("===== 1. Zone POI 数量统计 =====")
    zone_counts = Counter(poi2zone.values())
    for z, count in zone_counts.items():
        print(f"{z}: {count} POI")
    logging.info(f"Zone POI 数量范围: min={min(zone_counts.values())}, max={max(zone_counts.values())}\n")

    logging.info("===== 2. Zone 地理连续性 (中心点) =====")
    zone_points = defaultdict(list)
    for idx, pid in enumerate(poi_ids):
        zone = poi2zone[pid]
        zone_points[zone].append(coords[idx])

    for z, points in zone_points.items():
        points = np.array(points)
        center = points.mean(axis=0)
        print(f"{z}: center_lat={center[0]:.5f}, center_lon={center[1]:.5f}, num_points={len(points)}")
    print()

    logging.info("===== 3. 冷启动 POI 是否有 Zone =====")
    cold_pois = [r['business_id'] for r in reviews if r.get('review_count', 0) <= cold_threshold]
    missing_zone = [pid for pid in cold_pois if pid not in poi2zone]
    if len(missing_zone) == 0:
        logging.info(f"所有 {len(cold_pois)} 个冷启动 POI 都有 Zone")
    else:
        logging.info(f"{len(missing_zone)} 个冷启动 POI 没有 Zone: {missing_zone}")
    print()

    logging.info("===== 4. Zone 超边复用率 =====")
    zone_edge_counts = Counter()
    for edge in hyperedges:
        for node in edge:
            if node.startswith("zone_"):
                zone_edge_counts[node] += 1
    for z, count in zone_edge_counts.items():
        logging.info(f"{z} 出现在 {count} 条超边中")

    logging.info("\n===== Sanity check 完成 =====")
    return zone_counts, zone_points, zone_edge_counts


hyperedges = []
train_hyperedges = []
test_hyperedges = []
# 加载全部 poi
poi_infos = {}
# 构建超边
def build_hyperedge(poi_path, user_path, zone_path):
    with open(poi_path, 'r', encoding='utf-8') as f:
        for f_poi in f:
            poi = json.loads(f_poi)
            poi_infos[poi.get('business_id')] = poi

    # 加载全部地区节点
    zone_nodes = {}
    with open(zone_path, 'r') as f:
        zone_nodes = json.load(f)

    sorted_pois = []
    with open(user_path, mode='r') as f:
        for u_line in f:
            user = json.loads(u_line)
            poi_list = user.get('poi')
            #logging.info(f"{ user.get('user_id')}：访问过{len(poi_list)}个地点")
            for poi in poi_list:
                sorted_pois.append(poi)
    with open(user_path, mode='r') as f:
        for u_line in f:
            user = json.loads(u_line)
            user_id = user.get('user_id')
            poi_list = user.get('poi')
            # 按时间排序，保持同一个 user 的时间一致性，防止测试集中出现以前的访问超边
            poi_list.sort(key=lambda x: x['date'])
            current_hyperedges = []
            for poi in poi_list:
                business_id = poi.get('business_id')
                attr_node_list = build_attr_node(business_id, poi_infos.get(business_id).get('attributes'))
                # 时间离散化
                time_slot = discretize_time(poi.get('date'))
                zone_node = zone_nodes[business_id]
                edge = [
                    f"user_{user_id}",  # 人
                    f"poi_{business_id}",  # 去哪
                    f"time_{time_slot}",  # 时间
                    zone_node,  # 在哪一片
                ]

                edge.extend(attr_node_list)
                current_hyperedges.append(edge)
                # logging.info(edge)
            test_ratio = 0.2
            split_idx = int(len(current_hyperedges) * (1 - test_ratio))
            train_hyperedges.extend(current_hyperedges[:split_idx])  # 历史
            test_hyperedges.extend(current_hyperedges[split_idx:])  # 未来
            hyperedges.extend(current_hyperedges)



# build_spatial_node(YELP_DATA_REVIEW / 'reviewed_business.jsonl', K=50, output_name = 'zone_node.json')

build_hyperedge(YELP_DATA_REVIEW / 'reviewed_business.jsonl',
                YELP_DATA_USER / 'user_interest.jsonl',
                'zone_node.json')


# 矩阵构建
# def build_matrix():
all_nodes = set()
# 节点、边选择训练集构建。
for edge in (train_hyperedges + test_hyperedges):
    # 将每条超边的 node 都加入到 全部节点 中。
    all_nodes.update(edge)
# 转为列表并排序
all_nodes = sorted(list(all_nodes))
logging.info(f"总节点数：{len(all_nodes)}")

# 节点编号
node2id = {node: idx for idx, node in enumerate(all_nodes)}
id2node = {idx: node for node, idx in node2id.items()}

# 超边编号
edge2id = {idx: idx for idx in range(len(train_hyperedges))}

# 构建 coo 索引
# node index
row_idx = []
# train_hyperedges index
col_idx = []
for e_id, edge in enumerate(train_hyperedges):
    # 遍历每个超边 = list，设置其对应的索引
    for node in edge:
        col_idx.append(e_id)
        row_idx.append(node2id[node])

# 变为 位置矩阵
indices = torch.tensor([row_idx, col_idx], dtype=torch.long)
values = torch.ones(len(row_idx))


num_nodes = len(all_nodes)
num_edges = len(train_hyperedges)

# 按 行 、 列 和 value 构建 Coo 矩阵， value 决定 indices 的位置 存放的值 = 1，其它地方是0
H = torch.sparse_coo_tensor(
    indices,
    values,
    size=(num_nodes, num_edges)
)
logging.info(f"训练集大小：{len(train_hyperedges)}")
logging.info(f"测试集大小：{len(test_hyperedges)}")
logging.info(f"H 矩阵大小:{H.shape}")

# 度矩阵：对角矩阵，稀疏矩阵的度
# 节点度矩阵
Dv = torch.sparse.sum(H, dim=1).to_dense()
Dv_inv_sqrt = torch.pow(Dv, -0.5)
# 排除 1 / 0  = ∞ 的情况。
Dv_inv_sqrt[torch.isinf(Dv_inv_sqrt)] = 0

# 边度矩阵
De = torch.sparse.sum(H, dim=0).to_dense()
De_inv = torch.pow(De, -1)
De_inv[torch.isinf(De_inv)] = 0
#
# build_matrix()

# --------------------------
# Step 3: BPR Training
# --------------------------

def bpr_loss_batch(user_emb, pos_emb, neg_embs, reg_weight=1e-4):

    # user_emb: [1, emb_dim], pos_emb: [1, emb_dim], neg_emb_list: [num_neg, emb_dim]

    # logging.info(pos_score.shape)
    # 广播 做点积
    # logging.info((user_emb * neg_emb_list).sum(dim=1))
    # logging.info((user_emb * neg_emb_list).sum(dim=1,  keepdim = True ))
    """
        矩阵化 BPR 损失
        user_emb: [batch, dim]
        pos_emb:  [batch, dim]
        neg_embs: [batch, num_neg, dim]
        """
    # 1. 计算正样本得分 (点积)
    pos_score = (user_emb * pos_emb).sum(dim=1, keepdim=True)  # [batch, 1]

    # 2. 计算负样本得分 (批量矩阵乘法)
    # [batch, num_neg, dim] @ [batch, dim, 1] -> [batch, num_neg, 1]
    neg_score = torch.bmm(neg_embs, user_emb.unsqueeze(2)).squeeze(2)  # [batch, num_neg]

    # 3. 核心 BPR 公式: 推大正负得分差
    # 广播机制会让 pos_score 对齐每个 neg_score
    loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

    # 4. L2 正则化
    reg_loss = reg_weight * (user_emb.norm(2).pow(2) +
                             pos_emb.norm(2).pow(2) +
                             neg_embs.norm(2).pow(2))
    return loss + reg_loss
# def sample_negatives(user_idx, num_neg, poi_ids):
#     # 移除用户训练集中已访问的 poi
#     candidate = list(set(poi_ids.tolist()) - user_poi_dict[user_idx])
#     # 随机先采 50 个
#     candidates = random.sample(candidate, min(50, len(candidate)))
#
#     # logging.info(f"{poi_idx in all_poi_set}")
#     all_poi_set = set(poi_ids.tolist()) - user_poi_dict[user_idx]
#     #logging.info(len(all_poi_set))
#
#     # 返回全局 nodeid 索引
#     neg_idx = random.sample(list(all_poi_set), num_neg)
#     return neg_idx
#


# 假设：
# node2id: 节点名 -> 编号
# all_nodes: 节点列表
# X: [num_nodes, embed_dim] 经过 HypergraphConv 得到

# 用户节点列表【包含训练集和测试集的 user 】


def sample_negatives(user_idx, num_neg, poi_ids, X):
    """

    :param user_idx:
    :param num_neg: 每条正样本采 多少个负样本
    :param poi_ids:
    :param X:
    :return:
    """
    # 1. 过滤掉已访问的
    candidate_set = list(set(poi_ids.tolist()) - user_poi_dict[user_idx])

    # 2. 采样 200 个候选（扩大搜索范围）
    candidates = random.sample(candidate_set, min(200, len(candidate_set)))

    # 3. 计算分数寻找 Hard Negatives
    u_vec = X[user_idx]
    c_vecs = X[candidates]
    scores = torch.matmul(c_vecs, u_vec)

    # 4. 混合策略：前 num_neg // 2 选最难的，后一半随机选
    hard_num = num_neg // 2
    rand_num = num_neg - hard_num

    # 获取 hard
    hard_indices = scores.topk(hard_num).indices.cpu().tolist()
    hard_samples = [candidates[i] for i in hard_indices]

    # 获取 random (从剩下的里面随机选)
    remaining = list(set(candidates) - set(hard_samples))
    rand_samples = random.sample(remaining, min(rand_num, len(remaining)))

    return hard_samples + rand_samples


def sample_negatives_batch(user_idx_tensor, num_neg, poi_ids, X):
    """

    :param user_idx:
    :param num_neg: 每条正样本采 多少个负样本
    :param poi_ids: 地点: id
    :param X:
    :return:
   """
    negatives_samples_list = []
    for user_idx in user_idx_tensor:
        negatives_samples_list.append(sample_negatives(user_idx, num_neg, poi_ids, X))

    return negatives_samples_list

def sample_negatives_gpu(u_idx, num_neg, poi_ids, X_snapshot):
    # 1. GPU 随机采 100 个候选点
    cand_indices = torch.randint(0, len(poi_ids), (u_idx.size(0), 100), device=u_idx.device)
    cand_poi_ids = poi_ids[cand_indices]

    # 2. 计算候选点得分 (用无梯度的快照 X_snapshot)
    u_emb = X_snapshot[u_idx].unsqueeze(1)
    cand_emb = X_snapshot[cand_poi_ids]
    scores = torch.bmm(u_emb, cand_emb.transpose(1, 2)).squeeze(1)

    # 3. 选出最难的 num_neg 个负样本 ID
    _, hard_idx = scores.topk(num_neg, dim=1)
    final_neg_ids = torch.gather(cand_poi_ids, 1, hard_idx)
    return final_neg_ids
user_nodes = [n for n in all_nodes if n.startswith('user_')]
user_ids = torch.tensor([node2id[n] for n in user_nodes])

# POI 节点列表【包含训练集和测试集的 poi 】
poi_nodes = [n for n in all_nodes if n.startswith('poi_')]
poi_ids = torch.tensor([node2id[n] for n in poi_nodes])



# 训练集
# 建立同一个用户访问 poi id 的映射
user_poi_dict = defaultdict(set)
user_pos_pairs = []
for edge in train_hyperedges:
    # 找用户节点
    users_in_edge = [n for n in edge if n.startswith('user_')]
    # 找 POI 节点
    pois_in_edge = [n for n in edge if n.startswith('poi_')]

    user_idx = node2id[users_in_edge[0]]
    poi_idx = node2id[pois_in_edge[0]]
    # 每个用户-POI组合都作为正样本
    user_pos_pairs.append((user_idx, poi_idx))
    user_poi_dict[user_idx].add(poi_idx)

random.shuffle(user_pos_pairs)
logging.info(f"训练集数量：{len(user_pos_pairs)}" )
logging.info(f"示例前10条：{user_pos_pairs[:10]}")




# --------------------------
# Step 4: 推荐函数
# --------------------------
def recommend(user_idx, topk=10):
    u_vec = X[user_idx]
    poi_embed = X[poi_ids]
    scores = (poi_embed @ u_vec).detach().cpu().numpy()

    # 过滤训练集已访问 POI
    visited = user_poi_dict[user_idx]
    for i, poi_idx in enumerate(poi_ids):
        if poi_idx.item() in visited:
            scores[i] = -1e9

    topk_idx = scores.argsort()[-topk:][::-1]
    return [poi_nodes[i] for i in topk_idx]


def evaluate(test_hyperedges, X, node2id, topk=10):
    hits = 0
    total = 0
    for edge in test_hyperedges:
        users = [n for n in edge if n.startswith('user_')]
        pois = [n for n in edge if n.startswith('poi_')]
        for u in users:
            u_idx = node2id.get(u, None)
            # 排除新用户推荐
            if u_idx:
                recommended = recommend(u_idx, topk)
                #logging.info(f"实际：{pois}，推荐：{recommended}")
                # 计算交集
                hits += len(set(pois) & set(recommended))
                total += len(pois)
    if total == 0:
        return total
    recall = hits / total
    return recall

embed_dim = 128
# # 生成随机初始化矩阵
# X0 = nn.Embedding(len(all_nodes), embed_dim) # Embedding 对象
# # 创建模型对象
# hgc = HypergraphConv(embed_dim, embed_dim)
# optimizer = optim.Adam(list(hgc.parameters()) + list(X0.parameters()), lr=0.01)

# 预准备权重矩阵
#init_weights = torch.load('semantic_init.pt')

init_weights = torch.randn(len(all_nodes), embed_dim) # 但是！你又随手造了一个随机矩阵


# 然后再初始化模型
model = HyperGraphModel(len(all_nodes), embed_dim, all_nodes, init_weights=init_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
num_epochs = 1000
topk = 10
num_neg = 20  # 每条正样本采5个负样本
best_recall = 0
patience = 20
wait = 0

batch_size = 512 # 小数据集用 512，全量数据集可以用 1024 或 2048
num_batches = (len(user_pos_pairs) + batch_size - 1) // batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型和参数移动到 GPU
model = model.to(device)
H = H.to(device)
Dv_inv_sqrt = Dv_inv_sqrt.to(device)
De_inv = De_inv.to(device)
# 总节点数：1933，超边 2768
for epoch in range(num_epochs):
    model.train()
    random.shuffle(user_pos_pairs)
    # --- 重点：每轮开始前，先拿一个不带梯度的全局快照 ---
    with torch.no_grad():
        X_snapshot = model(H, Dv_inv_sqrt, De_inv).detach()
    epoch_loss = 0.0

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, len(user_pos_pairs))
        batch_data = user_pos_pairs[start:end]

        if len(batch_data) == 0:
            continue

        optimizer.zero_grad()

        # forward
        X = model(H, Dv_inv_sqrt, De_inv)  # 全图卷积
        X = X.to(device)

        # 批量索引
        u_idx = torch.tensor([p[0] for p in batch_data], device=device)
        pos_idx = torch.tensor([p[1] for p in batch_data], device=device)

        # 多负采样：批量生成负样本
        neg_idx = sample_negatives_gpu(u_idx, num_neg, poi_ids, X_snapshot)

        # 批量 embedding
        user_vecs = X[u_idx]                 # [batch, emb_dim]
        pos_vecs  = X[pos_idx]               # [batch, emb_dim]
        neg_vecs  = X[neg_idx]               # [batch, num_neg, emb_dim]

        # 计算 BPR loss
        loss = bpr_loss_batch(user_vecs, pos_vecs, neg_vecs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # 及时释放 GPU 内存

        del user_vecs, pos_vecs, neg_vecs, u_idx, pos_idx, neg_idx

        torch.cuda.empty_cache()

    scheduler.step()
    epoch_loss /= num_batches

    # ------------------------
    # 评估阶段
    # ------------------------
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            X_eval = model(H, Dv_inv_sqrt, De_inv).to(device)
            recall = evaluate(test_hyperedges, X_eval, node2id, topk=topk)
            recall_20 = evaluate(test_hyperedges, X_eval, node2id, topk=20)
            recall_50 = evaluate(test_hyperedges, X_eval, node2id, topk=50)

        logging.info(
            f"Epoch {epoch} | Loss {epoch_loss:.4f} | "
            f"R@{topk} {recall:.4f} | "
            f"R@20 {recall_20:.4f} | R@50 {recall_50:.4f}"
        )

        # early stopping
        if recall > best_recall:
            best_recall = recall
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logging.info("Early stopping")
                break


