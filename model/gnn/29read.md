好，**一步到位**，我们现在正式进入 **Step 2：Incidence Matrix (H) 构建 + 可直接训练的超图卷积骨架**。
我会非常工程化地来，不讲虚的。

---

# 一、你现在“建模层面”到底在干什么（一句话版）

> 你已经把 **“一次用户–POI–时间–空间–属性的真实访问场景”** 抽象成了一条**超边**。
> 接下来要做的，只是把这些超边**数学化**，变成超图卷积能吃的结构。

---

# 二、Incidence Matrix (H) 是什么（别被名字吓到）

### 定义（你论文里就这么写）

* 节点集合：
  [
  V = {v_1, v_2, \dots, v_{|V|}},\quad |V| = 1933
  ]

* 超边集合：
  [
  E = {e_1, e_2, \dots, e_{|E|}}
  ]

* **关联矩阵（Incidence Matrix）**：
  [
  H \in {0,1}^{|V| \times |E|}
  ]

[
H(v,e) =
\begin{cases}
1 & \text{if node } v \in \text{hyperedge } e \
0 & \text{otherwise}
\end{cases}
]

👉 **没有任何“用户偏好”“推荐逻辑”在这里**
👉 就是：**谁出现在哪条超边里**

---

# 三、Step 2.1：节点 & 超边编号（必须做）

## 1️⃣ 节点编号

```python
# all_nodes: 你刚刚统计出来的 1933 个节点
node2id = {node: idx for idx, node in enumerate(all_nodes)}
id2node = {idx: node for node, idx in node2id.items()}
```

---

## 2️⃣ 超边编号

```python
edge2id = {i: i for i in range(len(hyperedges))}
```

---

# 四、Step 2.2：构建 Incidence Matrix H（稀疏版）

⚠️ **重点**：

* 不能用 dense matrix（会浪费）
* 必须用 **COO / sparse tensor**

---

## 1️⃣ 构建 COO 索引

```python
import torch

row_idx = []  # node index
col_idx = []  # hyperedge index

for e_id, edge in enumerate(hyperedges):
    for node in edge:
        row_idx.append(node2id[node])
        col_idx.append(e_id)
```

---

## 2️⃣ 生成 PyTorch Sparse Tensor

```python
indices = torch.tensor([row_idx, col_idx], dtype=torch.long)
values = torch.ones(len(row_idx))

num_nodes = len(all_nodes)
num_edges = len(hyperedges)

H = torch.sparse_coo_tensor(
    indices,
    values,
    size=(num_nodes, num_edges)
)
```

✔️ **这一步完成后，你的超图在数学上已经成立**

---

# 五、Step 2.3：计算超图卷积所需的三个核心矩阵（论文硬核）

这是 **二区 reviewers 真正看的地方**。

---

## 1️⃣ 节点度矩阵 (D_v)

[
D_v(i,i) = \sum_e H(i,e)
]

```python
Dv = torch.sparse.sum(H, dim=1).to_dense()
Dv_inv_sqrt = torch.pow(Dv, -0.5)
Dv_inv_sqrt[torch.isinf(Dv_inv_sqrt)] = 0
```

---

## 2️⃣ 超边度矩阵 (D_e)

[
D_e(e,e) = \sum_v H(v,e)
]

```python
De = torch.sparse.sum(H, dim=0).to_dense()
De_inv = torch.pow(De, -1)
De_inv[torch.isinf(De_inv)] = 0
```

---

## 3️⃣ 超图传播算子（核心公式）

你论文里直接写这个：

[
\mathbf{X}^{(l+1)}
==================

D_v^{-\frac{1}{2}}
H
D_e^{-1}
H^\top
D_v^{-\frac{1}{2}}
\mathbf{X}^{(l)}
W^{(l)}
]

---

# 六、Step 2.4：Hypergraph Convolution 层（可直接用）

```python
class HypergraphConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, X, H, Dv_inv_sqrt, De_inv):
        # X: [num_nodes, in_dim]

        X = Dv_inv_sqrt.unsqueeze(1) * X
        X = torch.sparse.mm(H.t(), X)
        X = De_inv.unsqueeze(1) * X
        X = torch.sparse.mm(H, X)
        X = Dv_inv_sqrt.unsqueeze(1) * X

        return self.linear(X)
```

---

# 七、Step 2.5：节点初始特征（你现在可以先这样）

你现在 **不需要纠结特征工程**，先跑通。

```python
embed_dim = 64
X0 = torch.nn.Embedding(num_nodes, embed_dim).weight
```

---

# 八、现在你的“实验结构”已经是完整的

你已经拥有：

| 组件               | 状态     |
| ---------------- | ------ |
| 时空-属性超边          | ✅ 完成   |
| 节点集合             | ✅ 1933 |
| Incidence Matrix | ✅ 稀疏   |
| 超图卷积公式           | ✅ 二区级  |
| 冷启动结构建模          | ✅ 天然支持 |

---

# 九、下一步你**必须做的**（我来带）

👉 **Step 3：定义训练目标（Pairwise Ranking / BPR）**
👉 把 **user–POI 投影出来**，真正变成“推荐系统”

下一步我可以直接帮你做：

* 🎯 **如何从超图 embedding 中取 user / POI**
* 🎯 **BPR Loss + 负采样**
* 🎯 **为什么这一步解决评论稀疏 POI 的结构性冷启动**

你一句话：
**「继续 Step 3」**
我就接着来，不停。
