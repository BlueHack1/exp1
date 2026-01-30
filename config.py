import logging
from pathlib import Path

# source /root/.virtualenvs/project/bin/activate
# 项目根目录 (config.py 和 data、src 在同一级目录)
ROOT_DIR = Path(__file__).resolve().parent

CURRENT_ENV = 'local'  # wsl
if CURRENT_ENV == 'wsl':
    # 本地代理 到 wsl
    import os

    os.environ["HTTP_PROXY"] = "http://192.168.31.194:7897"
    os.environ["HTTPS_PROXY"] = "http://192.168.31.194:7897"
    import subprocess

    # 模型文件同步
    subprocess.run("rsync -avh --delete /mnt/d/exp/zh_kg_llm/project/model_test/download/ ~/models/download/",
                   shell=True)

    # 数据集同步
    subprocess.run("mkdir -p ~/exp/data", shell=True)
    subprocess.run("rsync -avh --delete /mnt/d/exp/zh_kg_llm/project/exp/data/ ~/exp/data/", shell=True)
    subprocess.run("rsync -avh --delete /mnt/d/exp/zh_kg_llm/project/exp/vectorization/ ~/exp/vectorization/",
                   shell=True)
    WSL_HOME = Path("/root/exp")
    BERT_MODEL = 'sentence-transformers/all-MPNet-base-v2'
elif CURRENT_ENV == 'autoDL':
    # autoDL 环境下， 不存在虚拟机，读取全在云服务器上
    ROOT_DIR = Path("/root/autodl-tmp")
    WSL_HOME = Path("/root/autodl-tmp")
    BERT_MODEL = str(WSL_HOME / '/all-MPNet-base-v2')
else:
    # 本地 环境下， 读取全在云服务器上
    ROOT_DIR = Path(__file__).resolve().parent
    WSL_HOME = Path(__file__).resolve().parent
    BERT_MODEL = 'sentence-transformers/all-MPNet-base-v2'


def ensure_dirs(*paths):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


# 数据目录
DATA_DIR = ROOT_DIR / "data"
YELP_DIR = ROOT_DIR / "data" / "yelp"
YELP_DATA_BUSINESS = YELP_DIR / "business"
YELP_DATA_USER = YELP_DIR / "user"
YELP_DATA_REVIEW = YELP_DIR / "review"

VECTORS_DIR = YELP_DIR / "vec_out"
VECTORS_POI_DIR = ROOT_DIR / "vectorization" / "poi"
VECTORS_QUERY_DIR = DATA_DIR / "vectorization" / "query"

GNN_DIR = DATA_DIR / "kg" / "gnn" / "train"

ensure_dirs(
    DATA_DIR, YELP_DIR, YELP_DATA_BUSINESS,
    YELP_DATA_USER, YELP_DATA_REVIEW, VECTORS_DIR
)

# 读取的时候 __init__ 文件同步 本地文件到 wsl，  写入的时候写入本机

DATA_DIR = WSL_HOME / "data"
YELP_DIR_WSL = WSL_HOME / "data" / "yelp"
YELP_DATA_BUSINESS_WSL = YELP_DIR / "business"
YELP_DATA_USER_WSL = YELP_DIR / "user"
YELP_DATA_REVIEW_WSL = YELP_DIR / "review"

VECTORS_POI_DIR_WSL = WSL_HOME / "vectorization" / "poi"
VECTORS_QUERY_DIR_WSL = WSL_HOME / "vectorization" / "query"

GNN_DIR_WSL = DATA_DIR / "kg" / "gnn" / "train"

# 放入列表，统一创建
dirs_to_create = [
    DATA_DIR,
    YELP_DIR_WSL,
    YELP_DATA_BUSINESS_WSL,
    YELP_DATA_USER_WSL,
    YELP_DATA_REVIEW_WSL,
    VECTORS_POI_DIR_WSL,
    VECTORS_QUERY_DIR_WSL,
    # GNN_DIR_WSL
]

for d in dirs_to_create:
    d.mkdir(parents=True, exist_ok=True)

logging.info("所有目录已创建或已存在。")

NEO4J_URL = "bolt://192.168.31.194:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "983496745"

logging.info("全局变量配置成功")
