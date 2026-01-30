from pathlib import Path


# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent
# 数据目录

DATA_DIR = ROOT_DIR / "data"
YELP_DIR = ROOT_DIR / "data" / "yelp"
YELP_DATA_DIR = YELP_DIR / "data"
OUT_DIR = DATA_DIR / "out"
VECTORS_DIR = YELP_DIR / "out"

DOWNLOAD_DIR =  ROOT_DIR / "download"

MODEL_WSL_HOME = Path("/root/autodl-tmp")
MODEL_WSL_NAME = "Qwen/Qwen3-8B"
MODEL_WSL_DOWNLOAD = MODEL_WSL_HOME / "models" / "download"

