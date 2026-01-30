#模型下载
from modelscope import snapshot_download
from path_config import MODEL_WSL_DOWNLOAD, MODEL_WSL_NAME
model_dir = snapshot_download(
    MODEL_WSL_NAME,
    cache_dir=str(MODEL_WSL_DOWNLOAD)  # ✅ 自定义模型下载位置
)