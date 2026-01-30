# source /root/.virtualenvs/project/bin/activate
import logging

# 只需要执行一次，全局生效
logging.basicConfig(
    level=logging.INFO,  # 输出 INFO 及以上级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("日志启动成功")

