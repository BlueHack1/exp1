import json
import logging
from collections import Counter

def abstract(path):
    counter = Counter()
    poi_number = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                poi = json.loads(line)
                attrs = poi.get("attributes", {})
                if attrs:
                    poi_number = poi_number + 1
                    counter.update(attrs.keys())
            except json.JSONDecodeError:
                logging.info(f"跳过无效行:{line}", )
                continue

    # 筛选出现次数超过阈值的属性
    # 计算阈值 = 出现频率 >= 5%
    threshold = poi_number * 0.05

    return [k for k, v in counter.items() if v >= threshold] , [k for k, v in counter.items() if v < threshold]



if __name__ == '__main__':
    # 为什么选择 0.05。
    import pandas as pd


    # 递归扁平化 dict
    def flatten_dict(d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items


    df = pd.read_json("cleaned_business.jsonl", lines=True)

    # 扁平化 attributes（dict 本身已是 JSON，不用 parse）
    df_attr = pd.json_normalize(df['attributes'], sep='.')

    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 计算每个子属性出现率
    ratio = df_attr.notnull().mean().sort_values(ascending=False)
    logging.info(ratio)
    ratio.to_csv("attribute_ratio.csv")