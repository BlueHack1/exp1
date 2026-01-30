import json

from abstract_attr import abstract as abstract
from clean_json import clean_jsonl_file
import json
import logging
from pathlib import Path

import pandas as pd
from exp.config import YELP_DATA_BUSINESS, YELP_DATA_REVIEW, YELP_DIR, YELP_DIR_WSL, YELP_DATA_BUSINESS_WSL, \
    YELP_DATA_REVIEW_WSL, YELP_DATA_USER_WSL


def sifted_city(city_name):
    '''
        筛选指定地点（拉斯维加斯） 且 正常营业地点  作为数据集
    '''
    folder = Path(YELP_DATA_BUSINESS)
    folder.mkdir(parents=True, exist_ok=True)

    # 1. 加载Business数据并过滤出Las Vegas的POI
    biz_data = []
    with open(YELP_DIR_WSL / 'yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            # 过滤条件：城市为拉斯维加斯，且是营业状态的开店
            if record.get('city') == city_name and record.get('is_open') == 1:
                biz_data.append(record)
    if len(biz_data) == 0:
        logging.info("筛选失败")
    else:
        # 转成按行显示的 dict，类似 excel，更方便字典访问。
        biz_df = pd.DataFrame(biz_data)

        # 获取Las Vegas的所有POI ID列表
        vegas_business_ids = set(biz_df['business_id'])

        # 2. 加载Review数据，但只保留那些评论了Vegas POI的评论
        review_data = []
        max_reviews = 100000  # 可选：限制评论数量，避免内存占用过大
        count = 0
        with open(YELP_DIR_WSL / 'yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if record['business_id'] in vegas_business_ids:
                    review_data.append(record)
                    count += 1
                    # 可以在这里加一个计数器，比如收集50万条评论后break，以控制规模
                    if count >= max_reviews:
                        break

        review_df = pd.DataFrame(review_data)

        biz_df.to_json(YELP_DATA_BUSINESS_WSL / "sifted_business.jsonl", orient="records", lines=True,
                       force_ascii=False)
        review_df.to_json(YELP_DATA_REVIEW_WSL / "sifted_review.jsonl", orient="records", lines=True, force_ascii=False)
        logging.info(f"共提取 {len(biz_data)}，用户评论：{len(review_data)}")
        logging.info(f'保存完成：'
                     f'\n{YELP_DATA_BUSINESS_WSL / 'sifted_business.jsonl'}'
                     f'\n{YELP_DATA_REVIEW_WSL / "sifted_review.jsonl"}')

    # 筛选用户
    user_ids = set()  # 收集用户 ID
    with open(YELP_DATA_REVIEW_WSL / "sifted_review.jsonl", "r", encoding="utf-8") as f_in:
        for line in f_in:
            review = json.loads(line)
            uid = review.get("user_id")
            if uid:
                user_ids.add(uid)
    logging.info(f"review ==> 筛选后 user 数量: {len(user_ids)}")
    logging.info(f"开始提取相关 user 信息")
    # -----------------------------
    # 2. 筛选用户 user，只保留和目标城市 POI 相关的
    # -----------------------------
    filtered_user = []
    with open(YELP_DIR_WSL / "yelp_academic_dataset_user.json", "r", encoding="utf-8") as f_in:
        for line in f_in:
            user = json.loads(line)
            if user.get("user_id") in user_ids:
                filtered_user.append(user)
    logging.info(f"提取user完毕 {len(filtered_user)} 相关信息")

    # -----------------------------
    # 3. 输出到 jsonl 文件
    # -----------------------------
    output_path = YELP_DATA_USER_WSL / "filtered_users.jsonl"
    with open(output_path, "w", encoding="utf-8") as f_out:
        for user in filtered_user:
            f_out.write(json.dumps(user, ensure_ascii=False) + "\n")
    logging.info(f"user 结果已保存到: {output_path}")


if __name__ == '__main__':
    # 筛选城市 及其 评论
    sifted_city('Philadelphia')


    jsoned_path = YELP_DATA_BUSINESS_WSL / 'jsoned_business.jsonl'
    filtered_path = YELP_DATA_BUSINESS_WSL /'filtered_business.jsonl'

    clean_jsonl_file(YELP_DATA_BUSINESS_WSL / 'sifted_business.jsonl', jsoned_path)
    attrs_name, failed_name = abstract(jsoned_path);

    logging.info(f"筛选后的属性：{attrs_name}", )
    logging.info(f"删除的属性：{failed_name}", )

    with open(jsoned_path, "r", encoding="utf-8") as fin, \
            open(filtered_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                poi = json.loads(line)
                attrs: dict = poi.get("attributes", {})
                if attrs:
                    # 过滤掉指定 key
                    poi["attributes"] = {k: v for k, v in attrs.items() if k not in failed_name}
                # 写回文件，每行一个 JSON
                fout.write(json.dumps(poi, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                logging.info(f"跳过无效行:{line}",)
                continue
