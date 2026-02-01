import logging
import time

from string import Template

from tqdm import tqdm

from exp.config import YELP_DATA_BUSINESS_WSL, YELP_DIR_WSL, YELP_DATA_REVIEW_WSL, YELP_DATA_USER

import json

# from model_test.download.interface import Qwen3Stream

'''
    通过用户历史访问地点，构建用户兴趣。
    用户兴趣 json 格式：
    user:{
        'poi':[
            {
                'business_id': review.get("business_id"),
                'name': business.get("name"),
                'text': text,
                'p': [],
                'n': []
            }
        ]
        # 用户静态兴趣变量，表征用户跨长期历史行为形成的稳定偏好，不依赖当前查询，可低频更新
        'interest_static_vec': [],
        # 用户动态兴趣变量，结合用户当前查询
        'interest_dynamic_vec': []
    
    }

'''
business_city = {}
user_interest = {}  # 收集用户 ID
# qwen3 = Qwen3Stream.get_instance()


def extract():
    # 取出来全部的 business_id =》 category
    with open(YELP_DATA_BUSINESS_WSL / "reviewed_business.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            business_city[b["business_id"]] = {
                'name': b["name"],
                'category': b["categories"],
            }
    logging.info(f"提取 business 实体：{len(business_city)}")

    with open(YELP_DATA_REVIEW_WSL / "sifted_review.jsonl", "r", encoding="utf-8") as f_in:
        for num, line in enumerate(f_in):
            logging.info(f"正在执行第 {num + 1} 个")
            logging.info(f"获取 {len(user_interest)} 个")
            review = json.loads(line)
            uid = review.get("user_id")
            business = business_city.get(review.get("business_id"))
            category = business.get("category")
            text = review.get("text")
            if len(user_interest) >= 1000000:
                # 数量超出，但保证遍历完毕 已存在的用户 的 访问
                if uid not in user_interest:
                    continue
            if uid:
                user_interest.setdefault(uid, {'poi': []})
                pois = user_interest[uid]['poi']
                # 超出 5 个 poi 不继续筛选了。
                if len(pois) > 5:
                    logging.info(f"{uid}:超出50个")
                    continue
                start = time.perf_counter()
                # pn = sentiment(text, category)
                # logging.info(f"{text}\n获取的结果是：{pn}")
                pois.append(
                    {
                        'business_id': review.get("business_id"),
                        'name': business.get("name"),
                        'date': review.get("date"),
                        'stars': review.get("stars"),
                        'text': text,
                        # 'p': pn['p'],
                        # 'n': pn['n']
                    }
                )
                end = time.perf_counter()
                logging.info(f"执行时间: {end - start:.6f} 秒")
    # 写出到 JSONL 文件（每个用户一个 JSON 行）
    output_path = YELP_DATA_USER / "user_interest.jsonl"
    with open(output_path, "w", encoding="utf-8") as f_out:
        batch = []
        for uid, data in user_interest.items():
            batch.append(json.dumps({"user_id": uid, "poi": data["poi"]}, ensure_ascii=False))

            if len(batch) >= 1000:
                f_out.write("\n".join(batch) + "\n")
                batch = []
        # 最后一批
        if batch:
            f_out.write("\n".join(batch) + "\n")
    logging.info("筛选结束 user_interest.jsonl")


def sentiment(des, cats):
    qwen3 = None
    content = qwen3.generate_text(Template("""
You are a sentiment classification assistant.

Task:
Given a "description" and a list of "categories", determine which categories are
mentioned or semantically related in the description, and classify each of them
as positive or negative.

Semantic matching is allowed: if the description clearly refers to something that
belongs to a category’s meaning, treat it as mentioned. If no clear relevance,
ignore the category.

Output must be ONLY the following strict JSON:
{"p":[], "n":[]}

Rules:
1) Positive meaning → add to "p"; negative meaning → add to "n".
2) If both appear, choose the dominant sentiment.
3) Categories not mentioned or unrelated → ignore.
4) No explanation, no extra text, no commentary.
5) Always output valid JSON.

Now classify the following:
Description: $des
Categories: $cats


        """).substitute(des=des, cats=cats))
    result = json.loads(content)
    if not result.get("p"):
        result["p"] = []
    elif not result.get("n"):
        result["n"] = []
    return result;

if __name__ == '__main__':
    extract()