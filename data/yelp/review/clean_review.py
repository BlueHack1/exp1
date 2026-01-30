import json
import logging
import re
from datetime import datetime
from collections import defaultdict
from exp.config import YELP_DATA_BUSINESS_WSL, YELP_DATA_REVIEW_WSL, YELP_DATA_BUSINESS
from model_test.download.interface import Qwen3Stream


# -----------------------------
# 文本清洗函数
def clean_review_text(text):
    """
    清洗评论文本：
    - 去掉 HTML 标签
    - 去掉换行/制表符
    - 去多余空格
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -----------------------------
# 评论评分函数（用于排序）
def compute_review_score(review):
    """
    综合评分：
    - useful: 社区认可度
    - rating: 偏离中立星级(3)
    - recency: 距今天数倒数
    """
    useful = review.get('useful', 0)
    rating = review.get('stars', 3)
    recency = 0
    if 'date' in review:
        try:
            days_ago = (datetime.now() - datetime.strptime(review['date'][:10], "%Y-%m-%d")).days
            recency = 1 / (days_ago + 1)
        except:
            recency = 0
    score = 0.5 * useful + 0.3 * abs(rating - 3) + 0.2 * recency
    return score


# -----------------------------
# Top-K 评论选择函数
def select_top_k_reviews(reviews, K=5):
    """
    输入: POI 的所有评论列表
    输出: Top-K 评论文本列表
    """
    if not reviews:
        return []

    # 清洗文本 & 评分
    for r in reviews:
        r['text'] = clean_review_text(r.get('text', ''))
        r['score'] = compute_review_score(r)

    # 分正负向评论
    positive = [r for r in reviews if r.get('stars', 3) >= 4]
    negative = [r for r in reviews if r.get('stars', 3) <= 2]

    top_pos = sorted(positive, key=lambda x: x['score'], reverse=True)[:K // 2]
    top_neg = sorted(negative, key=lambda x: x['score'], reverse=True)[:K // 2]

    top_reviews = top_pos + top_neg

    # 补齐到 K 条
    remaining = [r for r in reviews if r not in top_reviews]
    remaining_sorted = sorted(remaining, key=lambda x: x['score'], reverse=True)
    while len(top_reviews) < K and remaining_sorted:
        top_reviews.append(remaining_sorted.pop(0))

    return [r['text'] for r in top_reviews[:K]]


# -----------------------------
# 主 pipeline: JSONL 评论 → POI 描述字典
def generate_poi_descriptions(review_jsonl_path, poi_jsonl_path, K=5):
    model = Qwen3Stream()

    #  按 POI 聚合评论
    poi_reviews = defaultdict(list)
    with open(review_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            '''
                dict {
                    business_id: [
                        {
                            review_data
                        }
                    ]
                }
            '''
            review = json.loads(line)
            poi_reviews[review.get("business_id")].append(review)
    logging.info(f"加载 review {len(poi_reviews)} 条")


    current_num = 0
    with open(poi_jsonl_path, 'r', encoding='utf-8') as fin, \
            open(YELP_DATA_BUSINESS / "reviewed_business.jsonl", 'w', encoding='utf-8') as fout:

        for line in fin:
            current_num = current_num + 1
            logging.info(f"正在写入第 {current_num} 条")
            poi = json.loads(line)
            business_id = poi['business_id']

            # 获取对应 POI 的评论（外部提前构造好的 dict）
            reviews = poi_reviews.get(business_id, [])

            # 选择 Top-K 评论
            top_reviews = select_top_k_reviews(reviews, K=K)

            # 构建 LLM 输入
            poi_description = (
                f"POI 名称: {poi.get('name', '')}\n"
                f"POI 地址: {poi.get('address', '')}\n"
                f"类别: {', '.join(poi.get('categories', []) or [])}\n"
                f"主要属性: {poi.get('attributes', {})}\n"
                f"用户评论摘要:\n{'\n'.join(top_reviews)}"
            )

            # 调用 LLM 生成英文 POI 描述
            result = model.generate(
                f"""
    You are a professional POI description generator.

    Only use the provided information below. Do NOT add any details that are not explicitly mentioned.
    POI structured data and user reviews:
    {poi_description}

    Generate a concise English description (1-3 sentences) highlighting the features and user experience strictly based on this information.
    """,
                enable_thinking=False,
                max_new_tokens=50,
                do_sample=False,
            )

            print(f"\n生成后的描述：{result}")
            poi['description'] = result

            # ⚠️ 直接写入，不再存入列表
            fout.write(json.dumps(poi, ensure_ascii=False) + '\n')


    return "result"


# -----------------------------
# 示例使用
if __name__ == "__main__":
    review_jsonl_path = YELP_DATA_REVIEW_WSL / "sifted_review.jsonl"
    poi_jsonl_path = YELP_DATA_BUSINESS_WSL / "filtered_business.jsonl"
    poi_descriptions = generate_poi_descriptions(review_jsonl_path, poi_jsonl_path, K=5)
