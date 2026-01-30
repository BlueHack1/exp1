import json
import ast
import logging


def normalize_value(value, key = None):
    """
    递归处理 dict / list / 字符串，将其转换为 JSON 合法结构
    """
    if value is None:
        return None

    # 如果是 dict，递归各字段
    if isinstance(value, dict):
        return {k: normalize_value(v, k) for k, v in value.items()}

    # 如果是 list，递归处理列表内容
    if isinstance(value, list):
        return [normalize_value(v) for v in value]

    # 如果是字符串，尝试解析成 dict 或原值
    if isinstance(value, str):
        # ✅ None / 'none' / "none" / null 统一为 None

        v = value.strip()
        # 空字符串
        if v == '':
            return None
        # u'xxx' -> xxx
        if v.startswith("u'") and v.endswith("'"):
            v = v[2:-1]

        # 单引号包起来的字符串 "'average'" -> average
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            # 去掉最外层一层引号
            v = v[1:-1]



        # 补全时间
        if "-" in v and ":" in v and "{" not in v:

            start, end = v.split("-")

            def pad_hm(hm):
                if ":" not in hm:
                    return hm
                h, m = hm.split(":")
                h = h.zfill(2)  # 小时补全 2 位
                m = m.zfill(2)  # 分钟补全 2 位
                return f"{h}:{m}"
            return f"{pad_hm(start)}-{pad_hm(end)}"

        if key == 'categories':
            return  [c.strip() for c in value.split(",") if c.strip()]




        # wifi 一些可能有 no yes
        if v.lower() == 'no':
            return False
        if v.lower() == 'yes' or v.lower() == 'free':
            return True

        #  尝试转换 dict-like 字符串，例如 "{'a': True}"

        if v.startswith("{") or v.startswith("["):
            try:
                parsed = ast.literal_eval(v)
                return normalize_value(parsed)
            except Exception:
                pass


        #  JSON 替换（仅替换 dict 情况）
        v_json = (
            v.replace("'", '"')
            .replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
            .replace('none', "null")
        )

        try:
            # 解析成 py 对象
            parsed = json.loads(v_json)
            return normalize_value(parsed)
        except Exception:
            return v  # 不再返回 v_json，避免 "\"xxx\"" 问题

    # 其他类型 和 上面处理完毕返回，返回的是 python 数据类型
    return value




def clean_jsonl_file(input_path, output_jsonl_path=None):
    """
    读取 JSONL 文件，每行一个 JSON 对象，清洗后写入 JSONL 或 JSON 文件
    """
    cleaned_pois = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                poi = json.loads(line)
            except json.JSONDecodeError:
                logging.info("跳过无效行:", line)
                continue

            cleaned = normalize_value(poi)
            cleaned_pois.append(json.dumps(cleaned, ensure_ascii=False))

    # 写 JSONL
    if output_jsonl_path:
        with open(output_jsonl_path, "w", encoding="utf-8") as f:
            for poi in cleaned_pois:
                f.write(poi + "\n")


    return cleaned_pois


pois = [{"business_id": "Js3m_GdqNUGnEgXJ8WMtfQ", "name": "Hana Kitchen", "address": "503 State St",
         "city": "Santa Barbara", "state": "CA", "postal_code": "93101", "latitude": 34.416504,
         "longitude": -119.6956177, "stars": 3.0, "review_count": 253, "is_open": 1,
         "attributes": {"RestaurantsDelivery": "True", "DogsAllowed": "False", "RestaurantsAttire": "'casual'",
                        "WheelchairAccessible": "True", "RestaurantsPriceRange2": "1",
                        "RestaurantsGoodForGroups": "False", "BusinessAcceptsBitcoin": "False",
                        "BusinessAcceptsCreditCards": "True", "RestaurantsTableService": "False", "Caters": "True",
                        "GoodForMeal": "{'dessert': False, 'latenight': True, 'lunch': False, 'dinner': True, 'brunch': False, 'breakfast': False}",
                        "Alcohol": "u'beer_and_wine'", "RestaurantsReservations": "False", "BikeParking": "True",
                        "BusinessParking": "{'garage': False, 'street': True, 'validated': False, 'lot': True, 'valet': False}",
                        "OutdoorSeating": "True", "GoodForKids": "True", "RestaurantsTakeOut": "True",
                        "NoiseLevel": "'average'",
                        "Ambience": "{'touristy': False, 'hipster': False, 'romantic': False, 'divey': False, 'intimate': False, 'trendy': False, 'upscale': False, 'classy': False, 'casual': True}",
                        "HasTV": "True", "WiFi": "u'free'", "HappyHour": "True"},
         "categories": "Bubble Tea, Restaurants, Food, Asian Fusion, Vegan, Acai Bowls",
         "hours": {"Monday": "11:0-21:0", "Tuesday": "11:0-21:0", "Wednesday": "11:0-21:0", "Thursday": "11:0-21:0",
                   "Friday": "11:0-2:0", "Saturday": "11:0-2:0", "Sunday": "11:0-21:0"}},
        {"name": "Lunch Spot", "attributes": {"GoodForMeal": {"dessert": True, "latenight": False}}}]

# def clean_attributes(poi_list):
#     """遍历 POI 列表，递归规范 attributes 字段"""
#     return [normalize_value(poi) for poi in poi_list]
# # 测试
# cleaned_pois = clean_attributes(pois)
# print(json.dumps(cleaned_pois, indent=2, ensure_ascii=False))








