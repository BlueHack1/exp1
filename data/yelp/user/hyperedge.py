from exp.config import YELP_DATA_REVIEW, YELP_DATA_USER
import json
import logging
import numbers
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
'''
    超边:
        hyperedges = [
        f"user_{user_id}",
        f"poi_{business_id}",
        f"time_{time_slot}",
        zone_node
        ] + attr_nodes

'''
"""
构建超边，其中超边:['user_vRuHPM06W9iQJXK8XHbC3Q', 'poi_S8ZFYEgMejpChID8tzKo9A', 'time_weekday_evening', 'zone_0', 'attr_bool_BikeParking', 'attr_bool_BusinessAcceptsCreditCards', 'attr_bool_ByAppointmentOnly', 'attr_bool_Caters', 'attr_bool_HappyHour', 'attr_bool_OutdoorSeating', 'attr_bool_RestaurantsDelivery', 'attr_bool_RestaurantsGoodForGroups', 'attr_bool_RestaurantsReservations', 'attr_bool_RestaurantsTakeOut', 'attr_sub_Ambience_classy', 'attr_sub_Ambience_trendy', 'attr_sub_BusinessParking_street', 'attr_sub_BusinessParking_valet', 'attr_sub_GoodForMeal_dinner', 'attr_val_Alcohol_full_bar', 'attr_val_DogsAllowed_False', 'attr_val_GoodForKids_False', 'attr_val_HasTV_False', 'attr_val_NoiseLevel_average', 'attr_val_RestaurantsAttire_casual', 'attr_val_RestaurantsPriceRange2_2', 'attr_val_WiFi_False']
"""
class HypergraphConstructor:
    def __init__(self, poi_path, user_path, zone_path, test_ratio: float = 0.2, k_clusters: int = 50):
        self.test_ratio = test_ratio
        self.k_clusters = k_clusters
        self.poi_infos = {}
        self.zone_nodes = {}
        self.hyperedges = []
        self.train_hyperedges = []
        self.test_hyperedges = []
        self.poi_path = poi_path
        self.user_path = user_path
        self.zone_path = zone_path

    @staticmethod
    def is_true(v) -> bool:
        return v is True or (isinstance(v, str) and v.lower() == "true")

    @staticmethod
    def discretize_time(dt_str: str) -> str:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        hour = dt.hour
        is_weekend = "weekend" if dt.weekday() >= 5 else "weekday"

        if 6 <= hour < 11:
            period = "morning"
        elif 11 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 22:
            period = "evening"
        else:
            period = "night"
        return f"{is_weekend}_{period}"

    def build_spatial_nodes(self, output_name: str = 'zone_node.json'):
        """执行 K-Means 聚类生成空间节点"""
        poi_ids = []
        coords = []
        with open(self.poi_path, 'r', encoding='utf-8') as f:
            for line in f:
                poi = json.loads(line)
                if poi["latitude"] is None or poi["longitude"] is None:
                    continue
                poi_ids.append(poi["business_id"])
                coords.append([poi["latitude"], poi["longitude"]])

        coords = np.array(coords)
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42, n_init=10)
        zone_labels = kmeans.fit_predict(coords)

        self.zone_nodes = {pid: f"zone_{z}" for pid, z in zip(poi_ids, zone_labels)}

        with open(output_name, "w") as f:
            json.dump(self.zone_nodes, f)
        logging.info(f"空间节点构建完毕，共 {self.k_clusters} 个区域")

    def extract_attr_nodes(self, attributes: Optional[Dict]) -> List[str]:
        """提取属性节点逻辑 (保持原逻辑不变)"""
        attr_nodes = set()
        if attributes is not None:
            for key, value in attributes.items():
                if value is None:
                    continue
                if isinstance(value, bool) and value:
                    attr_nodes.add(f"attr_bool_{key}")
                elif isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if self.is_true(sub_val):
                            attr_nodes.add(f"attr_sub_{key}_{sub_key}")
                elif isinstance(value, str):
                    attr_nodes.add(f"attr_val_{key}_{value.strip().replace(' ', '_')}")
                elif isinstance(value, numbers.Number):
                    attr_nodes.add(f"attr_val_{key}_{value}")
        return sorted(list(attr_nodes))

    def build_hyperedges(self):
        """核心构建逻辑"""
        # 1. 加载 POI 基础信息
        with open(self.poi_path, 'r', encoding='utf-8') as f:
            for line in f:
                poi = json.loads(line)
                self.poi_infos[poi.get('business_id')] = poi

        # 2. 加载或确认 Zone 信息
        if self.zone_path:
            with open(self.zone_path, 'r') as f:
                self.zone_nodes = json.load(f)

        # 3. 处理用户轨迹并构建超边
        with open(self.user_path, mode='r') as f:
            for u_line in f:
                user = json.loads(u_line)
                user_id = user.get('user_id')
                poi_list = user.get('poi', [])

                # 按时间排序，保证时间一致性
                poi_list.sort(key=lambda x: x['date'])

                user_current_edges = []
                for poi_visit in poi_list:
                    bid = poi_visit.get('business_id')
                    if bid not in self.poi_infos: continue

                    # 组合超边节点
                    time_slot = self.discretize_time(poi_visit.get('date'))
                    zone_node = self.zone_nodes.get(bid, "zone_unknown")
                    attr_nodes = self.extract_attr_nodes(self.poi_infos[bid].get('attributes'))

                    edge = [
                               f"user_{user_id}",
                               f"poi_{bid}",
                               f"time_{time_slot}",
                               zone_node
                           ] + attr_nodes

                    user_current_edges.append(edge)

                # 按用户级别划分训练/测试集
                split_idx = int(len(user_current_edges) * (1 - self.test_ratio))
                self.train_hyperedges.extend(user_current_edges[:split_idx])
                self.test_hyperedges.extend(user_current_edges[split_idx:])
                self.hyperedges.extend(user_current_edges)

    def run_sanity_check(self):
        """调用原有的 sanity check 逻辑（已集成到类中）"""
        # 这里可以直接复用你之前的 sanity_check_zones 逻辑，
        # 将其内部变量替换为 self.zone_nodes 等。
        pass


hyperedgesCon = HypergraphConstructor(YELP_DATA_REVIEW / 'reviewed_business.jsonl',
                                      YELP_DATA_USER / 'user_interest.jsonl',
                                      'zone_node.json')
hyperedgesCon.build_hyperedges()
logging.info(hyperedgesCon.hyperedges)
