# 整理数据：不符合 python 要求的数据类型以及格式，例如 {} 字典字符串，
数据清洗的目的是将 Yelp 原始 JSON 数据中非标准格式（如字符串形式的字典、True/False、u'xxx' 字符串等）转换为严格的结构化数据，
以便后续将 POI、属性、类别等信息构建为知识图谱节点与三元组关系。


categories:" A, B, C"  => [A, B, C]

"True", "False", => true, false => True, False

"None","none" => "null" => None

"13:0-15.30" => "13:00-15.30"

"yes", "free", "'yes'" => True

"no", "'no'" => False


# 属性筛选

属性筛选采用“出现频率筛选 + 推荐目标相关性”，
保留出现频率至少占 5% 的属性，并进一步根据 POI 推荐任务保留对用户体验和偏好相关的属性，如 Ambience、PriceRange、GoodForMeal 等。
数据文件：attribute_ratio.csv


# 清理完毕
保证最终的结果。开始创建 KG
