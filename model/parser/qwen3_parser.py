# qwen3_parser.py
import torch
import torch.nn as nn
from modelscope import AutoModelForCausalLM, AutoTokenizer
import json
from model_test.path_config import WSL_DOWNLOAD


class IntentSegModule(nn.Module):
    """Token-level intent segmentation"""
    def __init__(self, hidden_size, num_intent_labels):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_intent_labels)

    def forward(self, hidden_states):
        logits = self.classifier(hidden_states)
        preds = torch.argmax(logits, dim=-1)
        return preds, logits


class NegationHeadModule(nn.Module):
    """Intent-level preference score"""
    def __init__(self, hidden_size):
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1)  # 输出实数分数

    def forward(self, intent_vectors):
        scores = self.scorer(intent_vectors)
        return scores.squeeze(-1)  # [num_intents]


class AttributeAdapterModule(nn.Module):
    """Map intents to KG attributes"""
    def __init__(self, hidden_size, num_attributes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_attributes)

    def forward(self, intent_vectors):
        logits = self.classifier(intent_vectors)
        preds = torch.sigmoid(logits)  # multi-label
        return preds

# -------------------------
# Parser 主类
# -------------------------

class Qwen3Parser:
    def __init__(self, model_path=str(WSL_DOWNLOAD / "Qwen" / "Qwen3-8B"), device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device
        )
        self.model.eval()  # frozen

        hidden_size = self.model.config.hidden_size
        self.num_intent_labels = 10
        self.num_attributes = 42

        self.intent_seg = IntentSegModule(hidden_size, self.num_intent_labels).to(device)
        self.negation_head = NegationHeadModule(hidden_size).to(device)
        self.attribute_adapter = AttributeAdapterModule(hidden_size, self.num_attributes).to(device)

        # KG属性列表
        self.attribute_list = [
            "AcceptsInsurance", "Alcohol", "Ambience.casual", "Ambience.classy",
            "Ambience.divey", "Ambience.hipster", "Ambience.intimate", "Ambience.romantic",
            "Ambience.touristy", "Ambience.trendy", "Ambience.upscale", "BikeParking",
            "BusinessAcceptsBitcoin", "BusinessAcceptsCreditCards", "BusinessParking.garage",
            "BusinessParking.lot", "BusinessParking.street", "BusinessParking.validated",
            "BusinessParking.valet", "ByAppointmentOnly", "Caters", "DogsAllowed",
            "GoodForKids", "GoodForMeal.breakfast", "GoodForMeal.brunch", "GoodForMeal.dessert",
            "GoodForMeal.dinner", "GoodForMeal.latenight", "GoodForMeal.lunch", "HasTV",
            "HappyHour", "NoiseLevel", "OutdoorSeating", "RestaurantsAttire", "RestaurantsDelivery",
            "RestaurantsGoodForGroups", "RestaurantsPriceRange2", "RestaurantsReservations",
            "RestaurantsTableService", "RestaurantsTakeOut", "WheelchairAccessible", "WiFi"
        ]

    @torch.no_grad()
    def encode_user_input(self, user_input):
        tokens = self.tokenizer(user_input, return_tensors="pt").to(self.model.device)
        outputs = self.model.model(**tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        return hidden_states, tokens

    def parse_user_input(self, user_input):
        hidden_states, tokens = self.encode_user_input(user_input)

        # 1. Intent segmentation
        intent_preds, intent_logits = self.intent_seg(hidden_states)

        # 2. Intent vector pool (平均 token hidden states per intent)
        intent_vectors = []
        intent_map = []
        for i in range(self.num_intent_labels):
            mask = (intent_preds[0] == i)
            if mask.sum() == 0:
                continue
            vec = hidden_states[0][mask].mean(dim=0)
            intent_vectors.append(vec)
            intent_map.append(i)
        if len(intent_vectors) == 0:
            return {"positive": [], "negative": []}
        intent_vectors = torch.stack(intent_vectors)

        # 3. Preference score
        preference_scores = self.negation_head(intent_vectors)

        # 4. Attribute mapping
        attr_preds = self.attribute_adapter(intent_vectors)

        # 5. 构建 JSON
        positive = []
        negative = []
        for i, score in enumerate(preference_scores):
            pojo = {
                "poi_name": "",
                "paragraphs": [user_input],
                "attributes": {},
                "score": float(score)
            }
            for j, attr_name in enumerate(self.attribute_list):
                if attr_preds[i][j] > 0.5:
                    pojo["attributes"][attr_name] = "true"
            if score >= 0:
                positive.append(pojo)
            else:
                negative.append(pojo)
        return {"positive": positive, "negative": negative}

# -------------------------
# 测试
# -------------------------
if __name__ == "__main__":
    parser = Qwen3Parser()
    user_text = "I want a quiet place that allows pets but not cafes"
    result = parser.parse_user_input(user_text)
    print(json.dumps(result, indent=2))
