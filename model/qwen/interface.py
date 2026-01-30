import logging
from path_config import MODEL_WSL_DOWNLOAD, MODEL_WSL_NAME

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import threading
from string import Template

THINK_TOKEN_ID = 151668  # </think> token


class Qwen3Stream:
    _instance = None  # 单例

    @classmethod
    def get_instance(cls, model_path=str(MODEL_WSL_DOWNLOAD / MODEL_WSL_NAME), device="auto", dtype=torch.float16):
        if cls._instance is None:
            cls._instance = cls(model_path, device, dtype)
        return cls._instance

    def __init__(self, model_path=str(MODEL_WSL_DOWNLOAD / MODEL_WSL_NAME), device="auto", dtype=torch.float16):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.device = device

        print(f"[INFO] Loading tokenizer and model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            load_in_8bit=True,  # 8-bit量化
            device_map=device
        )

        # Structured prompt
        self.structured_prompt = Template("""
You are an assistant that extracts user preferences from text and outputs structured JSON.

- Each input may mention one or more points of interest (POIs).
- For each POI, determine whether the user has a positive or negative preference.
- Only include attributes that are explicitly mentioned or can be reasonably inferred from the text.
- Do NOT guess or invent POI names or attribute values if they are not clearly indicated.
- Use the following attributes:
  AcceptsInsurance, Alcohol, Ambience.casual, Ambience.classy, Ambience.divey, Ambience.hipster, Ambience.intimate, Ambience.romantic, Ambience.touristy, Ambience.trendy, Ambience.upscale, BikeParking, BusinessAcceptsBitcoin, BusinessAcceptsCreditCards, BusinessParking.garage, BusinessParking.lot, BusinessParking.street, BusinessParking.validated, BusinessParking.valet, ByAppointmentOnly, Caters, DogsAllowed, GoodForKids, GoodForMeal.breakfast, GoodForMeal.brunch, GoodForMeal.dessert, GoodForMeal.dinner, GoodForMeal.latenight, GoodForMeal.lunch, HasTV, HappyHour, NoiseLevel, OutdoorSeating, RestaurantsAttire, RestaurantsDelivery, RestaurantsGoodForGroups, RestaurantsPriceRange2, RestaurantsReservations, RestaurantsTableService, RestaurantsTakeOut, WheelchairAccessible, WiFi

- Output JSON format (vertical, no extra explanation):
{
  "positive": [
    {
      "poi_name": "<name of POI or empty string if unknown>",
      "paragraphs": ["<user text relevant to positive preference>"],
      "attributes": { "<attribute>": "true"/"false"/"quiet"/"average"/"loud" }
    }
  ],
  "negative": [
    {
      "poi_name": "<name of POI or empty string if unknown>",
      "paragraphs": ["<user text relevant to negative preference>"],
      "attributes": { "<attribute>": "true"/"false"/"quiet"/"average"/"loud" }
    }
  ]
}

- Notes:
  1. Do not add any extra details or invent reasons; keep paragraphs as close as possible to user input.
  2. If the user input is vague (e.g., just a category like "coffee"), leave `poi_name` as an empty string and only include attributes that are clearly mentioned or implied.
  3. Maintain proper JSON formatting with vertical (multi-line) structure for readability.
  4. A single input may result in multiple positive and/or negative entries.

User input: $input

Your output should strictly follow the JSON format above.
""")

        self._initialized = True

    # ===== 流式生成核心 =====
    def stream_generate(self, prompt, do_sample=True, max_new_tokens=1024, enable_thinking=True, callback=None):
        """
        流式生成主函数
        返回: thinking_content, content
        """

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        device = next(self.model.parameters()).device
        inputs = self.tokenizer([text], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "streamer": streamer
            }
        )
        thread.start()

        thinking_content = []
        content_tokens = []
        thinking_done = not enable_thinking

        for token_str in streamer:
            if not token_str.strip():
                continue

            token_ids = self.tokenizer(token_str, add_special_tokens=False)["input_ids"]
            if enable_thinking and not thinking_done and THINK_TOKEN_ID in token_ids:
                thinking_done = True
                thinking_content.append(token_str)
                print(token_str, end="", flush=True)
                token_str = token_str.replace("</think>", "")
                if token_str.strip():
                    content_tokens.append(token_str)
            else:
                if thinking_done:
                    content_tokens.append(token_str)
                else:
                    print(token_str, end="", flush=True)
                    thinking_content.append(token_str)

            if callback:
                callback(token_str)

        thread.join()

        return "".join(thinking_content).strip(), "".join(content_tokens).strip()

    # ===== 简单生成接口 =====
    def generate1(self, prompt, enable_thinking=True, max_new_tokens=1024, do_sample=False):
        _, content = self.stream_generate(prompt, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, do_sample=do_sample)
        return content

    # ===== 生成结构化 JSON =====
    def generate_structured(self, user_input):
        logging.info(f"用户输入：{user_input}")
        return self.generate1(
            self.structured_prompt.substitute(input=user_input),
            enable_thinking=True,
            max_new_tokens=150,
            do_sample=False
        )

    # ===== 普通文本生成 =====
    def generate_text(self, user_input):
        return self.generate1(user_input, enable_thinking=False, max_new_tokens=150, do_sample=False)


# ===== 使用示例 =====
if __name__ == "__main__":
    model = Qwen3Stream.get_instance()
    prompt = "你好"
    result = model.generate1(prompt)
    print("【回复】:\n", result)
