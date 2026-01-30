from modelscope import AutoModelForCausalLM, AutoTokenizer
from model_test.path_config import WSL_DOWNLOAD
import torch
import time

model_name = str(WSL_DOWNLOAD / "Qwen" / "Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="cuda",
)

THINK_TOKEN_ID = 151668  # </think> token

def stream_generate(model, tokenizer, input_ids, max_new_tokens=1024, sleep_time=0.01):
    """
    边生成边输出 token
    返回 thinking_content, content
    """
    generated_ids = []
    thinking_done = False
    thinking_content = ""
    content = ""

    while True:
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        next_token_id = outputs.sequences[0, -1].item()
        if next_token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id)
        token_str = tokenizer.decode([next_token_id], skip_special_tokens=True)

        if not thinking_done:
            if next_token_id == THINK_TOKEN_ID:
                thinking_done = True
                print("\n[模型思考结束]\n回答:", end="", flush=True)
                continue
            print(token_str, end="", flush=True)
            thinking_content += token_str
        else:
            print(token_str, end="", flush=True)
            content += token_str

        # 更新 input_ids 以进行下一步生成
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(model.device)], dim=-1)
        time.sleep(sleep_time)

    print("\n")
    return thinking_content.strip(), content.strip()


while True:
    try:
        prompt = input("\n你: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            print("结束对话。")
            break

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print("\n[模型思考中]", end="", flush=True)
        thinking_content, content = stream_generate(model, tokenizer, model_inputs["input_ids"], max_new_tokens=1024)

        if thinking_content:
            print("[模型思考内容]:", thinking_content)
        print("[模型回答]:", content)

    except KeyboardInterrupt:
        print("\n结束对话。")
        break
