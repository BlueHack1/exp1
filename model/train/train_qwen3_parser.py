import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from exp.model.parser.qwen3_parser import Qwen3Parser
import json
from tqdm import tqdm

class POIDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "intents": torch.tensor(item["intents"], dtype=torch.long),
            "scores": torch.tensor(item["scores"], dtype=torch.float),
            "attributes": torch.tensor(item["attributes"], dtype=torch.float)
        }

def train_parser(model, dataloader, device="cuda", lr=1e-4, num_epochs=3):
    model.intent_seg.train()
    model.negation_head.train()
    model.attribute_adapter.train()

    optimizer = optim.Adam(
        list(model.intent_seg.parameters()) +
        list(model.negation_head.parameters()) +
        list(model.attribute_adapter.parameters()),
        lr=lr
    )

    intent_loss_fn = nn.CrossEntropyLoss()
    score_loss_fn = nn.MSELoss()
    attr_loss_fn = nn.BCELoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intents_true = batch["intents"].to(device)
            scores_true = batch["scores"].to(device)
            attributes_true = batch["attributes"].to(device)

            with torch.no_grad():
                outputs = model.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]

            # IntentSeg
            intent_preds, intent_logits = model.intent_seg(hidden_states)
            intent_logits = intent_logits[:, :intents_true.shape[1], :]
            intent_loss = intent_loss_fn(intent_logits.view(-1, intent_logits.size(-1)), intents_true.view(-1))

            # 聚合 token -> intent vector
            intent_vectors = []
            for i in range(model.num_intent_labels):
                mask = (intent_preds == i)
                if mask.sum() == 0:
                    continue
                vec = hidden_states[mask].mean(dim=0)
                intent_vectors.append(vec)
            if len(intent_vectors) == 0:
                continue
            intent_vectors = torch.stack(intent_vectors)

            # NegationHead
            score_preds = model.negation_head(intent_vectors)
            score_loss = score_loss_fn(score_preds, scores_true[:intent_vectors.size(0)])

            # AttributeAdapter
            attr_preds = model.attribute_adapter(intent_vectors)
            attr_loss = attr_loss_fn(attr_preds, attributes_true[:intent_vectors.size(0)])

            loss = intent_loss + score_loss + attr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    parser_model = Qwen3Parser().to("cuda")
    dataset = POIDataset("train.jsonl", parser_model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    train_parser(parser_model, dataloader, device="cuda", lr=1e-4, num_epochs=3)
