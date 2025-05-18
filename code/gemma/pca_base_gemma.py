import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "google/gemma-7b" #or "Yalmess/gemma_dora_pretrained"
TEXT_PATH = "med_data.txt"
MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
model.eval()


def compute_clm_perplexity(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

    losses = []
    with torch.no_grad():
        for line in lines:
            inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
            outputs = model(**inputs, labels=inputs["input_ids"])

            loss = outputs.loss.item()
            if math.isnan(loss):
                continue

            ppl = math.exp(loss)
            losses.append(ppl)

    if len(losses) == 0:
        return float('nan')

    avg_ppl = sum(losses) / len(losses)
    print(f"Avg Perplexity = {avg_ppl:.2f}")
    return avg_ppl


if __name__ == "__main__":
    compute_clm_perplexity(TEXT_PATH)
