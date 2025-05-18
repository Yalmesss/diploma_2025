import pandas as pd
import torch
import math
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import re
from peft import PeftModel
from huggingface_hub import login
from transformers import AutoModelForCausalLM


BASE_MODEL = "mistralai/Mistral-7B-v0.3" # mistralai/Mistral-7B-v0.3   
ADAPTER_PATH = "Yalmess/mistral_dora_pretrained"
DATA_PATH = "pca_data.csv"
TEXT_PATH = "med_data.txt"
OUTPUT_DIR = "./eval_outputs"
MAX_LENGTH = 128

login(token="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")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df[['medical_specialty', 'keywords']]

def parse_keywords(keyword_str: str) -> list[str]:
    if pd.isna(keyword_str) or not isinstance(keyword_str, str):
        return []
    keyword_list = [kw.strip().lower() for kw in keyword_str.split(",") if kw.strip()]
    stop_words = {
        'procedure', 'incision', 'closure', 'note', 'surgery', 'patient',
        'transcription', 'report', 'sample', 'medical', 'biopsy', 'discharge',
        'sutures', 'tissue', 'lesion'
    }
    disclaimer_pattern = re.compile(r'these\s*transcribed\s*medical\s*transcription', re.I)
    cleaned_keywords = []
    for kw in keyword_list:
        if disclaimer_pattern.search(kw) or kw.endswith('note') or len(kw) > 50:
            continue
        if kw in stop_words:
            continue
        cleaned_keywords.append(kw)
    seen = set()
    return [kw for kw in cleaned_keywords if not (kw in seen or seen.add(kw))]

df["keywords"] = df["keywords"].apply(parse_keywords)
df = df[df["keywords"].apply(len) >= 3]
df = df.sample(frac=1, random_state=42).head(200)
NUM_CLUSTERS = df['medical_specialty'].nunique()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


base_encoder = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
encoder = PeftModel.from_pretrained(base_encoder, ADAPTER_PATH).eval()

base_clm = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
clm_model = PeftModel.from_pretrained(base_clm, ADAPTER_PATH).eval()


def compute_clm_perplexity(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

    losses = []
    with torch.no_grad():
        for line in lines:
            inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
            outputs = clm_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            ppl = math.exp(loss)
            losses.append(ppl)

    avg_ppl = sum(losses) / len(losses)
    print(f"[CLM Perplexity] Avg Perplexity = {avg_ppl:.2f}")
    return avg_ppl

if __name__ == "__main__":

    print("Computing CLM perplexity...")
    compute_clm_perplexity(TEXT_PATH)
