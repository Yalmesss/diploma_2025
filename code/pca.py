# import pandas as pd
# import numpy as np
# import torch
# import re
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score
# from huggingface_hub import login

# # Authenticate with Hugging Face
# login(token="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")

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model and tokenizer
# #model1_path = "Yalmess/mistral_med_pretrained"
# model1_path = "mistralai/Mistral-7B-v0.3"  # Uncomment and set to your second model
# tokenizer = AutoTokenizer.from_pretrained(model1_path)
# model1 = AutoModelForCausalLM.from_pretrained(
#     model1_path, torch_dtype=torch.float16, device_map="auto"
# )
# # model2 = AutoModelForCausalLM.from_pretrained(
# #     model2_path, torch_dtype=torch.float16, device_map="auto"
# # )  # Uncomment for second model

# # Load and preprocess dataset
# df = pd.read_csv('pca_data.csv')
# df = df[['medical_specialty', 'keywords']]

# # Parse and clean keywords
# def parse_keywords(keyword_str):
#     if pd.isna(keyword_str) or not isinstance(keyword_str, str):
#         return []
#     # Split on commas and clean
#     keyword_list = [kw.strip() for kw in keyword_str.split(",") if kw.strip()]
#     # Remove disclaimer text and artifacts
#     cleaned_keywords = []
#     disclaimer_pattern = re.compile(r'These\s*transcribed\s*medical\s*transcription')
#     for kw in keyword_list:
#         if disclaimer_pattern.search(kw) or kw.endswith('NOTE'):
#             continue  # Skip disclaimers and NOTE artifacts
#         if len(kw) < 100:  # Arbitrary length limit to avoid long invalid terms
#             cleaned_keywords.append(kw)
#     return cleaned_keywords

# df["keywords"] = df["keywords"].apply(parse_keywords)
# # Keep rows with 2+ valid keywords
# df = df[df["keywords"].apply(len) >= 2]
# # Shuffle and take up to 200 rows
# df = df.sample(frac=1, random_state=42).head(200)

# # Debug: Inspect keywords
# print("Inspecting keywords after preprocessing:")
# for idx, row in df.iterrows():
#     print(f"Row {idx}, Specialty: {row['medical_specialty']}, Keywords: {row['keywords']}")

# def get_keyword_embedding(keyword, model, tokenizer, device, max_length=128):
#     try:
#         inputs = tokenizer(
#             keyword,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=max_length
#         ).to(device)
#         with torch.no_grad():
#             outputs = model(**inputs, output_hidden_states=True)
#         embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
#         return embedding
#     except Exception as e:
#         print(f"Error embedding keyword '{keyword}': {e}")
#         return None

# # Compute similarity for each specialty
# similarity_scores = {"model1": []}
# specialties = []
# for _, row in df.iterrows():
#     keywords = row["keywords"]
#     specialty = row["medical_specialty"]
#     try:
#         # Get embeddings, filter out None
#         embeddings = [
#             emb for emb in (get_keyword_embedding(kw, model1, tokenizer, device) for kw in keywords)
#             if emb is not None
#         ]
#         if len(embeddings) < 2:
#             print(f"Skipping specialty '{specialty}': too few valid embeddings")
#             similarity_scores["model1"].append(0.0)
#         else:
#             emb = np.vstack(embeddings)
#             sim_matrix = cosine_similarity(emb)
#             sim = np.triu(sim_matrix, k=1).mean()
#             similarity_scores["model1"].append(sim if not np.isnan(sim) else 0.0)
#         specialties.append(specialty)
#     except Exception as e:
#         print(f"Error processing specialty '{specialty}': {e}")
#         similarity_scores["model1"].append(0.0)
#         specialties.append(specialty)

# # Summarize similarity results
# print("\nAverage Keyword Similarity per Specialty:")
# results = pd.DataFrame({
#     "Specialty": specialties,
#     "Model1_Similarity": similarity_scores["model1"]
# })
# print(results)
# print("\nOverall Average (excluding zeros):")
# valid_scores = [s for s in similarity_scores["model1"] if s != 0.0]
# print(f"Model1: {np.mean(valid_scores):.3f}" if valid_scores else "No valid scores")

# # Keyword clustering
# print("\nComputing clustering metrics...")
# all_keywords = [kw for row in df["keywords"] for kw in row]
# specialty_labels = [row["medical_specialty"] for row in df.itertuples() for _ in row.keywords]
# try:
#     embeddings = [
#         emb for emb in (get_keyword_embedding(kw, model1, tokenizer, device) for kw in all_keywords)
#         if emb is not None
#     ]
#     if len(embeddings) >= len(df["medical_specialty"].unique()):
#         emb_matrix = np.vstack(embeddings)
#         kmeans = KMeans(n_clusters=len(df["medical_specialty"].unique()), random_state=42).fit(emb_matrix)
#         valid_labels = specialty_labels[:len(embeddings)]  # Match labels to valid embeddings
#         ari = adjusted_rand_score(valid_labels, kmeans.labels_)
#         print(f"Clustering ARI: {ari:.3f}")
#     else:
#         print("Not enough valid embeddings for clustering")
# except Exception as e:
#     print(f"Clustering error: {e}")

# # Optional: Text generation (uncomment to enable)
# """
# def generate_text(prompt, model, tokenizer, max_length=100):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_length=max_length, do_sample=True, top_p=0.9)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# print("\nGenerating sample texts...")
# for _, row in df.head(5).iterrows():
#     prompt = f"Describe a {row['medical_specialty']} case involving {', '.join(row['keywords'][:2])}."
#     print(f"\nSpecialty: {row['medical_specialty']}")
#     print(f"Generated: {generate_text(prompt, model1, tokenizer)}")
# """
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

from huggingface_hub import login

# ---------------- SETTINGS ----------------
MODEL_PATH = "Yalmess/mistral_med_pretrained"
DATA_PATH = "pca_data.csv"        # CSV with 'medical_specialty' and 'keywords'
TEXT_PATH = "med_data.txt"        # Plain medical text
OUTPUT_DIR = "./eval_outputs"
MAX_LENGTH = 128
NUM_CLUSTERS = 10

login(token="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")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD AND CLEAN DATA ----------------
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

# ---------------- LOAD MODELS ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
encoder = AutoModel.from_pretrained(MODEL_PATH).cuda().eval()
clm_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).cuda().eval()

# ---------------- EMBEDDING CLUSTERING ----------------
def extract_embeddings(keywords_list):
    all_embeddings = []
    with torch.no_grad():
        for kw_set in keywords_list:
            text = ", ".join(kw_set)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
            outputs = encoder(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            all_embeddings.append(emb)
    return all_embeddings

def cluster_embeddings(embeddings, labels, save_path):
    reduced = PCA(n_components=2).fit_transform(embeddings)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42).fit(reduced)
    clusters = kmeans.labels_
    score = silhouette_score(reduced, clusters)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="tab10", alpha=0.7)
    plt.title(f"Embedding Clusters (Silhouette Score: {score:.2f})")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Embedding Clustering] Saved to {save_path}, Silhouette Score = {score:.2f}")

# ---------------- CLM PERPLEXITY ----------------
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

# ---------------- RUN EVALUATION ----------------
if __name__ == "__main__":
    print("📌 Extracting embeddings...")
    embeddings = extract_embeddings(df["keywords"])

    print("📌 Clustering embeddings...")
    cluster_embeddings(embeddings, df["medical_specialty"], os.path.join(OUTPUT_DIR, "embedding_clusters.png"))

    print("📌 Computing CLM perplexity...")
    compute_clm_perplexity(TEXT_PATH)