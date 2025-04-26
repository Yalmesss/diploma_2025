import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
from huggingface_hub import login
from peft import get_peft_model, LoraConfig, TaskType
import os
import logging

login(token="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")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = "mistralai/Mistral-7B-v0.3"  #"mistralai/Mistral-7B-v0.1"  #google/gemma-7b
DATA_FILES = ["med_data.txt",
              "transcriptions.txt"]
OUTPUT_DIR = "/home/ubuntu/your_pretrained_models/mistral_dora"
HF_REPO = "Yalmess/mistral_dora_pretrained"
HF_TOKEN = os.getenv("HF_TOKEN", "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")

MAX_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACC = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3


logger.info("Logging in to Hugging Face Hub...")
login(token=HF_TOKEN)


logger.info("Loading model and tokenizer...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


logger.info("Adding DoRA adapters...")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=True
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


logger.info("Loading and chunking dataset...")
def load_all_texts(file_paths):
    all_texts = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            all_texts.extend(lines)
    return all_texts

def chunk_text(text, max_tokens=MAX_LENGTH):
    tokens = tokenizer.encode(text, truncation=False)
    return [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

texts_raw = load_all_texts(DATA_FILES)
texts_chunked = []
for text in texts_raw:
    texts_chunked.extend(chunk_text(text))

dataset = Dataset.from_dict({"text": texts_chunked})


def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

logger.info("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


train_size = int(0.9 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_steps=2000,
    eval_steps=2000,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
    push_to_hub=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)


logger.info("Starting training...")
trainer.train()


logger.info("Saving model and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("Pushing to HF Hub...")
model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)

logger.info("✅ Training complete.")
