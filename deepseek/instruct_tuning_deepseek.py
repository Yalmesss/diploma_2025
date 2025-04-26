import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
import torch
from huggingface_hub import login


MODEL_NAME = "Yalmess/deepseek_dora_pretrained"
JSON_FILE = "combined_files.json"
OUTPUT_DIR = "/home/ubuntu/your_pretrained_models/deepseek_tuned"
HF_REPO = "Yalmess/deepseek_dora_finetuned"
HF_TOKEN = "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"


login(token=HF_TOKEN)


def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    formatted = []
    for entry in data:
        input_text = entry["input"]
        output_text = json.dumps(entry["output"], indent=2)
        prompt = (
            "Please analyze the following health message and return structured information "
            "as a JSON object with the following keys: symptoms, category, diagnosis, recommendations, "
            "suggested medications, generalization.\n\n"
            f"Message: {input_text}\n\nJSON Output:\n{output_text}"
        )
        formatted.append({"text": prompt})
    return formatted


def prepare_dataset(json_data):
    return Dataset.from_list(json_data)


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


def tokenize_dataset(dataset, tokenizer, max_length=1024):
    def tokenize_fn(examples):
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings
    return dataset.map(tokenize_fn, batched=True).remove_columns(["text"])


def configure_lora(model):
    if isinstance(model, PeftModel):
        print("Unloading existing PEFT model to avoid double wrapping.")
        model = model.base_model.model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def setup_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=HF_REPO,
        hub_strategy="checkpoint",
        report_to="none",
    )


def fine_tune_model():
    json_data = load_json_data(JSON_FILE)
    dataset = prepare_dataset(json_data)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    model = configure_lora(model)
    training_args = setup_training_args(OUTPUT_DIR)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    trainer.push_to_hub()
    print("✅ Fine-tuning complete and pushed to Hugging Face Hub.")


if __name__ == "__main__":
    fine_tune_model()
