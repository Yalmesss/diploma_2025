import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import torch

from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


login(token="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")


DATA_PATH = "combined_files.json"
MODEL_NAME = "Yalmess/mistral_dora_pretrained"
OUTPUT_DIR = "./mistral_finetuned"
HF_REPO = "Yalmess/mistral_dora_finetuned"
JSON_FILE = "combined_files.json"



def load_json_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    

    formatted_data = []
    for entry in data:
        input_text = entry["input"]
        output_text = json.dumps(entry["output"], indent=2, ensure_ascii=False)

        prompt = (
            "Please analyze the following health message and return structured information "
            "as a JSON object with the following keys: symptoms, category, diagnosis, recommendations, "
            "suggested medications, generalization.\n\n"
            f"Message: {input_text}\n\n"
            f"JSON Output:\n{output_text}"
        )
        
        formatted_data.append({"text": prompt})
    
    return formatted_data


def prepare_dataset(json_data):
    dataset = Dataset.from_list(json_data)
    return dataset


def load_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")
    

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


def tokenize_dataset(dataset, tokenizer, max_length=1024):
    def tokenize_function(examples):

        encodings = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

        encodings["labels"] = encodings["input_ids"].copy()
        return encodings
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    return tokenized_dataset


def configure_lora(model):

    if isinstance(model, PeftModel):
        model = model.unload()
        print("Unloaded existing PEFT adapters.")
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def setup_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=HF_REPO,
        hub_strategy="checkpoint",
    )


def fine_tune_model():
    try:

        print("Loading JSON data...")
        json_data = load_json_data(JSON_FILE)
        dataset = prepare_dataset(json_data)
        

        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
        

        print("Tokenizing dataset...")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        

        print("Configuring LoRA...")
        model = configure_lora(model)
        

        print("Setting up training arguments...")
        training_args = setup_training_args(OUTPUT_DIR)
        

        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        

        print("Starting fine-tuning...")
        trainer.train()
        

        print("Saving model...")
        trainer.save_model(OUTPUT_DIR)

        print(f"Pushing model to Hugging Face: {HF_REPO}")
        trainer.push_to_hub()
        
        print("Fine-tuning completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    fine_tune_model()
