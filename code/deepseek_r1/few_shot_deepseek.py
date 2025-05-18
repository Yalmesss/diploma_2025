import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login, HfApi

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    hf_token = "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"
    print("Using provided token. For security, set HF_TOKEN environment variable instead.")
login(token=hf_token)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)


lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


dataset = load_dataset("json", data_files="medical_reports.jsonl")


def tokenize_function(examples):
    inputs = examples["instruction"]
    outputs = examples["output"]
    # Combine input and output with a separator
    combined = [f"{inp}\n\n### Output:\n{out}" for inp, out in zip(inputs, outputs)]
    tokenized = tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "output"])

#training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_medical_reports",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    remove_unused_columns=False,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()


local_save_path = "./fine_tuned_medical_reports"
model.save_pretrained(local_save_path)
tokenizer.save_pretrained(local_save_path)

repo_id = "Yalmess/report_generator"
model.push_to_hub(repo_id, use_auth_token=hf_token)
tokenizer.push_to_hub(repo_id, use_auth_token=hf_token)
print(f"Model and tokenizer pushed to {repo_id}")


def generate_report(json_input):
    prompt = f"Generate a patient-friendly medical report from the provided JSON data. Structure it with sections: Overview, Symptoms, Diagnosis, Treatment Plan, Medications, What to Do Now?. In the Overview section state the brief description of the diagnosis, give basic information. In the Symptoms section state the most comon symptoms and their explanation or how they appear. In the section Diagnosis explain why exactly this diagnosis, what ca be a trigger and why it occurs. In Medications sections state the names of the medications with analogues and explain what it does.Reflect the urgency level (‘Routine,’ ‘Moderate,’ or ‘Urgent’) in the tone and recommendations. For ‘Urgent,’ use ‘Go to a doctor or hospital immediately’; for ‘Routine’ or ‘Moderate,’ use ‘See a doctor soon.’ Include all JSON details (symptoms, diagnosis, generalization, recommendations, medications) clearly in markdown format.\n\nJSON Input:\n{json_input}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=2048, num_beams=4, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


test_json = {
      "symptoms": ["persistent cough", "weight loss", "night sweats"],
      "category": ["Pulmonology", "Diagnosis", "Urgent"],
      "diagnosis": "Suspected Lung Cancer",
      "recommendations": "Oncology referral, bronchoscopy with biopsy, smoking cessation",
      "suggested medications": {
        "No medications until confirmed": [],
        "Paracetamol": ["Panadol", "Tylenol"]
      },
      "generalization": "Cough and weight loss due to possible lung malignancy"
    }
print(generate_report(test_json))
