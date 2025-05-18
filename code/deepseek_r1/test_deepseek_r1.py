 import json
 import torch
 from transformers import AutoTokenizer
 from peft import AutoPeftModelForCausalLM
 from huggingface_hub import login
 import logging

 logging.basicConfig(level=logging.INFO)
 logger = logging.getLogger(__name__)


 HF_TOKEN = "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"
 login(token=HF_TOKEN)
 model_path2 = "Yalmess/report_generator"
 max_length = 2048


 tokenizer2 = AutoTokenizer.from_pretrained(model_path2, use_auth_token=HF_TOKEN)
 model2 = AutoPeftModelForCausalLM.from_pretrained(
     model_path2,
     torch_dtype=torch.float16,
     device_map="auto",
     use_auth_token=HF_TOKEN
 )

 def generate_report(structured_json):
     prompt = (
         "Generate a patient-friendly medical report from the provided JSON data. "
         "Structure it with sections: Overview, Symptoms, Diagnosis, Treatment Plan, Medications, What to Do Now?. "
         "In the Overview section state the brief description of the diagnosis, give basic information. "
         "In the Symptoms section state the most common symptoms and their explanation or how they appear. "
         "In the Diagnosis section explain why exactly this diagnosis, what can be a trigger and why it occurs. "
         "In the Medications section state the names of the medications with analogues and explain what it does. "
         "Reflect the urgency level (‘Routine,’ ‘Moderate,’ or ‘Urgent’) in the tone and recommendations. "
         "For ‘Urgent,’ use ‘Go to a doctor or hospital immediately’; for ‘Routine’ or ‘Moderate,’ use ‘See a doctor soon.’ "
         "Include all JSON details (symptoms, diagnosis, generalization, recommendations, medications) clearly in markdown format.\n\n"
         f"JSON Input:\n{json.dumps(structured_json, ensure_ascii=False)}"
     )
     inputs = tokenizer2(prompt, return_tensors="pt").to(model2.device)
     outputs = model2.generate(
         **inputs,
         max_length=max_length,
         pad_token_id=tokenizer2.eos_token_id,
         eos_token_id=tokenizer2.eos_token_id,
         do_sample=False
     )
     full_text = tokenizer2.decode(outputs[0], skip_special_tokens=True)
     marker = "# Patient-Friendly Medical Report"
     idx = full_text.find(marker)
     if idx != -1:
         return full_text[idx + len(marker):].lstrip()
     else:
         return full_text

 def check_structure(text):
     required_sections = [
         "# Overview",
         "## Symptoms",
         "## Diagnosis",
         "## Treatment Plan",
         "## Medications",
         "## What to Do Now"
     ]
     return all(section in text for section in required_sections)

 def evaluate_model_on_json_file(json_file_path, max_samples=100):
     with open(json_file_path, "r", encoding="utf-8") as f:
         test_samples = json.load(f)

     test_samples = test_samples[:max_samples]

     total = len(test_samples)
     correct_count = 0

     for idx, sample in enumerate(test_samples):
         structured_json = sample["output"]
        
         output_text = generate_report(structured_json)

         if check_structure(output_text):
             correct_count += 1

     accuracy = correct_count / total * 100
     print(f"Правильная структура в {accuracy:.2f}%.")
     return accuracy

 if __name__ == "__main__":
     json_file_path = "/home/ubuntu/combined_files.json"
     evaluate_model_on_json_file(json_file_path, max_samples=100)


