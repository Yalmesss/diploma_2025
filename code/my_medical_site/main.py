import json
import re
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
from huggingface_hub import login


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HF_TOKEN = "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"
login(token=HF_TOKEN)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def markdown_to_html(text: str) -> str:

    text = re.sub(r'^##\s*(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!<br>)\s+(-)', r'<br>\1', text)
    text = text.replace('\n\n', '<br><br>')

    return text



templates.env.filters['markdown_to_html'] = markdown_to_html


logger.info("Загрузка модели 1 (финетюн для JSON)")
model_path1 = "Yalmess/mistral_dora_finetuned"
tokenizer1 = AutoTokenizer.from_pretrained(model_path1, use_auth_token=HF_TOKEN)
model1 = AutoPeftModelForCausalLM.from_pretrained(
    model_path1,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=HF_TOKEN
)

logger.info("Загрузка модели 2 (для отчётов)")
model_path2 = "Yalmess/report_generator"
tokenizer2 = AutoTokenizer.from_pretrained(model_path2, use_auth_token=HF_TOKEN)
model2 = AutoPeftModelForCausalLM.from_pretrained(
    model_path2,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=HF_TOKEN
)

def generate_response(model, tokenizer, input_text, max_length=2048):
    prompt_template = (
        "Please analyze the following health message and return structured information "
        "as a JSON object with the following keys: symptoms, category, diagnosis, recommendations, "
        "suggested medications, generalization.\n\n"
        "Message: {}\n\nJSON Output:\n"
    )
    prompt = prompt_template.format(input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        pred_json = pred.split("JSON Output:\n")[1].strip()
    except IndexError:
        pred_json = pred
        logger.warning(f"Не удалось извлечь JSON из предсказания: {pred}")
    return pred_json

def generate_report(model, tokenizer, structured_json, max_length=2048):
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
        f"JSON Input:\n{structured_json}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    marker = "# Patient-Friendly Medical Report"
    idx = full_text.find(marker)
    if idx != -1:
        return full_text[idx + len(marker):].lstrip()
    else:
        return full_text

def capitalize_and_join(words):
    return ", ".join(w.title() for w in words)

def insert_line_break_before_dash(text: str) -> str:
    return re.sub(r'(?<!<br>)\s+(-)', r'<br>\1', text)

@app.get("/", response_class=HTMLResponse)
async def form_view(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def generate_report_view(request: Request, user_input: str = Form(...)):
    logger.info(f"Входное сообщение пользователя:\n{user_input}")

    raw_output1 = generate_response(model1, tokenizer1, user_input)
    logger.info(f"Вывод модели 1:\n{raw_output1}")

    try:
        match = re.search(r"\{.*\}", raw_output1, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("JSON объект не найден в выводе модели")
        
        structured_data = json.loads(json_str)


        for key in ["symptoms", "category"]:
            if key in structured_data and isinstance(structured_data[key], str):
                try:
                    structured_data[key] = json.loads(structured_data[key])
                except Exception:
                    pass


        symptoms_str = ""
        if "symptoms" in structured_data and isinstance(structured_data["symptoms"], list):
            symptoms_str = capitalize_and_join(structured_data["symptoms"])

        category_str = ""
        if "category" in structured_data and isinstance(structured_data["category"], list):
            medical_field = structured_data["category"][0] if len(structured_data["category"]) > 0 else ""
            urgency_level = structured_data["category"][2] if len(structured_data["category"]) > 2 else ""
            category_str = f"Medical field: {medical_field}"
            if urgency_level:
                category_str += f", Urgency level: {urgency_level}"

        medications_str = ""
        if "suggested medications" in structured_data and isinstance(structured_data["suggested medications"], dict):
            meds_list = []
            for med, analogs in structured_data["suggested medications"].items():
                analogs_str = ", ".join(analogs)
                meds_list.append(f"{med} (analogs: {analogs_str})")
            medications_str = "; ".join(meds_list)

        output_for_template = {
            "symptoms": symptoms_str,
            "category": category_str,
            "diagnosis": structured_data.get("diagnosis", ""),
            "recommendations": structured_data.get("recommendations", ""),
            "suggested_medications": medications_str,
            "generalization": structured_data.get("generalization", ""),
        }

    except Exception as e:
        logger.error(f"Ошибка парсинга JSON из модели 1: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Ошибка парсинга JSON: {str(e)}",
            "user_input": user_input,
            "json_output_parsed": {},
            "report_output": ""
        })

    json_input = json.dumps(structured_data, indent=2, ensure_ascii=False)
    raw_output2 = generate_report(model2, tokenizer2, json_input)
    logger.info(f"Отчёт от модели 2:\n{raw_output2}")


    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": user_input,
        "json_output_parsed": output_for_template,
        "report_output": raw_output2,
        "error": None
    })
