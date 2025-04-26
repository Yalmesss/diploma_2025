import json
import logging
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
from bert_score import score
from huggingface_hub import login


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


login(token="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")


def symptoms_accuracy(pred_json, gt_json):
    try:
        pred_symptoms = set(pred_json.get("symptoms", []))
        gt_symptoms = set(gt_json.get("symptoms", []))
        if not gt_symptoms:
            return 1.0 if not pred_symptoms else 0.0
        intersection = len(pred_symptoms & gt_symptoms)
        union = len(pred_symptoms | gt_symptoms)
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

def category_accuracy(pred_json, gt_json):
    try:
        pred_categories = set(pred_json.get("category", []))
        gt_categories = set(gt_json.get("category", []))
        return 1.0 if pred_categories == gt_categories else 0.0
    except:
        return 0.0

def diagnosis_accuracy(pred_json, gt_json):
    try:
        pred_diagnosis = pred_json.get("diagnosis", "").lower().strip()
        gt_diagnosis = gt_json.get("diagnosis", "").lower().strip()
        return 1.0 if pred_diagnosis == gt_diagnosis else 0.0
    except:
        return 0.0

def recommendations_similarity(pred_json, gt_json):
    try:
        pred_recs = pred_json.get("recommendations", "").strip()
        gt_recs = gt_json.get("recommendations", "").strip()
        if not gt_recs:
            return 1.0 if not pred_recs else 0.0
        P, R, F1 = score([pred_recs], [gt_recs], lang="en", rescale_with_baseline=True)
        return F1.item()
    except:
        return 0.0

def generalization_similarity(pred_json, gt_json):
    try:
        pred_text = pred_json.get("generalization", "").strip()
        gt_text = gt_json.get("generalization", "").strip()
        if not gt_text:
            return 1.0 if not pred_text else 0.0
        P, R, F1 = score([pred_text], [gt_text], lang="en", rescale_with_baseline=True)
        return F1.item()
    except:
        return 0.0

def json_structure_accuracy(prediction):
    required_keys = {"symptoms", "category", "diagnosis", "recommendations", "suggested medications", "generalization"}
    try:
        pred_json = json.loads(prediction)
        has_required_keys = set(pred_json.keys()) == required_keys
        if has_required_keys and isinstance(pred_json.get("suggested medications"), dict):
            return 1.0
        return 0.0
    except json.JSONDecodeError:
        return 0.0


def evaluate_model(model_path, test_files, max_length=512):
    logger.info("Loading model and tokenizer from %s", model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    

    prompt_template = (
        "Please analyze the following health message and return structured information "
        "as a JSON object with the following keys: symptoms, category, diagnosis, recommendations, "
        "suggested medications, generalization.\n\n"
        "Message: {}\n\nJSON Output:\n{}"
    )
    
    results_per_file = {}
    all_predictions = {}
    all_ground_truths = {}
    
    for test_file in test_files:
        logger.info(f"Loading dataset from {test_file}")
        dataset = load_dataset("json", data_files=test_file)["train"]
        
        logger.info(f"Generating predictions for {test_file}")
        predictions = []
        ground_truths = []
        for sample in dataset:
            input_text = sample["input"]
            # Use the fine-tuning prompt, leaving output_text empty
            prompt = prompt_template.format(input_text, "")
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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
                pred_json = "{}"
                logger.warning(f"Failed to extract JSON from prediction: {pred}")
            predictions.append(pred_json)
            ground_truths.append(json.dumps(sample["output"], ensure_ascii=False))
        
        logger.info(f"Computing metrics for {test_file}")
        metrics = {
            "symptoms_accuracy": [],
            "category_accuracy": [],
            "diagnosis_accuracy": [],
            "recommendations_similarity": [],
            "generalization_similarity": [],
            "json_structure_accuracy": [],
        }
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            pred_json = {}
            try:
                pred_json = json.loads(pred)
            except json.JSONDecodeError:
                logger.warning(f"Sample {i} in {test_file}: Invalid JSON: {pred}")
            gt_json = json.loads(gt)
            
            metrics["symptoms_accuracy"].append(symptoms_accuracy(pred_json, gt_json))
            metrics["category_accuracy"].append(category_accuracy(pred_json, gt_json))
            metrics["diagnosis_accuracy"].append(diagnosis_accuracy(pred_json, gt_json))
            metrics["recommendations_similarity"].append(recommendations_similarity(pred_json, gt_json))
            metrics["generalization_similarity"].append(generalization_similarity(pred_json, gt_json))
            metrics["json_structure_accuracy"].append(json_structure_accuracy(pred))
            
            # Log incorrect predictions
            if (json_structure_accuracy(pred) < 1.0 or 
                diagnosis_accuracy(pred_json, gt_json) < 1.0 or 
                symptoms_accuracy(pred_json, gt_json) < 1.0):
                logger.warning(f"Sample {i} in {test_file}: Input: {dataset[i]['input']}, Pred: {pred}, GT: {gt}")
        

        results = {}
        for metric, scores in metrics.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            results[metric] = avg_score
            logger.info(f"{test_file} - Average {metric}: {avg_score:.4f}")
        
        results_per_file[test_file] = results
        all_predictions[test_file] = predictions
        all_ground_truths[test_file] = ground_truths
    

    output_file = "evaluation_results.json"
    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump({
            "results": results_per_file,
            "predictions": all_predictions,
            "ground_truths": all_ground_truths
        }, f, ensure_ascii=False, indent=2)
    
    return results_per_file, all_predictions, all_ground_truths


if __name__ == "__main__":
    model_path = "Yalmess/mistral_dora_finetuned"
    test_files = ["test1.json", "test2.json"]
    results, predictions, ground_truths = evaluate_model(model_path, test_files)
    
    print("\nEvaluation Results:")
    for test_file, metrics in results.items():
        print(f"\n{test_file}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")
