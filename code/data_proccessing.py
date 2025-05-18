import json
import os
from datasets import Dataset


DATA_DIR = "/Users/darianaumenko/Desktop/diploma_2025"

with open("/Users/darianaumenko/Desktop/diploma_2025/combined_files.json", "r") as f:
    data = json.load(f)

# print(data[0])  # Verify the first entry
def format_prompt(entry):
    return f"""### Input:
{entry['input']}

### Output:
{json.dumps(entry['output'], indent=2)}
"""

# Example
formatted_data = [format_prompt(entry) for entry in data]
# print(formatted[0])


dataset = Dataset.from_list(formatted_data)

# Split into train and evaluation sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Save datasets locally
train_dataset.save_to_disk(os.path.join(DATA_DIR, "train_dataset"))
eval_dataset.save_to_disk(os.path.join(DATA_DIR, "eval_dataset"))
print("Train and eval datasets saved.")
