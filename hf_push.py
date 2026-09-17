#CUDA_VISIBLE_DEVICES=0 nohup python hf_push.py > log.txt 2>&1 &
import os
import json
from datasets import Dataset, DatasetDict, Features, Image, Value, Sequence

LOCAL_IMAGES_DIR = "/DATA/ai20resch11003/all_images"
TRAIN_JSONL = "/DATA/ai20resch11003/Hal_Detect/data/train_new.jsonl"
TEST_JSONL = "/DATA/ai20resch11003/Hal_Detect/data/test_new.jsonl"
REPO_ID = "NLIP-lab/LID"

# explicit schema features to prevent ArrowTypeError
explicit_features = Features({
    "image_name": Value("string"),
    "image": Image(), # The image data column
    "description": Value("string"),
    "any_hal": Value("int64"),
    "obj_hal": Value("int64"),
    "missing_info": Value("int64"),
    "position": Value("int64"),
    "count": Value("int64"),
    "hal_level": Value("int64"),
    "items_hal": Sequence(Value("string")),
    "no_of_items_hal": Value("int64"),
    "no_of_missing_categories": Value("int64"),
    "objects_missed": Sequence(Value("string")),
    # Using string for the dictionary mapping to avoid nested structure validation errors
    "hal_word_pos_index": Value("string"), 
    "tot_word_count": Value("int64"),
    "remark": Value("string"), # Explicitly handled as string to cover null/None values safely
    "prompt": Value("string"),
    "model_name": Value("string")
})


def process_jsonl(jsonl_path, images_dir):
    data_rows = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                
                # Map local image path
                img_path = os.path.join(images_dir, item["image_name"])
                if os.path.exists(img_path):
                    item["image"] = img_path
                else:
                    print(f"Warning Line {line_num}: {item['image_name']} not found in {images_dir}")
                    item["image"] = None
                
                # Sanitize fields according to the explicit schema rules
                if "hal_word_pos_index" in item:
                    item["hal_word_pos_index"] = json.dumps(item["hal_word_pos_index"])
                else:
                    item["hal_word_pos_index"] = None
                    
                if "remark" not in item or item["remark"] is None:
                    item["remark"] = "" # Normalize null/None to empty string
                
                # Ensure array sequences are clean lists of strings
                if "items_hal" in item and item["items_hal"] is not None:
                    item["items_hal"] = [str(x) for x in item["items_hal"]]
                else:
                    item["items_hal"] = []
                    
                if "objects_missed" in item and item["objects_missed"] is not None:
                    item["objects_missed"] = [str(x) for x in item["objects_missed"]]
                else:
                    item["objects_missed"] = []
                
                data_rows.append(item)
                
            except Exception as e:
                print(f"Error parsing line {line_num} in {jsonl_path}: {e}")
                continue
            
    # Reorganize list of dicts to a dict of lists for Arrow processing
    columns = {key: [] for key in explicit_features.keys()}
    for row in data_rows:
        for key in explicit_features.keys():
            columns[key].append(row.get(key, None))
            
    # Inject the features directly during instantiation to bypass auto-inferencing
    return Dataset.from_dict(columns, features=explicit_features)

print("Processing Train Split...")
train_dataset = process_jsonl(TRAIN_JSONL, LOCAL_IMAGES_DIR)

print("Processing Test Split...")
test_dataset = process_jsonl(TEST_JSONL, LOCAL_IMAGES_DIR)

# Bundle them into a DatasetDict
final_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print("Pushing data to Hugging Face...")
# By default, push_to_hub places DatasetDict structures cleanly into a data/ directory
final_dataset.push_to_hub(
    repo_id=REPO_ID,
    commit_message="Converted dataset to parquet"
)

print("Done! Check your repository workspace.")
