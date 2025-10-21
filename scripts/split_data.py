import os
import shutil
import random

DATASET_DIR = "data/raw"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
SPLIT_RATIO = 0.8  # 80% train, 20% test

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for category in os.listdir(DATASET_DIR):
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(category_path):
        continue

    files = [f for f in os.listdir(category_path) if f.endswith((".mp4", ".mov"))]
    random.shuffle(files)
    
    split_index = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_index]
    test_files = files[split_index:]
    
    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)
    
    for f in train_files:
        shutil.copy(os.path.join(category_path, f),
                    os.path.join(TRAIN_DIR, category, f))
    for f in test_files:
        shutil.copy(os.path.join(category_path, f),
                    os.path.join(TEST_DIR, category, f))

print("✅ Dataset split complete.")
