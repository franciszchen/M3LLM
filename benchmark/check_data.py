#!/usr/bin/env python3
import os
import json

JSON_PATH = "/Users/yihang/Desktop/PMC-codebase/benchmark/benchmark_data/final_instruction_MultipleSubimageVQA_nocontext_1021.json"
IMG_DIR = "/Users/yihang/Desktop/PMC-codebase/benchmark/benchmark_images"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

exclude_indices = []
for i, sample in enumerate(data):
    all_exist = True
    for img in sample.get("images", []):
        img_path = os.path.join(IMG_DIR, os.path.basename(img))
        if not os.path.exists(img_path):
            all_exist = False
            break
    if not all_exist:
        exclude_indices.append(i)

print(f"Total samples: {len(data)}")
print(f"Samples to exclude (missing images): {len(exclude_indices)}")
print(exclude_indices[:50])  # show first 50 indices
