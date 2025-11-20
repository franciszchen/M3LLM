# Unified MIMIC-CXR Inference Framework

A unified, organized framework for running inference on MIMIC-CXR chest X-ray tasks using multiple vision-language models.

## Overview

This framework supports **two medical imaging tasks**:
1. **Binary Disease Classification**: Determine if a disease is present (Yes/No)
2. **Disease Progression**: Assess disease progression status (worsening/stable/improving)

## Supported Models

- **LLaVA** (LLaVA-NeXT)
- **Qwen2.5-VL**
- **Lingshu**
- **InternVL** 
- **HuatuoGPT-Vision**
- **MedGemma**

## Directory Structure

```
MIMIC-unified-inference/
├── data/
│   ├── disease_binary.jsonl          # Binary classification task data
│   └── disease_progression.jsonl     # Disease progression task data
├── scripts/
│   ├── llava_inference.py            # LLaVA-v1.6 inference script
│   ├── qwen_inference.py             # Qwen2.5-VL inference script
│   ├── internvl_inference.py         # InternVL inference script
│   ├── huatuo_inference.py           # HuatuoGPT-Vision inference script
│   └── medgemma_inference.py         # MedGemma inference script
├── outputs/                          # Output directory (created automatically)
└── README.md                         # This file
```

# Quick Start Guide

## 🚀 Getting Started 

### Step 0 prepare dataset
get the access and download the MIMIC dataset from [link](https://physionet.org/content/mimic-cxr/2.1.0/)

### Step 1: Set Environment Variables

```bash
# Set image root directory
export IMAGE_ROOT=/path/to/MIMIC-CXR-JPG-sorted
# Optional: Set HuggingFace cache directory
# export HF_HOME=/path/to/hf_cache
```

### Step 2: Test with a Small Sample

```bash
cd /path/to/MIMIC-unified-inference

# Test LLaVA on 10 samples
python scripts/llava_inference.py \
  --model_path llava-hf/llava-v1.6-mistral-7b-hf \
  --input_jsonl data/disease_binary.jsonl \
  --image_root $IMAGE_ROOT \
  --output_dir outputs/test_llava \
  --limit 10
```

### Step 3: Run Full Inference

#### Binary Classification Task

```bash
# InternVL on binary classification
python scripts/internvl_inference.py \
  --model_path OpenGVLab/InternVL3-8B \
  --input_jsonl data/disease_binary.jsonl \
  --image_root $IMAGE_ROOT \
  --output_dir outputs/internvl_binary
```

#### Disease Progression Task

```bash
# Qwen2.5-VL on progression
python scripts/qwen_inference.py \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --input_jsonl data/disease_progression.jsonl \
  --image_root $IMAGE_ROOT \
  --output_dir outputs/qwen_progression
```

## 📊 Available Models

| Model | Script |
|-------|--------|
| LLaVA-v1.6 | `llava_inference.py`
| Qwen2.5-VL | `qwen_inference.py`
| Lingshu-7b | `qwen_inference.py`
| InternVL | `internvl_inference.py`
| HuatuoGPT-Vision | `huatuo_inference.py`
| MedGemma | `medgemma_inference.py`

## 📁 Data Files

- **Binary Classification** : `data/disease_binary.jsonl`
  - Task: Determine if disease is present (Yes/No)
  - Diseases: consolidation, edema, pleural effusion, pneumonia, pneumothorax

- **Disease Progression** : `data/disease_progression.jsonl`
  - Task: Assess progression (worsening/stable/improving)
  - Requires 2 images: previous and current examination

## 🎯 Quick Commands

### Run All Models on Binary Task

```bash
# Create a simple batch script
for model in llava qwen internvl medgemma; do
  echo "Running $model..."
  python scripts/${model}_inference.py \
    --model_path <model_path_here> \
    --input_jsonl data/disease_binary.jsonl \
    --image_root $IMAGE_ROOT \
    --output_dir outputs/${model}_binary
done
```

### SLURM Submission

```

# Submit
sbatch submit_job.sh

```

