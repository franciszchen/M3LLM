# M3LLM: Medical Multimodal Large Language Model

This repository contains the codebase for finetuning InternVL3-8B on PMC (PubMed Central) medical imaging datasets to create M3LLM, a specialized medical vision-language model.

## 🔍 Overview

M3LLM is built on top of [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B) and finetuned on comprehensive medical imaging tasks including:
- Pure text QA
- Bounding box-based visual QA
- Single and multiple sub-image reasoning
- Compound image understanding
- Sub-image option selection

## 📊 Dataset

The training data are curated and synthesized from PMC medical literature, organized into 6 task categories:

| Task Category | Samples | Description |
|--------------|---------|-------------|
| `qa-puretext` | 40,382 | Text-only medical QA |
| `qa-bbox` | 40,293 | Bounding box visual VQA |
| `qa-single_subimage` | 40,287 | Single sub-image VQA |
| `qa-multiple_subimage` | 39,462 | Multiple sub-image VQA |
| `qa-subimage-option` | 40,295 | Sub-image option VQA |
| `qa-compound-image` | 37,029 | Compound figure VQA |

### Dataset Structure

```
<YOUR_DATA_PATH>/PMC_Insturction_Data/
├── images/                          # Images for text and compound tasks
├── full_data_images/                # Images for visual reasoning tasks
└── full_training_data_with_description/
    ├── stage5_puretext_new.jsonl
    ├── stage4_boundingboxVQA_v1.jsonl
    ├── stage5_single_subimage.jsonl
    ├── stage5_multi_subimage_new.jsonl
    ├── stage5_subimage_option.jsonl
    └── stage5_compound_image.jsonl
```

Dataset configuration is defined in:
```
InternVL/internvl_chat/shell/data/PMC-data.json
```


## 🚀 Training

Two training approaches are provided:

### 1. **LoRA Finetuning** (Recommended for limited resources)

**Script:** `InternVL/internvl_chat/shell/internvl3.0/2nd_finetune/M3LLM-fulldata-lora.sh`

**Configuration:**
- **LoRA rank:** 16
- **Frozen:** Vision backbone, LLM, MLP (only LoRA adapters trained)
- **Batch size:** 16 (1 per device × 1 GPU × 16 gradient accumulation)
- **Learning rate:** 5e-6
- **Max sequence length:** 8192
- **Max dynamic patches:** 6
- **Training epochs:** 1
- **DeepSpeed:** Optional

**Usage:**
```bash
cd InternVL
export GPUS=1  # Adjust based on available GPUs
bash internvl_chat/shell/internvl3.0/2nd_finetune/M3LLM-fulldata-lora.sh
```

### 2. **Full Model Finetuning** (Better performance, more resources)

**Script:** `InternVL/internvl_chat/shell/internvl3.0/2nd_finetune/M3LLM-fulldata-fullmodel.sh`

**Configuration:**
- **Trainable:** LLM, MLP
- **Frozen:** Vision backbone only
- **Batch size:** 2 (1 per device × 2 GPUs × 1 gradient accumulation)
- **Learning rate:** 5e-6
- **Max sequence length:** 16384
- **Max dynamic patches:** 12
- **Training epochs:** 1
- **DeepSpeed:** Optional

**Usage:**
```bash
cd InternVL
export GPUS=2  # Adjust based on available GPUs
bash internvl_chat/shell/internvl3.0/2nd_finetune/M3LLM-fulldata-fullmodel.sh
```

## ⚙️ Configuration

Before running, update the following in the training scripts:

1. **Dataset paths in `PMC-data.json`:**
   - Update `root` paths to point to your image directories
   - Update `annotation` paths to point to your JSONL files

2. **Weights & Biases API key:**
   ```bash
   export WANDB_API_KEY='your_wandb_api_key'
   ```

3. **Output directory:**
   ```bash
   OUTPUT_DIR='<YOUR_OUTPUT_PATH>/M3LLM-fulldatav2-lora-lr=5e-6'
   ```

4. **Optional - HOME directory (if needed):**
   ```bash
   export HOME=<YOUR_HOME_DIR>
   ```

## 🔧 Key Training Parameters

| Parameter | LoRA | Full Model | Description |
|-----------|------|------------|-------------|
| `model_name_or_path` | OpenGVLab/InternVL3-8B | OpenGVLab/InternVL3-8B | Base model |
| `conv_style` | internvl2_5 | internvl2_5 | Conversation template |
| `force_image_size` | 448 | 448 | Input image resolution |
| `max_dynamic_patch` | 6 | 12 | Maximum dynamic patches |
| `freeze_backbone` | True | True | Freeze vision encoder |
| `freeze_llm` | True | False | Freeze language model |
| `freeze_mlp` | True | False | Freeze projection layer |
| `use_llm_lora` | 16 | - | LoRA rank (LoRA only) |
| `learning_rate` | 5e-6 | 5e-6 | Learning rate |
| `max_seq_length` | 8192 | 16384 | Maximum sequence length |
| `num_train_epochs` | 1 | 1 | Training epochs |
| `save_steps` | 200 | 200 | Checkpoint frequency |

## 📁 Directory Structure

```
M3LLM/
├── InternVL/
│   └── internvl_chat/
│       ├── shell/
│       │   ├── data/
│       │   │   └── PMC-data.json              # Dataset configuration
│       │   └── internvl3.0/
│       │       └── 2nd_finetune/
│       │           ├── M3LLM-fulldata-lora.sh      # LoRA training script
│       │           └── M3LLM-fulldata-fullmodel.sh # Full model training script
│       └── train/
│           └── internvl_chat_finetune.py      # Training script (InternVL)
└── full_training_data/                         # Training data (JSONL files)
```

