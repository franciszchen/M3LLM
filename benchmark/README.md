# PMC-Bench Evaluation codebase

## Overview

This benchmark evaluates medical VLMs on six different task types:
- **Pure Text QA**: Text-only medical question answering
- **Multiple Choice VQA**: Multiple-choice medical questions with images
- **Single Sub-image VQA**: Questions about individual sub-images
- **Compound Image VQA**: Questions about compound medical images
- **Multi-subimage VQA**: Questions requiring multiple sub-images
- **Bounding Box VQA**: Spatial relationship questions about sub-images

## Installation
### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd benchmark
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

Create/modify the environment variables in `PMC_Data_Benchmark/eval.sh`:

```bash
export HF_HOME=/path/to/huggingface/cache       # Hugging Face cache directory
export HF_TOKEN=your_hf_token_here              # Hugging Face access token
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export PYTHONPATH=/path/to/benchmark:$PYTHONPATH
```

4. **Authenticate with Hugging Face:**
```bash
huggingface-cli login
```

## Supported Models

The benchmark currently supports the following medical VLMs:

| Model Name |
|-----------|
| **Qwen2.5-VL** | 
| **LLaVA** | 
| **LLaVA-Med** | 
| **HuatuoGPT** | 
| **InternVL** | 
| **HealthGPT** | 
| **MedGemma** | 

## Benchmark Tasks

The benchmark includes six task types, each with its own configuration in `PMC_Data_Benchmark/task_config.py`:

### 1. Pure Text QA (`puretext`)
- Text-only medical question answering
- No images required
- Dataset: `benchmark_data/puretextQA_benchmark.json`

### 2. Multiple Choice VQA (`multi-choice`)
- Multiple-choice questions with images
- Evaluated by accuracy
- Dataset: `benchmark_data/multiplechoiceVQA_benchmark.json`

### 3. Single Sub-image VQA (`single-subimageVQA`)
- Questions about individual sub-images from compound figures
- Dataset: `benchmark_data/single_subimageVQA_benchmark.json`

### 4. Compound Image VQA (`compoundVQA`)
- Questions about entire compound medical images
- Dataset: `benchmark_data/compound-imageVQA_benchmark.json`

### 5. Multi-subimage VQA (`multisubimageVQA`)
- Questions requiring analysis of multiple sub-images
- Dataset: `benchmark_data/multiplesubimageVQA_benchmark.json`

### 6. Bounding Box VQA (`bboxVQA`)
- Spatial relationship questions between sub-images
- Dataset: `benchmark_data/spatial_relation_benchmark.json`

## Quick Start

Use the provided shell script `eval.sh`:

```bash
cd PMC_Data_Benchmark
bash eval.sh
```

This will run all configured tasks sequentially for the selected model.

### Model Selection

Edit `PMC_Data_Benchmark/eval.sh` to change the model:

```bash
# Select your model
MODEL='medgemma'
MODEL_PATH='google/medgemma-27b-it'

# Or uncomment other models:
# MODEL='qwen2.5-vl'
# MODEL_PATH='Qwen/Qwen2.5-VL-7B-Instruct'

# MODEL='llava'
# MODEL_PATH='llava-hf/llava-1.5-7b-hf'
```

## Running Evaluations

### Standard Evaluation

```bash
cd PMC_Data_Benchmark

# Run evaluation for a specific task
python evaluate_all.py \
  --model_name <model_name> \
  --model_path <model_path> \
  --task <task_name>
```

**Arguments:**
- `--model_name`: Model identifier (see [Supported Models](#supported-models))
- `--model_path`: HuggingFace model path or local path
- `--task`: Task name (see [Benchmark Tasks](#benchmark-tasks))

### GPT-based Re-evaluation

To re-evaluate existing predictions using GPT as a judge:

```bash
python run_gpt_answer_evaluation.py \
  --model_name <model_name> \
  --task <task_name>
```

This will:
1. Load predictions from existing result files
2. Use GPT to re-score answers
3. Save updated metrics

## Results

### Output Directory Structure

All results are saved in the `PMC_Data_Benchmark/results/` directory:

```
PMC_Data_Benchmark/results/
├── <task_name>_<model_name>.json           # Detailed per-sample results
└── <task_name>_<model_name>_summary.json   # Overall performance summary
```

## Evaluation Metrics

The benchmark uses comprehensive metrics for evaluation:

### For Generation Tasks:

| Metric | Description |
|--------|-------------|
| **BLEU-1/2/3/4** | N-gram overlap scores (1-4 grams) |
| **ROUGE-1/2/L** | Recall-oriented scores for summary quality |
| **Exact Match (EM)** | Binary score for perfect matches |
| **F1 Score** | Token-level F1, Precision, and Recall |
| **BERTScore** | Semantic similarity using BERT embeddings |
| **BERT Similarity** | Cosine similarity in BERT embedding space |
| **CIDEr** | Consensus-based image description evaluation |

### For Multiple-Choice Tasks:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct option selections |



## License

[Specify your license here]

## Contact

For questions or issues, please contact [your contact information] or open an issue on GitHub.
