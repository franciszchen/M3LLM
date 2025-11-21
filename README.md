# M3LLM: Medical Multimodal Large Language Model

**M3LLM** is a specialized medical vision-language model built on [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B), finetuned on comprehensive PMC medical imaging datasets to achieve state-of-the-art performance on multi-image medical visual question answering tasks.
![Overall performance](assets/radar-v4.pdf)
---

## 📊 Evaluation Results

### PMC-MI-Benchmark

Our model demonstrates strong performance on the PMC-MI-Benchmark, a comprehensive evaluation suite for multi-image medical visual question answering:

<!-- Insert PMC-MI-Benchmark results image here -->
![PMC-MI-Benchmark Results](assets/PMC-MI-multi-image.png)
![PMC-MI-Benchmark Results](assets/PMC-MI-textonly-multichoice.png)
**Key Results:**
- [Add specific metrics and comparisons here]

### Public Benchmarks

M3LLM achieves competitive performance across various public medical VQA benchmarks:

<!-- Insert public benchmark results image here -->
![Public Benchmark Results](assets/Public-benchmark.png)

**Evaluated Benchmarks:**
- [List benchmarks and key results]

### MIMIC Evaluation

Performance evaluation on MIMIC-CXR and related clinical datasets:

<!-- Insert MIMIC results image here -->
![MIMIC Results](assets/MIMIC.png)

**Clinical Capabilities:**
- [Highlight clinical reasoning capabilities]

---

## 🔄 Training Data Synthesis Pipeline

M3LLM's training data is generated through a comprehensive 5-stage synthetic data pipeline that processes medical images and captions from PubMed Central (PMC) literature.

### Pipeline Overview

```
Raw PMC Data → Stage 1-3: Preprocessing → Stage 4: Task-Specific QA → Stage 5: Context Refinement → Training Data
```

### Data Statistics

| Task Category | Samples | Description |
|--------------|---------|-------------|
| **Pure Text QA** | 40,382 | Text-only medical QA |
| **Bounding Box VQA** | 40,293 | Spatial relationship questions |
| **Single Sub-image** | 40,287 | Single sub-image reasoning |
| **Multiple Sub-images** | 39,462 | Multi-image reasoning |
| **Sub-image Options** | 40,295 | Multiple-choice questions |
| **Compound Images** | 37,029 | Compound figure understanding |
| **Total** | **~238K** | Six diverse task types |

### Pipeline Stages

**Stage 1-3: Data Preprocessing**
- `1_inline_text_summarization.py`: Summarizes medical observations from inline text
- `2_medical_knowledge_complementation.py`: Extracts keywords and generates medical knowledge
- `3_visual_perception_enhancement.py`: Creates visual perception descriptions using multimodal models

**Stage 4: Task-Specific QA Generation**
- Six specialized scripts for different medical VQA task types
- Generates questions, contexts, and answers based on medical images and captions
- Supports pure text, spatial reasoning, single/multi-image, and multiple-choice tasks

**Stage 5: Context Refinement**
- Improves question contexts to prevent data leakage
- Removes answer-revealing information while maintaining clinical reasoning requirements
- Ensures high-quality instruction-following data

📁 **Detailed pipeline documentation**: See [`Instruction_data_generation/`](Instruction_data_generation/) for implementation details and usage instructions.

---

## 🚀 Training

M3LLM provides two training approaches to accommodate different computational resources:

### Quick Start

```bash
# Navigate to training directory
cd InternVL

# LoRA training (recommended for limited resources)
bash internvl_chat/shell/internvl3.0/2nd_finetune/M3LLM-fulldata-lora.sh

# Full model training (better performance)
bash internvl_chat/shell/internvl3.0/2nd_finetune/M3LLM-fulldata-fullmodel.sh
```
📁 **Detailed training documentation**: See [`InternVL/`](InternVL/) for configuration details, hyperparameters, and troubleshooting.

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/franciszchen/M3LLM.git
cd M3LLM
```

### Step 2: Set Up Environment

```bash
# Create conda environment
conda create -n m3llm python=3.10
conda activate m3llm

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers accelerate deepspeed
pip install wandb pandas tqdm pillow
pip install flash-attn --no-build-isolation
```

### Step 3: Prepare Data

If you want to use our synthetic data generation pipeline:

```bash
cd Instruction_data_generation
# Configure your paths in the scripts
# Run the pipeline stages sequentially (1 → 2 → 3 → 4 → 5)
```

Or download our preprocessed training data:
```bash
# Instructions for downloading preprocessed data
# [Add download links or instructions]
```

### Step 4: Configure Training

Update the configuration files with your paths:

```bash
cd InternVL/internvl_chat/shell/data
# Edit PMC-data.json with your data paths

cd ../internvl3.0/2nd_finetune
# Edit training scripts:
# - Set WANDB_API_KEY
# - Set OUTPUT_DIR
# - Adjust GPU settings
```

### Step 5: Start Training

```bash
cd InternVL
# For LoRA training
bash internvl_chat/shell/internvl3.0/2nd_finetune/M3LLM-fulldata-lora.sh

# For full model training
bash internvl_chat/shell/internvl3.0/2nd_finetune/M3LLM-fulldata-fullmodel.sh
```

---

## 📂 Repository Structure

```
M3LLM/
├── Instruction_data_generation/     # Synthetic data generation pipeline
│   ├── 1_inline_text_summarization.py
│   ├── 2_medical_knowledge_complementation.py
│   ├── 3_visual_perception_enhancement.py
│   ├── 4_*.py                       # Stage 4: Task-specific QA generation
│   └── 5_*.py                       # Stage 5: Context refinement
│
├── InternVL/                        # Training codebase
│   ├── internvl_chat/
│   │   ├── shell/
│   │   │   ├── data/
│   │   │   │   └── PMC-data.json   # Dataset configuration
│   │   │   └── internvl3.0/
│   │   │       └── 2nd_finetune/   # Training scripts
│   │   ├── internvl/
│   │   │   ├── model/              # Model implementations
│   │   │   └── train/              # Training utilities
│   │   └── eval/                   # Evaluation scripts
│   └── README.md                   # Detailed training documentation
│
├── benchmark/                       # Evaluation benchmarks
│   └── MIMIC-unified-inference/    # MIMIC evaluation framework
│
└── README.md                        # This file
```

---

## 🎯 Model Checkpoints

We release the following model checkpoints:

| Model | Training Method | Download Link | Size |
|-------|----------------|---------------|------|
| M3LLM-LoRA | LoRA (rank 16) | [Coming Soon] | ~XX MB |
| M3LLM-Full | Full Finetuning | [Coming Soon] | ~XX GB |

### Loading the Model

```python
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("path/to/m3llm-checkpoint")
tokenizer = AutoTokenizer.from_pretrained("path/to/m3llm-checkpoint")

# Example inference
# [Add inference example]
```

---

## 📖 Citation

If you find M3LLM useful for your research, please cite:

```bibtex
@article{m3llm2024,
  title={M3LLM: Medical Multimodal Large Language Model},
  author={[Your Name and Collaborators]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

Please also cite the base model:

```bibtex
@article{internvl3,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and others},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

---

## 📄 License

This project is released under the [MIT License](LICENSE).

The InternVL codebase is subject to its original license terms. Please refer to the [InternVL repository](https://github.com/OpenGVLab/InternVL) for details.

---

## 🙏 Acknowledgments

- [InternVL](https://github.com/OpenGVLab/InternVL) for the foundational vision-language model
- [HuggingFace](https://huggingface.co/) for the model hosting and inference infrastructure
- [PMC Open Access Subset](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) for medical image data
- All contributors and collaborators who made this project possible

---

## 📧 Contact

For questions or collaborations, please contact:
- [Your Name/Team]
- Email: [your.email@institution.edu]
- GitHub Issues: [https://github.com/franciszchen/M3LLM/issues](https://github.com/franciszchen/M3LLM/issues)

---

## 🔗 Related Resources

- 🏥 [PMC-MI-Benchmark](link-to-benchmark)
- 📚 [Paper](link-to-paper)
- 🤗 [Model on HuggingFace](link-to-hf-model)
- 📊 [Training Logs](link-to-wandb)

---

<div align="center">
  
**Built with ❤️ for advancing medical AI**

⭐ If you find this project helpful, please consider giving it a star!

</div>

