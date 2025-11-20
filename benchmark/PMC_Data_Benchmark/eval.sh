#!/bin/bash
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=xxxxx:$PYTHONPATH
export OPENAI_API_KEY=your_hf_token_here
export HF_HOME= HF_HOME
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
#hf auth login 

huggingface-cli whoami || true

MODEL='medgemma'
MODEL_PATH='google/medgemma-27b-it'
#MODEL='qwen2.5-vl'
#MODEL_PATH='lingshu-medical-mllm/Lingshu-7B'
#MODEL_PATH='Qwen/Qwen2.5-VL-7B-Instruct'
#MODEL='llava'
#MODEL_PATH='llava-hf/llava-1.5-7b-hf'
#MODEL_PATH='llava-hf/llava-v1.6-mistral-7b-hf'
#MODEL='llava-med'
#MODEL_PATH='microsoft/llava-med-v1.5-mistral-7b'
#MODEL='huatuo'
#MODEL_PATH='FreedomIntelligence/HuatuoGPT-Vision-34B'
#MODEL='internvl'
#MODEL_PATH='OpenGVLab/InternVL3-8B'
#MODEL='healthgpt'
#MODEL_PATH='microsoft/phi-4'

python /gpfs/radev/pi/q_chen/yf329/PMC_instruction_data/benchmark/PMC_Data_Benchmark/evaluate_all.py \
  --model_name $MODEL \
  --model_path $MODEL_PATH \
  --task multisubimageVQA

python /gpfs/radev/pi/q_chen/yf329/PMC_instruction_data/benchmark/PMC_Data_Benchmark/evaluate_all.py \
  --model_name $MODEL \
  --model_path $MODEL_PATH \
  --task compoundVQA

python /gpfs/radev/pi/q_chen/yf329/PMC_instruction_data/benchmark/PMC_Data_Benchmark/evaluate_all.py \
  --model_name $MODEL \
  --model_path $MODEL_PATH \
  --task single-subimageVQA

python /gpfs/radev/pi/q_chen/yf329/PMC_instruction_data/benchmark/PMC_Data_Benchmark/evaluate_all.py \
  --model_name $MODEL \
  --model_path $MODEL_PATH \
  --task bboxVQA

python /gpfs/radev/pi/q_chen/yf329/PMC_instruction_data/benchmark/PMC_Data_Benchmark/evaluate_all.py \
  --model_name $MODEL \
  --model_path $MODEL_PATH \
  --task puretext

