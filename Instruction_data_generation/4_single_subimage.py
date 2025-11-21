import json
import torch
import os
import re
import argparse
import random
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM,AutoTokenizer
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time
import pandas as pd

"""
Argument Parser for Input File
Get the input JSON file path from the argument and Save output in the same directory as input
Then, Extract index from filename (e.g., captions_1.json → 1)
"""
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--classification_file", type=str, required=True, help="Path to the classification CSV file.")

args = parser.parse_args()

input_file = args.input_file
classification_csv = args.classification_file
output_dir = '<YOUR_OUTPUT_DIR>/PMC_Insturction_Data/stage4_single_subimage'
input_filename = os.path.basename(input_file)

# Match "random_chunk_000.json"
file_index = re.search(r'random_chunk_(\d+)\.json$', input_filename)
file_idx = file_index.group(1) if file_index else "unknown"

# Construct output file name using this index
output_file = os.path.join(output_dir, f"final_instruction_single_subimage_QA_{file_idx}.json")


# === Load classification CSV ===
df = pd.read_csv(classification_csv)
confidence_dict = dict(zip(df["filename"], df["prob"]))

# === Load Model & Tokenizer ===
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="<YOUR_HF_CACHE_DIR>"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
    cache_dir="<YOUR_HF_CACHE_DIR>"
)

def generate_system_prompt():
    """
    Construct system prompt
    """
    
    SYSTEM_PROMPT = """
    ### **Role**
    You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to generate structured and concise Context-Question-Answer outputs based on the analysis of medical images for clinical education and AI training.

    ### **Task**
    #### **Data Description**
    The input includes:  
    1. **Compound Image Information**:  
    - Index of the compound image.  
    - Caption of the compound image.  
    - Inline Summary of observations, analyses and conclusions related to the compound image.  
    - Medical knowledge related to the compound image.  
    - Visual perception description of the compound image.  
    2. **Sub-Image List**: A list of indices for all sub-images within the compound image.  
    3. **Specific Sub-Image Details**:  
    - Index of the specific sub-image.  
    - Caption of the sub-image.  
    - Visual perception description of the sub-image.  
    #### **Specific Task**
    Generate a structured output consisting of:  
    1. **Context**: A concise medical background introducing the compound image, based on its Caption, Inline Summary and Medical Knowledge.  
    2. **Question**: A  clear question about the specific sub-image that encourages detailed observation with clinical value.  
    3. **Answer**: A precise and accurate answer regarding the question.  
    The output must focus on the visible content of the specific sub-image to guide understanding and interpretation.

    ### **Objective**
    1. **Accuracy**:  
    - Ensure all outputs are medically accurate and strictly based on visible features or provided references.
    2. **Clarity**:  
    - Keep Context, Question, and Answer concise, clear, and relevant.
    3. **Relevance**:  
    - Focus on the visible findings and their clinical significance without introducing unrelated or speculative details.
    4. **Answer Separation**:  
    - Ensure the key information in the Answer does not appear in the Context or Question.
    5. **Focus on Sub-Image**:  
    - Ensure the Question and Answer specifically address visible features of the assigned sub-image, without referencing unrelated sub-images or details.

    ### **Instructions**
    1. **Use References**:  
    - Base outputs on the provided compound image and sub-image details.  
    - Avoid speculative or unrelated details.
    2. **Maintain Clarity**:  
    - Write all outputs in a clear, professional, and concise manner, adhering to medical accuracy.
    3. **Refer Images by Index**:  
    - Use the assigned numeric index (e.g., Fig-0 for the compound image, Fig-1 for the first sub-image, etc.) when referring to the compound image and sub-images.

    ### **Output Format**
    - **Context**: A concise and relevant medical background introducing the compound image.  
    - **Question**: A specific and clear question about the image that encourages detailed observation with clinical value.  
    - **Answer**: A precise and accurate answer regarding the question.

    ### **Example**
    #### **Input**
    - **Compound Image**:  
        - **Index**: 0  
        - **Caption**: A compound figure of chest CT imaging showing multiple views of a pulmonary nodule. 
        - **Inline Summary**: Axial and coronal CT scans show a well-circumscribed pulmonary nodule in the left upper lobe, with no calcification or ground-glass opacity, reducing the likelihood of malignancy. Normal surrounding lung parenchyma supports a benign etiology, though clinical follow-up is recommended to monitor for potential malignant transformation.  
        - **Medical Knowledge**: Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. Imaging techniques from multiple planes, such as axial and coronal views, are critical for assessing size, shape, margins, and surrounding tissue.  
        - **Visual Perception Description**: The compound figure contains multiple sub-images showing different views of a solitary pulmonary nodule in the left upper lobe. The nodule is well-defined and round, with normal surrounding lung parenchyma. 
    - **Sub-Image List**:  
        - Index: 1  
        - Index: 2  
    - **Specific Sub-Image**:  
        - **Index**: 1  
        - **Caption**: An axial CT image of the chest demonstrating a solitary pulmonary nodule in the left upper lobe.  
        - **Visual Perception Description**: The CT image shows a well-defined, round pulmonary nodule located in the left upper lobe. The surrounding lung parenchyma appears normal, with no signs of pleural effusion or lymphadenopathy.  
    #### **Output**
    - **Context**: This axial CT image (Fig-1) shows a solitary pulmonary nodule in the left upper lobe. Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. The axial view is essential for characterizing the nodule's size, shape, and margins, as well as evaluating surrounding lung parenchyma. 
    - **Question**: What are the key visible imaging features of the pulmonary nodule in (Fig-1)?
    - **Answer**: The axial CT image (Fig-1) demonstrates a well-defined, round pulmonary nodule located in the left upper lobe. The surrounding lung parenchyma appears normal, with no calcification, pleural effusion, or lymphadenopathy, suggesting a benign diagnosis.
    """


    return SYSTEM_PROMPT.strip()



# === Define helper functions ===
def construct_user_prompt(sample, selected_subfigs):
    compound_caption = sample.get("caption", "").strip()
    medical_knowledge = sample.get("medical knowledge", "").strip()
    inline_summary = sample.get("references", "")
    compound_image = sample.get("image", "")

    def get_idx(sf):
        raw_idx = sf.get('idx', sf.get('sub_idx', 'unknown'))
        if isinstance(raw_idx, int):
            return raw_idx + 1
        try:
            return int(raw_idx) + 1
        except:
            return raw_idx

    lines = [
        "#### **Input**",
        "- **Compound Image**:",
        f"- **Compound Image**: {compound_image}",
        f"- **Caption**: {compound_caption}",
        f"- **Medical Knowledge**: {medical_knowledge}",
        f"- **Inline Summary**: {inline_summary}",
        "",
        "- **Selected Sub-Images**:",
        "- **Indices**: " + ", ".join([f"Fig-{get_idx(sf)}" for sf in selected_subfigs]),
        "",
        "- **Specific Sub-Image Details:**",
    ]

    for subfig in selected_subfigs:
        lines.extend([
            f"- **Index**: Fig-{get_idx(subfig)}",
            f"- **Subfigure**: {subfig.get('image', 'unknown')}",
            f"    - **Caption**: {subfig.get('text', 'unknown')}",
            f"    - **Visual Perception Description**: {subfig.get('visual perception description', 'unknown').strip()}",
            ""
        ])
    return "\n".join(lines)

def generate_instruction_for_qa(sample, selected_subfigs):
    SYSTEM_PROMPT = generate_system_prompt()
    user_prompt = construct_user_prompt(sample, selected_subfigs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    #print(prompt_text)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()
    #print(output_text)
    context_match = re.search(r'- \*\*Context\*\*:\s*(.*?)(?=\n- \*\*Question\*\*:)', output_text, re.DOTALL)
    question_match = re.search(r'- \*\*Question\*\*:\s*(.*?)(?=\n- \*\*Answer\*\*:)', output_text, re.DOTALL)
    answer_match = re.search(r'- \*\*Answer\*\*:\s*(.*)', output_text, re.DOTALL)

    context = context_match.group(1).strip() if context_match else ""
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    return {
        "context": context,
        "question": question,
        "answer": answer,
        "raw_output": output_text
    }




# === Load input data ===
with open(input_file, "r") as f:
    data = json.load(f)

results = []
SYSTEM_PROMPT = generate_system_prompt()

for sample in tqdm(data):
    subfigs = sample.get("subcaptions", [])
    if len(subfigs) < 1:
        continue

    # Add confidence scores
    for s in subfigs:
        s["confidence"] = confidence_dict.get(s["image"], 0.0)

    # Filter by confidence
    eligible = [s for s in subfigs if s["confidence"] > 0.7]
    if not eligible:
        continue

    # Randomly select 1 subimage
    selected = [random.choice(eligible)]

    # Generate QA
    qa_result = generate_instruction_for_qa(sample, selected)
    if qa_result is None:
        continue

    # Build metadata
    selected_image_paths = [s["image"] for s in selected]
    image_index = [f"Fig-{s.get('idx', '?')}" for s in selected]

    qa_pairs = [
        {
            "context": qa_result["context"],
            "question": qa_result["question"],
            "answer": qa_result["answer"],
            "selected_image_paths": selected_image_paths,
            "image_index": image_index,
            "raw_output": qa_result["raw_output"],
            "grouped_visual_word_count": len(qa_result["answer"].split())
        }
    ]

    sample["qa_pairs"] = qa_pairs
    results.append(sample)


# === Save to output JSON ===
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)