import json
import torch
import os
import re
import argparse
import random
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM,AutoTokenizer
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import random
import time
import glob
import pandas as pd
from PIL import Image  # ← Add this


"""
Argument Parser for Input File
Get the input JSON file path from the argument and Save output in the same directory as input
Then, Extract index from filename (e.g., captions_1.json → 1)
"""
# === Parse input arguments ===
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--classification_file", type=str, required=True, help="Path to the classification CSV file.")
args = parser.parse_args()
#SUBFIGURE_FOLDER = "<YOUR_IMAGE_DIR>/mm_image_files/subfigures"  # Alternative path
input_file = args.input_file
classification_csv = args.classification_file

# === Prepare output filename ===
output_dir = '<YOUR_OUTPUT_DIR>/PMC_MLLM_data/stage4_multi_subimage'
input_filename = os.path.basename(input_file)

# Match "random_chunk_000.json"
file_index = re.search(r'random_chunk_(\d+)\.json$', input_filename)
file_idx = file_index.group(1) if file_index else "unknown"

# Construct output file name using this index
output_file = os.path.join(output_dir, f"final_instruction_multi_subimage_QA_{file_idx}.json")

SUBFIGURE_FOLDER = "<YOUR_IMAGE_DIR>/mm_image_files/subfigures"

COMPOUND_FOLDER_ROOT = "<YOUR_IMAGE_DIR>/mm_image_files/compound"

def find_compound_image_path(filename):
    # Search all subdirectories under compound root
    matches = glob.glob(os.path.join(COMPOUND_FOLDER_ROOT, "group_*", filename))
    if len(matches) == 0:
        print(f"[Warning] Compound image not found: {filename}")
        return None
    elif len(matches) > 1:
        print(f"[Warning] Multiple matches found for {filename}, using first: {matches[0]}")
    return matches[0]

def find_subfigure_image_path(filename):
    """
    Recursively search for a subfigure image under all 'subfigures' subfolders of SUBFIGURE_ROOT.
    """
    search_pattern = os.path.join(SUBFIGURE_FOLDER, "*", "subfigures", filename)
    matches = glob.glob(search_pattern)
    
    if len(matches) == 0:
        print(f"[Warning] Subfigure image not found: {filename}")
        return None
    elif len(matches) > 1:
        print(f"[Warning] Multiple matches found for {filename}, using first: {matches[0]}")
    
    return matches[0]

def resolve_image_path(image_info):
    if image_info["type"] == "main":
        return find_compound_image_path(image_info["image"])
    else:
        return find_subfigure_image_path(image_info["image"])



def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open image: {image_path}")
        return None

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
You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to generate structured and concise **Context-Question-Answer** outputs based on the analysis of multiple selected sub-images from a compound medical image. 

### **Task**
#### **Data Description**  
The input includes:  
1. **Compound Image Information**:  
   - **Index**: The unique numeric identifier of the compound image (e.g., Fig-0).  
   - **Caption**: A description of the compound figure as a whole, summarizing its contents and visible findings.
   - **Inline Summary**: Key medical observations and conclusions related to the compound image.  
   - **Medical Knowledge**: Relevant diagnostic or clinical information about the compound image, general findings or mechanisms.  
2. **Selected Sub-Images**:  
   - A list of indices for the selected sub-images (e.g., Fig-1, Fig-2).  
3. **Specific Details for Each Selected Sub-Image**:  
   - **Index**: The numeric identifier of the sub-image.  
   - **Caption**: A description of the visual content of the sub-image.  
   - **Visual Perception Description**: Observations of the sub-image’s visual features, including key findings and structures.  
   - **Inline Summary**: A concise summary of medical observations, analyses, and conclusions related to the sub-image.  
#### **Specific Task**  
Generate a structured output consisting of:  
1. **Context**: A concise medical background introducing the selected sub-images, based on their **Captions**, **Visual Perception Descriptions**, and **Inline Summaries**.  
2. **Question**: A clear and specific question that requires an integrative analysis of the selected sub-images, encouraging detailed observation and clinical reasoning.  
3. **Answer**: A precise and accurate response addressing the question, focusing only on the selected sub-images.
The outputs must focus **only on the selected sub-images** and avoid interference from other sub-images or the compound image as a whole.

### **Objective**  
1. **Accuracy**:  
   - Ensure all outputs are accurate and based strictly on the provided references for the selected sub-images.  
2. **Clarity**:  
   - Ensure the **Question** and **Answer** are logically connected and specific to the selected sub-images.  
3. **Relevance**:  
   - Focus on the selected sub-images, avoiding interference from other sub-images or the compound image.  

### **Instructions**  
1. **Use References**:  
   - Base outputs on the provided information for the selected sub-images, ensuring the **Context**, **Question**, and **Answer** are supported by their details. 
2. **Focus on Selected Sub-Images**:  
   - Ensure all outputs focus only on the selected sub-images, avoiding references to unselected sub-images or the compound image as a whole.  
3. **Integrate Sub-Image Information**:  
   - Synthesize information from the selected sub-images to create a cohesive **Context**, **Question**, and **Answer** that encourages integrative thinking.  

### **Output Format**  
- **Context**: A concise and relevant medical background introducing the selected sub-images.  
- **Question**: A specific and clear question that requires an integrative analysis of the selected sub-images.  
- **Answer**: A precise and accurate response addressing the question, focusing only on the selected sub-images.


### **Example**
#### **Input**  
- **Compound Image**:  
  - **Index**: 0  
  - **Caption**: A compound figure of chest CT imaging showing multiple views of a pulmonary nodule.  
  - **Medical Knowledge**: Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. Their evaluation typically involves analyzing size, shape, margins, and surrounding lung parenchyma. Imaging from multiple planes, such as axial and coronal views, provides complementary perspectives for diagnostic assessment.  

- **Selected Sub-Images**:  
  - **Indices**: Fig-1, Fig-2  

- **Specific Sub-Image Details**:  
  - **Index**: Fig-1  
    - **Caption**: An axial CT image of the chest demonstrating a solitary pulmonary nodule in the left upper lobe.  
    - **Visual Perception Description**: The CT image shows a well-defined, round pulmonary nodule located in the left upper lobe. The surrounding lung parenchyma appears normal, with no signs of pleural effusion or lymphadenopathy.  
    - **Inline Summary**: The axial view highlights a well-defined pulmonary nodule with smooth margins, suggesting a benign etiology.  

  - **Index**: Fig-2  
    - **Caption**: A coronal CT image showing the same pulmonary nodule in the left upper lobe from a different plane.  
    - **Visual Perception Description**: The coronal CT image demonstrates the nodule's position relative to the surrounding lung structures, confirming its well-defined borders and normal adjacent lung parenchyma.  
    - **Inline Summary**: The coronal view provides complementary information about the nodule's location and its relationship to nearby lung structures, supporting the interpretation of a benign lesion.

#### **Output**  
- **Context**: The selected sub-images (Fig-1 and Fig-2) highlight axial and coronal CT views of a solitary pulmonary nodule located in the left upper lobe. Axial and coronal imaging provide complementary perspectives for evaluating the nodule’s size, margins, and its relationship to adjacent lung structures. Together, these views confirm the nodule’s well-defined appearance and normal surrounding lung parenchyma, which are key features in assessing its clinical significance.  
- **Question**: How do the axial and coronal CT views (Fig-1 and Fig-2) contribute to the evaluation of the pulmonary nodule in the left upper lobe?  
- **Answer**: Axial and coronal CT views provide complementary information for assessing the pulmonary nodule. The axial view characterizes the nodule’s shape, size, and margins, while the coronal view offers insights into its location and relationship to surrounding lung structures. Together, these perspectives confirm the nodule’s well-defined borders and the absence of abnormalities in the adjacent lung parenchyma, supporting the interpretation of a benign lesion.  
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
        "- **Index**: Fig-0",
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
    if len(subfigs) < 2:
        continue

    # Add confidence scores to subfigures
    for s in subfigs:
        s["confidence"] = confidence_dict.get(s["image"], 0.0)

    # Select subfigures with confidence > 0.7, up to 5
    qualified = [s for s in subfigs if s["confidence"] > 0.7]
    # selected = selected[:5]
    
    num_to_select = random.choice([2, 3])
    if len(qualified) < num_to_select:
        continue  # Not enough to fulfill random choice
    
    #selected = selected[:3] # cz added, up to 3 sub-images #有个问题，如果数量足够，似乎固定是三张
    selected = random.sample(qualified, num_to_select)
    
    if len(selected) < 2:
        continue  # Require at least 2 selected subfigs for meaningful QA

    # Run QA generation
    qa_result = generate_instruction_for_qa(sample, selected)
    if qa_result is None:
        continue

    # Collect metadata
    selected_image_paths = [s["image"] for s in selected]
    image_index = [f"Fig-{s.get('idx', '?')}" for s in selected]

    # Store results
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

print(f"✅ Saved results to {output_file}")
