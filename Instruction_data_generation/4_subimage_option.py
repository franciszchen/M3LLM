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
# === Parse input arguments ===
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--classification_file", type=str, required=True, help="Path to the classification CSV file.")
args = parser.parse_args()

input_file = args.input_file
classification_csv = args.classification_file

# === Prepare output filename ===
output_dir = '<YOUR_OUTPUT_DIR>/PMC_Insturction_Data/stage4_subimage_option'
input_filename = os.path.basename(input_file)

# Match "random_chunk_000.json"
file_index = re.search(r'random_chunk_(\d+)\.json$', input_filename)
file_idx = file_index.group(1) if file_index else "unknown"

# Construct output file name using this index
output_file = os.path.join(output_dir, f"final_instruction_subimage_option_{file_idx}.json")



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
    You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to generate structured and concise Four-Choice Question-Answer outputs based on the analysis of medical images for clinical education and AI training. 

    ### **Task**
    #### **Data Description**  
    The input includes:  
    1. **Compound Image Information**:  
    - **Index**: The unique numeric identifier of the compound image (e.g., Fig-0).  
    - **Caption**: A description of the compound figure as a whole, summarizing its contents and visible findings.  
    - **Inline Summary**: A concise summary of medical observations, analyses, and conclusions related to the compound image.  
    - **Medical Knowledge**: Relevant diagnostic or clinical information about the compound image, emphasizing general findings or mechanisms.  
    - **Visual Perception Description**: Observations of the compound image’s overall visual features, including key findings and structures.  
    2. **Specific Sub-Image Details**:  
    - **Index**: The numeric identifier of the sub-image (e.g., Fig-1).  
    - **Caption**: A description of the visual content of the sub-image.  
    - **Visual Perception Description**: Observations of the sub-image’s visual features, including spatial orientation and key findings.  
    #### **Specific Task**  
    Generate a structured output consisting of:  
    1. **Question**: A clear and specific four-choice question about the compound image or selected sub-image(s) that encourages detailed analysis and clinical reasoning.  
    2. **Options**: Four plausible answer options, with one correct option and three incorrect distractors. The distractors should be relevant but clearly incorrect based on the input data.  
    3. **Correct Answer**: Indicate the correct option (i.e., A, B, C or D)  from the four choices.  
    The four-choice question should derive from the **caption** content of the given image and maintain relevance to the visual and clinical context, with one correct answer and three plausible but incorrect distractors.

    ### **Objective**  
    1. **Accuracy**:  
    - Ensure all outputs are medically accurate and based strictly on the provided references for the compound image or sub-image(s).    
    2. **Clarity**:  
    - Write all outputs in a clear, concise, and professional manner.  
    3. **Relevance**:  
    - Focus on clinically significant findings or diagnostic insights from the input data.  
    4. **Option Plausibility**:  
    - Ensure the distractors are relevant to the context and plausible enough to challenge the reader, but clearly incorrect upon analysis.  
    - The correct answer will have an equal 25% probability of distribution across A, B, C, or D in the Options list.

    ### **Instructions**  
    1. **Use References**:  
    - Base outputs on the provided information for the compound image or sub-image(s), ensuring the **Question**, and **Options** are supported by their details.  
    2. **Focus on Clinical Value**:  
    - Ensure the question and options are clinically relevant and encourage critical thinking about the image data.  
    3. **Create Plausible Distractors**:  
    - Distractors should be related to the same clinical or diagnostic context but be clearly incorrect upon closer analysis.  

    ### **Output Format**  
    - **Question**: A specific and clear four-choice question that encourages detailed analysis and clinical reasoning.  
    - **Options**: Four plausible answer options, with one correct answer and three incorrect distractors.  
    - **Correct Answer**: Indicate the correct option (e.g., A, B, C, or D).  

    ### **Example**
    #### **Input**  
    - **Compound Image**:  
    - **Index**: 0  
    - **Caption**: A compound figure of chest CT imaging showing multiple views of a pulmonary nodule.  
    - **Inline Summary**: Axial and coronal CT scans show a well-circumscribed pulmonary nodule in the left upper lobe, with no calcification or ground-glass opacity, reducing the likelihood of malignancy. Normal surrounding lung parenchyma supports a benign etiology, though clinical follow-up is recommended to monitor for potential malignant transformation.  
    - **Medical Knowledge**: Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. Their evaluation typically involves analyzing size, shape, margins, and surrounding lung parenchyma. Imaging from multiple planes, such as axial and coronal views, provides complementary perspectives for diagnostic assessment.  
    - **Specific Sub-Image**:  
    - **Index**: Fig-1  
    - **Caption**: An axial CT image of the chest demonstrating a solitary pulmonary nodule in the left upper lobe.  
    - **Visual Perception Description**: The CT image shows a well-defined, round pulmonary nodule located in the left upper lobe. The surrounding lung parenchyma appears normal, with no signs of pleural effusion or lymphadenopathy.  
    #### **Output**  
    - **Question**: Based on Fig-1, which of the following features most strongly supports a benign diagnosis for the pulmonary nodule?  
    - **Options**:  
    A. Presence of ground-glass opacity  
    B. Smooth, well-defined borders  
    C. Enlarged mediastinal lymph nodes  
    D. Irregular shape with spiculated margins  
    - **Correct Answer**: B           
"""

    return SYSTEM_PROMPT.strip()



# === Define helper functions ===
def construct_user_prompt(sample, selected_subfigs):
    compound_caption = sample.get("caption", "").strip()
    medical_knowledge = sample.get("Medical_Knowledge", "").strip()
    inline_summary = sample.get("references", "")
    lines = [
        "#### **Input**",
        "- **Compound Image**:",
        "-**Index**: 0",
        f"- **Caption**: {compound_caption}",
        f"- **Inline Summary**: {inline_summary}",
        f"- **Medical Knowledge**: {medical_knowledge}"
        
        "",
        "- **Specific Sub-Image**:",
        # "- **Index**: " + ", ".join([f"Fig-{int(sf.get('idx', 0)) + 1}" for sf in selected_subfigs]),
        # "",
        # "- **Specific Sub-Image Details:**"
    ]
    for subfig in selected_subfigs:
        lines.extend([
            f"- **Index**: Fig-{subfig.get('idx', 'unknown')+1}",
            f"    - **Caption**: {subfig.get('text', '').strip()}",
            f"    - **Visual Perception Description**: {subfig.get('visual perception description', '').strip()}",
            # f"    - **Inline Summary**: {subfig.get('inline_summary', '').strip()}",
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
    #print(prompt_text)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #print(output_text)    
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()
        
    #context_match = re.search(r'\*\*Context\*\*:\s*(.*?)(?=\n+- \*\*Question\*\*:)', output_text, re.DOTALL)
    question_match = re.search(r'\*\*Question\*\*:\s*(.*?)(?=\n+- \*\*Options\*\*:)', output_text, re.DOTALL)
    options_match = re.search(r'\*\*Options\*\*:\s*(.*?)(?=\n+- \*\*Correct Answer\*\*:)', output_text, re.DOTALL)
    answer_match = re.search(r'\*\*Correct Answer\*\*:\s*(.*)', output_text, re.DOTALL)

    return {
        #"context": context_match.group(1).strip() if context_match else "",
        "question": question_match.group(1).strip() if question_match else "",
        "options": options_match.group(1).strip() if options_match else "",
        "correct answer": answer_match.group(1).strip() if answer_match else "",
        "raw_output": output_text
    }

# === Load input data ===
with open(input_file, "r") as f:
    data = json.load(f)

results = []
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
    image_index = [f"Fig-{int(s.get('idx', 0)) + 1}" for s in selected]


    qa_pairs = [
        {
            "question": qa_result["question"],
            "option": qa_result["options"],
            "answer": qa_result["correct answer"],
            "selected_image_paths": selected_image_paths,
            "image_index": image_index,
            "raw_output": qa_result["raw_output"],
            "grouped_visual_word_count": len(qa_result["correct answer"].split())
        }
    ]

    sample["qa_pairs"] = qa_pairs
    results.append(sample)



# === Save result ===
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved results to {output_file}")