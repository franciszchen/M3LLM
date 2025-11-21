import json
import torch
import os
import re
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time

# Argument Parser for Input File
# === Argument Parser ===
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
args = parser.parse_args()

input_file = args.input_file
output_dir = '<YOUR_OUTPUT_DIR>/PMC_instruction_data/stage1_1'
input_filename = os.path.basename(input_file)

# === Extract prefix for output naming ===
# Supports both with and without "zc_"
prefix_match = re.match(r'^((?:zc_)?\d+_\d+)_filtered_converted_with_reference\.json$', input_filename)

if prefix_match:
    prefix = prefix_match.group(1)  # includes zc_ if present
    output_file = os.path.join(output_dir, f"inline_summary_{prefix}.json")
else:
    raise ValueError(f"Unexpected input filename format: {input_filename}")



def clean_caption(output_text):
    """
    Cleans the generated caption by:
    - Removing any prefix like "(a)", "a)", "Figure 1:", etc.
    - Ensuring the response is clean and concise.
    """
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()

    answer_match = re.search(r'\*\*Inline Summary\*\*:\s*(.*)', output_text, re.DOTALL)

    # Regular expression to remove unwanted prefixes like "(a)", "a)", "Figure 1:", etc.
    #output_text = re.sub(r"^\(?[a-zA-Z0-9]+\)?[:.)]\s*", "", output_text).strip()

    return output_text


# System Prompt
SYSTEM_PROMPT = """
### Role
You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your primary task is to summarize inline text from PubMed biomedical research papers, focusing on key medical observations, analyses, and conclusions related to patient conditions and diagnostic findings.

### Task
#### **Input Description**
The input consists of inline text, which refers to multiple text excerpts from a research paper where a compound medical image is mentioned. These excerpts describe the observations, analyses, or clinical interpretations of the compound image and are often scattered across the text. Inline text provides fragmented but relevant details about the image that need to be consolidated into a coherent summary.  
#### **Specific Task**
Summarize the key points from the inline text, focusing on:  
- Observations directly related to patient conditions, medical findings, and diagnostic conclusions.  
- Insights that highlight the clinical or research significance of the compound image.  
- Excluding irrelevant, repetitive, or vague information.  
The summary must be clear, concise, and strictly limited to 100 words.

### Objective
1. **Accuracy**:  
   - Extract only medically relevant details from the inline text.  
   - Avoid adding unverified or speculative information.  
2. **Clarity**:  
   - Ensure the summary is concise and easy to understand.    
   - Use formal, standardized medical terminology.
3. **Relevance**:
   - Focus on essential observations, analyses and conclusions related to the compound image.  
   - Exclude redundant or non-contributory content.  
4. **Brevity**:  
   - Eliminate redundancy or unnecessary complexity.
   - Consolidate information into a logical and compact form.  

### Instructions
1. **Focus on Key Medical Details**:  
   - Identify and include only details related to the patient conditions, diagnostic findings, or medical significance of the compound image.  
2. **Eliminate Irrelevant Content**:  
   - Remove vague, redundant, or non-medical details that do not contribute to understanding the compound image.
3. **Maintain Clarity and Precision**:  
   - Use clear, professional language suitable for academic and clinical audiences.  
4. **Word Limit**:  
   - Ensure the summary does not exceed 100 words. 

### Output Format
- **Inline Summary**: A concise summary of the inline text, focusing on the most important medical observations, analyses, and conclusions related to the compound image.

### Example
#### **Input**
- **Inline Text**: ["The compound image highlights axial and coronal CT views of a pulmonary nodule located in the left upper lobe. The nodule is well-circumscribed, suggesting a benign etiology. Surrounding lung parenchyma appears normal.",  
"Further analysis of the pulmonary nodule indicates no evidence of calcification or ground-glass opacity. These findings reduce the likelihood of malignancy.",  
"Although the nodule's smooth margins suggest benignity, clinical follow-up is recommended to rule out potential malignant transformation."]  
#### **Output**: 
- **Inline Summary**: Axial chest CT demonstrates a focal consolidation in the right lung, suggestive of a potential infectious process.
"""

# Load Model and Processor
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct-AWQ", torch_dtype="auto", device_map="auto", cache_dir="<YOUR_HF_CACHE_DIR>"
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir="<YOUR_HF_CACHE_DIR>")

# Load Dataset
with open(input_file, "r") as f:
    data = json.load(f)

# Function to Generate Image Caption (Extracting Only Assistant's Response)
def generate_instruction(references):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Rewrite this caption in a professional and medically precise manner:\n\n{references}"}
    ]

    # Prepare Input
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #image_inputs, video_inputs = process_vision_info(messages)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate Response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]  # Ensure we take only the first output
    #print(output_text)
    # Extract only the assistant's response
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()
    print(output_text)
    return clean_caption(output_text)  # Return only the assistant's response

# Process Each Image-Caption Pair
for entry in tqdm(data):
    #compound_image_path = os.path.join(COMPOUND_FOLDER, entry["image"])
    
    # Generate new description for the compound figure
    entry["Inline Summary"] = generate_instruction(entry["references"])

    # Generate descriptions for subfigures
    # if "subcaptions" in entry and entry["subcaptions"]:
    #     for sub in entry["subcaptions"]:
    #         #subfigure_image_path = os.path.join(sub["image"])
    #         sub["Inline Summary"] = generate_instruction(sub["text"])


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Inline summary generated and saved to: {output_file}")
