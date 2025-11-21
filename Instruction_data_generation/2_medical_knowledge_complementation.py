import json
import torch
import os
import re
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM,AutoTokenizer
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time

# Argument Parser for Input File
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
args = parser.parse_args()

input_file = args.input_file  # Get the input JSON file path from the argument
output_dir = '<YOUR_OUTPUT_DIR>/PMC_instruction_data/stage1_2'  # Output directory
input_filename = os.path.basename(input_file)

# Updated regex to handle optional 'zc_' prefix
file_index_match = re.match(r'inline_summary_((?:zc_)?\d+_\d+)\.json$', input_filename)
file_idx = file_index_match.group(1) if file_index_match else "unknown"

output_file = os.path.join(output_dir, f"background_info_{file_idx}.json")



# def parse_llm_output(text):
#     """
#     Extracts 'keywords' and 'medical knowledge' from a structured text block.
#     """
#     # Match: **Keywords:** ...
#     keywords_match = re.search(r"\*\*Keywords\*\*", text)
#     keywords = []
#     if keywords_match:
#         keywords_str = keywords_match.group(1).strip()
#         keywords = [kw.strip() for kw in re.split(r",\s*", keywords_str)]

#     # Match: **Medical Knowledge:** ...
#     medical_match = re.search(r"\*\*Medical Knowledge\*\*", text, re.DOTALL)
#     medical_knowledge = medical_match.group(1).strip() if medical_match else ""

#     return keywords, medical_knowledge

# System Prompt
SYSTEM_PROMPT = """
### Role
You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your primary task is to generate concise and accurate medical knowledge for clinical, academic, and AI training purposes in English.

### Task
#### **Input Description**
The input includes:  
1. **Caption**: Describes the visual contents of medical images (e.g., CT, MRI, histology). 
2. **Inline Summary**: A concise summary of key medical observations, analyses, and conclusions related to the compound image.  
#### **Specific Task**
1. Extract medically significant keywords from both the input Caption and Inline Summary.  
2. Generate concise medical knowledge focusing on the clinical or diagnostic relevance of the extracted keywords.

### Objective
1. **Accuracy**:  
   - Ensure all extracted keywords and medical knowledge are medically accurate and supported by established guidelines or literature.
2. **Conciseness**:  
   - Limit the medical knowledge to 5 sentences, focusing on the most clinically relevant details.
3. **Professionalism**:  
   - Use formal, standardized medical terminology.
4. **Relevance**:  
   - Align the medical knowledge with the extracted keywords and clinical context.

### Instructions
1. **Extract Keywords**:  
   - Identify key medical terms directly related to the Caption and Inline Summary, such as diseases, imaging modalities, anatomical structures, or findings.
2. **Generate Knowledge**:  
   - Create concise medical knowledge based on the extracted keywords, emphasizing their clinical or diagnostic importance.

### Output Format
- **Keywords**: A list of extracted keywords.  
- **Medical Knowledge**: A concise explanation of the keywords, limited to 5 sentences.

### Example
#### **Input**
- **Caption**: CT scan of the chest showing a solitary pulmonary nodule in the upper lobe of the right lung.
- **Inline Summary**: Axial and coronal CT scans show a well-circumscribed pulmonary nodule in the right upper lobe, with no calcification or ground-glass opacity, reducing the likelihood of malignancy. Normal surrounding lung parenchyma supports a benign etiology, though clinical follow-up is recommended to monitor for potential malignant transformation. 
#### **Output**
- **Keywords**: ["pulmonary nodule", "right upper lobe", "benign etiology", "CT imaging"]
- **Medical Knowledge**: A pulmonary nodule is a small, round lesion within the lung that can represent benign or malignant conditions. CT imaging is essential for evaluating nodules, providing detailed information about their size, margins, and internal characteristics. Smooth margins and the absence of calcification or ground-glass opacity suggest a benign etiology. However, clinical follow-up is recommended to rule out malignancy, as some nodules may transform over time. The location in the right upper lobe and normal surrounding lung parenchyma further support the benign diagnosis.
"""

# Load Model and Processor
# Alternative model loading (commented out)
# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2.5-32B-Instruct-AWQ", torch_dtype="auto", device_map="auto", cache_dir="<YOUR_HF_CACHE_DIR>"
# )

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct-AWQ", torch_dtype="auto", device_map="auto", cache_dir="<YOUR_HF_CACHE_DIR>")

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir="<YOUR_HF_CACHE_DIR>")

# Load Dataset
with open(input_file, "r") as f:
    data = json.load(f)


def construct_user_prompt(entry):
    compound_caption = entry.get("caption", "")
    compound_inline_text = entry.get("Inline Summary", "")
    
    user_input = (
        f"#### **Input**\n"
        f"- **Caption**: {compound_caption}\n"
        f"- **Inline Summary**: {compound_inline_text}"
    )
    
    return user_input


def generate_instruction(user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()
    print(output_text)
    lower_text = output_text.lower()

    pattern = re.search(
        r"(?:### Output:)?\s*"
        r"-?\s*\**keywords?\**[:：]?\s*(.*?)\n+"
        r"-?\s*\**medical knowledge\**[:：]?\s*(.*)",
        output_text,
        re.IGNORECASE | re.DOTALL,
    )


    keywords = []
    background_info = ""

    if pattern:
        kw_text = pattern.group(1).strip()
        mk_text = pattern.group(2).strip()

        keywords = [kw.strip() for kw in re.split(r",\s*", kw_text) if kw.strip()]
        background_info = mk_text

    return background_info, keywords, output_text






# Process Each Image-Caption Pair
for entry in tqdm(data):
    #compound_image_path = os.path.join(COMPOUND_FOLDER, entry["image"])
    
    # Generate new description for the compound figure
    background_info, keywords, output_text = generate_instruction(entry["caption"])
    entry["medical knowledge"] = background_info
    entry["keywords"] = keywords
    #entry["raw_output"] = output_text

    # Generate descriptions for subfigures
    # if "subcaptions" in entry and entry["subcaptions"]:
    #     for sub in entry["subcaptions"]:
    #         #subfigure_image_path = os.path.join(sub["image"])
    #         sub_background_info, sub_keywords, sub_output_text = generate_instruction(sub["text"],sub["rewritten_caption"])
    #         sub["medical knowledge"] = sub_background_info
    #         sub["keywords"] = sub_keywords  # Add extracted keywords
    #         sub["raw_output"] = sub_output_text
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Descriptions generated and saved to: {output_file}")
