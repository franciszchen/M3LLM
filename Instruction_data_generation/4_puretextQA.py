import json
import torch
import os
os.environ["TRITON_CACHE_DIR"] = "<YOUR_HF_CACHE_DIR>"
import re
import argparse
import random
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM,AutoTokenizer
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time
import glob

from PIL import Image



"""
Argument Parser for Input File
Get the input JSON file path from the argument and Save output in the same directory as input
Then, Extract index from filename (e.g., captions_1.json → 1)
"""
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
args = parser.parse_args()


input_file = args.input_file
output_dir = '<YOUR_OUTPUT_DIR>/PMC_Insturction_Data/stage4_puretext'
input_filename = os.path.basename(input_file)

# Match "random_chunk_000.json"
file_index = re.search(r'random_chunk_(\d+)\.json$', input_filename)
file_idx = file_index.group(1) if file_index else "unknown"

# Construct output file name using this index
output_file = os.path.join(output_dir, f"final_instruction_puretextQA_benchmark_{file_idx}.json")


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



def generate_system_prompt():
    """
    Construct system prompt
    """
    
    SYSTEM_PROMPT = """
    ### **Role**  
    You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to generate structured and concise **Context-Question-Answer** outputs based on the text-based analysis of compound medical images for clinical education and AI training.
    Note: The model will **not be shown the image**. All outputs must be based solely on the provided textual information (caption, summary, and medical knowledge).

    ### **Task**
    #### **Data Description**  
    The input includes:
    1. **Compound Image Information**:  
    - **Index**: Unique identifier of the compound image (e.g., Fig-0).  
    - **Caption**: A description summarizing the compound image’s content and clinical relevance.  
    - **Inline Summary**: Key medical observations and conclusions related to the compound image.  
    - **Medical Knowledge**: Relevant diagnostic or clinical background information about the compound image.  
    #### **Specific Task**
    Generate a structured output consisting of:  
    1. **Context**: Summarize the **Caption** and **Inline Summary** into a concise medical background. The **Context** should focus on the key findings and clinical significance described in the input text and **exclude any Medical Knowledge details**.  
    2. **Question**: Create a **self-contained and insightful question** based on **Medical Knowledge**. The question should relate to the clinical concepts or keywords described in the context, but it must expand the discussion beyond the context by leveraging general medical knowledge.
    3. **Answer**: Provide a **precise and detailed response** to the question, expanding on the medical knowledge or principles relevant to the case.

    ### **Objective**  
    1. **Accuracy**:  
    - Ensure all outputs are medically accurate and based strictly on the provided references.  
    - Avoid speculative or unrelated details.  
    2. **Clarity**:  
    - Write in a clear, concise, and professional manner.  
    3. **Relevance**:  
    - Focus on meaningful and clinically significant concepts. 
    4. **Distinct Context, Question, and Answer**:
    - The **Context** should summarize the **Caption** and **Inline Summary**, while the **Question** and **Answer** should focus on expanding the discussion using **Medical Knowledge**.  
    - The **Answer** must introduce new content that is not already present in the **Context**.    
    
    ### **Instructions**
    1. **Summarize for Context**:  
    - Use the **Caption** and **Inline Summary** to create a concise, text-based summary. Avoid including any general medical knowledge or broader clinical concepts in this section.  
    2. **Expand with the Question**:  
    - Identify keywords or concepts in the context and use **Medical Knowledge** to construct an insightful question.  
    - Ensure the question is **self-contained** and does not depend on the context to be understood.  
    3. **Answer Separately**:  
    - Provide a detailed and accurate answer to the question, focusing on relevant medical knowledge.  
    - Avoid duplicating any information already present in the context.  
    4. **Examples of Good and Bad Practices**:
    - Good:  
    What are the common risk factors for pulmonary nodule malignancy, and how are they assessed clinically?
    - Bad:  
    What are the key observations in the compound figure?
    - Bad:  
    What does the image suggest about the lesion’s malignancy?
    **Reason**: These imply the model has visual access, which it does not.
    
    ### **Output Format**  
    - **Context**: A concise summary of the **Caption** and **Inline Summary**.  
    - **Question**: A clinically insightful question based on **Medical Knowledge** that expands the discussion beyond the **Context**.  
    - **Answer**: A precise and detailed response to the question, introducing information not present in the context.

    ### **Example**
    #### **Input**  
    - **Compound Image Information**:  
    - **Index**: 0  
    - **Caption**: A compound figure of chest CT imaging showing multiple views of a pulmonary nodule.  
    - **Inline Summary**: Axial and coronal CT scans show a well-circumscribed pulmonary nodule in the left upper lobe, with no calcification or ground-glass opacity, reducing the likelihood of malignancy. Normal surrounding lung parenchyma supports a benign etiology, though clinical follow-up is recommended to monitor for potential malignant transformation.  
    - **Medical Knowledge**: Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. Malignant nodules tend to exhibit irregular or spiculated margins, rapid growth, or ground-glass opacity. Benign nodules often have smooth, well-defined borders and may contain calcifications. Risk factors for malignancy include smoking history, older age, and a history of cancer.  
    #### **Output**  
    - **Context**: Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. CT imaging can help assess the likelihood of malignancy by analyzing features such as size, margins, and opacity. A well-circumscribed nodule with no calcification or ground-glass opacity, as described in this case, is less likely to be malignant.  
    - **Question**: What are the risk factors for pulmonary nodule malignancy, and what imaging features are most commonly associated with malignant nodules?  
    - **Answer**: Risk factors for pulmonary nodule malignancy include a history of smoking, older age, prior cancer diagnosis, and exposure to carcinogens. Imaging features commonly associated with malignancy include irregular or spiculated margins, ground-glass opacity, and rapid growth on serial imaging. These findings can help clinicians prioritize further diagnostic steps, such as biopsy or PET imaging, for high-risk patients.
    """

    return SYSTEM_PROMPT.strip()




# === Summary function ===
def summarize_sample_images(sample):
    main_image_info = {
        "idx": sample.get("idx", ""),
        "caption": sample.get("caption", ""),
        "visual perception description": sample.get("visual perception description", ""),
        "medical knowledge": sample.get("medical knowledge", ""),
        "inline summary": sample.get("inline summary", ""),
        "image": sample.get("image", ""),
        "type": "main"
    }
    return [main_image_info]

# === Visual grouping ===
def get_combined_visual_description(images_info, max_words=300, max_images=3):
    sampled_images = random.sample(images_info, min(max_images, len(images_info)))
    sorted_images = sorted(sampled_images, key=lambda x: images_info.index(x))

    grouped_lines = []
    total_words = 0
    used_images = []

    for img in sorted_images:
        desc = img.get("visual perception description", "").strip()
        word_count = len(desc.split())
        if not desc:
            continue
        if total_words + word_count > max_words:
            break

        fig_idx = "Fig-0" if img["type"] == "main" else f'Fig-{int(img.get("sub_idx", 0)) + 1}'
        #grouped_lines.append(f"Visual component of {fig_idx}: {desc}")
        total_words += word_count
        used_images.append(img)

    formatted_description = "\n".join(grouped_lines)
    target_image = random.choice(used_images) if used_images else sorted_images[0]
    return formatted_description, used_images, target_image, total_words

# === Prompt construction ===
def construct_user_prompt(entry, image_info, visual_summary_grouped):
    prompt_json = {
        "instruction": "Generate structured and concise **Context-Question-Answer** outputs based on the analysis of compound medical images for clinical education and AI training. Please directly generate answer using the output_format",
        "context": {
            "Compound Image": image_info.get("image", ""),
            "Caption": entry.get("caption", ""),
            "Index": '0',
            "Inline Summary": entry.get("Inline Summary", ""),
            "Medical Knowledge": entry.get("medical knowledge", ""),
            "Visual Perception Description": entry.get("visual perception description", "")
        }
    }
    return json.dumps(prompt_json, indent=4)


# === QA generation ===
from qwen_vl_utils import process_vision_info

def generate_instruction_for_qa(sample, image_info, visual_summary_grouped):
    user_prompt = construct_user_prompt(sample, image_info, visual_summary_grouped)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    #print(prompt_text)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(output_text)
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()

    context_match = re.search(r'- \*\*Context\*\*:\s*(.*?)(?=\s*- \*\*Question\*\*:)', output_text, re.DOTALL)
    question_match = re.search(r'- \*\*Question\*\*:\s*(.*?)(?=\s*- \*\*Answer\*\*:)', output_text, re.DOTALL)
    answer_match = re.search(r'- \*\*Answer\*\*:\s*(.*)', output_text, re.DOTALL)




    qa_result = {
        "context": context_match.group(1).strip() if context_match else "",
        "question": question_match.group(1).strip() if question_match else "",
        "answer": answer_match.group(1).strip() if answer_match else "",
        "image_path": image_info.get("image", ""),
        "image_index": f"Fig-{int(image_info.get('sub_idx', 0)) + 1}" if image_info["type"] == "sub" else "Fig-0",
        "raw_output": output_text,
        "grouped_visual_word_count": len(visual_summary_grouped.split())
    }

    return qa_result


# === Main wrapper ===
def construct_qa_pairs_for_sample(sample, max_visuals=3, max_words=500):
    images_info = summarize_sample_images(sample)
    visual_summary_grouped, _, target_image, total_word_count = get_combined_visual_description(
        images_info, max_words=max_words, max_images=max_visuals
    )
    sample["grouped_visual_words"] = total_word_count
    sample["grouped_visual_images"] = [img.get("image") for img in images_info]
    qa = generate_instruction_for_qa(sample, target_image, visual_summary_grouped)
    return [qa]

# === Load model and data ===
SYSTEM_PROMPT = generate_system_prompt()
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct-AWQ", torch_dtype="auto", device_map="auto", cache_dir="<YOUR_HF_CACHE_DIR>"
)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct-AWQ', cache_dir="<YOUR_HF_CACHE_DIR>")

# === Process input ===
with open(input_file, "r") as f:
    data = json.load(f)

for entry in tqdm(data):
    qa_pairs = construct_qa_pairs_for_sample(entry)
    entry["qa_pairs"] = qa_pairs

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Descriptions generated and saved to: {output_file}")

