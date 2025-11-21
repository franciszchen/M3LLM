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
output_dir = '<YOUR_OUTPUT_DIR>/PMC_Insturction_Data/stage4_compoundVQA'
input_filename = os.path.basename(input_file)

# Match "random_chunk_000.json"
file_index = re.search(r'random_chunk_(\d+)\.json$', input_filename)
file_idx = file_index.group(1) if file_index else "unknown"

# Construct output file name using this index
output_file = os.path.join(output_dir, f"final_instruction_compoundVQA_{file_idx}.json")



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

SUBFIGURE_FOLDER = "<YOUR_IMAGE_DIR>/mm_image_files/subfigures"
COMPOUND_ROOT = "<YOUR_IMAGE_DIR>/mm_image_files/compound"

def find_compound_image_path(filename):
    """
    Search compound image across group_1 to group_16 folders.
    Return full path if found, else None.
    """
    for i in range(1, 17):  # group_1 to group_16
        group_path = os.path.join(COMPOUND_ROOT, f"group_{i}", filename)
        if os.path.isfile(group_path):
            return group_path
    return None

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
    You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to generate structured and concise **Context-Question-Answer** outputs based on the analysis of compound medical images for clinical education and AI training. 

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
    1. **Context**: A concise medical background introducing the compound image, based on its **Caption**, **Inline Summary**, and **Medical Knowledge**.  
    2. **Question**: A clear and specific question that encourages detailed clinical analysis of the compound image.  
    3. **Answer**: A precise and accurate response, focusing solely on the compound image as a whole.
    The outputs should focus on the **compound image as a whole**, avoiding sub-image-specific details unless they are critical.

    ### **Objective**  
    1. **Accuracy**:  
    - Ensure all outputs are medically accurate and based strictly on the provided references.  
    - Avoid speculative or unrelated details.  
    2. **Clarity**:  
    - Write in a clear, concise, and professional manner for readability and educational value.  
    3. **Relevance**:  
    - Focus on the compound image’s overall findings and clinical significance.  
    4. **Avoid Sub-Image Interference**:  
    - Do not include sub-image-specific details unless they are essential for understanding the compound image.  

    ### **Instructions**
    1. **Use References**:
    - Base outputs on the provided Compound Image Information, ensuring they address the compound image as a whole.
    2. **Focus on the Compound Image**:
    - Ensure all outputs provide insights into the compound image’s overall findings, clinical importance, or diagnostic significance.
    - Avoid focusing on specific sub-images or their details unless explicitly relevant to the compound image.
    3. **Answer Separation**:  
    - Ensure the **Answer** provides unique information not repeated in the **Context** or **Question**.  

    ### **Output Format**  
    - **Context**: Concise medical background for the compound image.  
    - **Question**: A specific clinical question about the compound image.  
    - **Answer**: A precise response based on the provided textual information.

    ### **Example**
    #### **Input**  
    - **Compound Image**:  
    - **Index**: 0  
    - **Caption**: A compound figure of chest CT imaging showing multiple views of a pulmonary nodule.  
    - **Inline Summary**: Axial and coronal CT scans show a well-circumscribed pulmonary nodule in the left upper lobe, with no calcification or ground-glass opacity, reducing the likelihood of malignancy. Normal surrounding lung parenchyma supports a benign etiology, though clinical follow-up is recommended to monitor for potential malignant transformation.  
    - **Medical Knowledge**: Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. Their evaluation involves analyzing size, shape, margins, and surrounding lung parenchyma. Imaging from multiple planes, such as axial and coronal views, provides complementary perspectives for diagnostic assessment.  
    #### **Output**  
    - **Context**: This compound figure includes multiple CT views of a pulmonary nodule in the left upper lobe. Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. Their evaluation typically involves analyzing size, shape, margins, and the condition of surrounding lung parenchyma. Imaging from multiple planes, such as axial and coronal views, provides detailed information about the nodule’s characteristics and its relationship to surrounding structures.  
    - **Question**: What is the clinical significance of using multiple CT views to evaluate pulmonary nodules in Fig-0?  
    - **Answer**: Multiple CT views, such as axial and coronal planes, provide complementary perspectives for assessing pulmonary nodules. They help determine the nodule’s size, shape, boundaries, and its relationship to nearby tissues. This approach aids in distinguishing between benign and malignant nodules and guides further diagnostic or treatment decisions.  
        """

    return SYSTEM_PROMPT.strip()


# === Define helper functions ===
def construct_user_prompt(sample):
    compound_caption = sample.get("caption", "").strip()
    medical_knowledge = sample.get("medical knowledge", "").strip()
    inline_summary = sample.get("references", "")
    compound_image = sample.get("image", "")

    lines = [
        "#### **Input**",
        f"- **Compound Image**: {compound_image}",
        f"- **Caption**: {compound_caption}",
        f"- **Medical Knowledge**: {medical_knowledge}",
        f"- **Inline Summary**: {inline_summary}"
    ]

    return "\n".join(lines)


def generate_instruction_for_qa(sample):
    SYSTEM_PROMPT = generate_system_prompt()
    user_prompt = construct_user_prompt(sample)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()

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
    qa_result = generate_instruction_for_qa(sample)
    if qa_result is None:
        continue

    qa_pairs = [
        {
            "context": qa_result["context"],
            "question": qa_result["question"],
            "answer": qa_result["answer"],
            "raw_output": qa_result["raw_output"],
            "grouped_visual_word_count": len(qa_result["answer"].split())
        }
    ]

    sample["qa_pairs"] = qa_pairs
    results.append(sample)

# === Save to output JSON ===
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
