import json
import torch
import os
import re
import argparse
import random
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM,AutoTokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time

# Argument Parser for Input File
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
args = parser.parse_args()

input_file = args.input_file  # Get the input JSON file path from the argument
output_dir = '<YOUR_OUTPUT_DIR>/PMC_Insturction_Data/stage5_puretext'  # Save output in the same directory as input
input_filename = os.path.basename(input_file)  # Extract filename

# Extract index from filename 
file_index = re.search(r'final_instruction_puretextQA_(\d+)\.json$', input_filename)
file_idx = file_index.group(1) if file_index else "unknown"

# Define Output Filename
output_file = os.path.join(output_dir, f"final_instruction_recontext_puretextQA_{file_idx}.json")

def generate_system_prompt():
    """
    构建系统 Prompt, 定义 LLM 的角色和任务。
    """
    SYSTEM_PROMPT = """
    ### **Role**
    You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to revise and consolidate information to generate a single improved context for a compound image. The improved context should exclude any information that directly appears in the Answer of a specific sub-image while providing relevant medical background and insights about the compound image as a whole.

    ### **Task**
    #### **Data Description**
    The input includes:  
    1. **Compound Image Information**:  
        - Caption: A description summarizing the compound image’s content and clinical relevance.
        - Inline Summary: A concise summary of medical observations, analyses, and conclusions related to the compound image.
        - Visual Perception Description: Highlights visible findings in the compound image.  
    2. **Specific Instruction**:  
        - Context, Question, and Answer for the compound image.
    #### **Specific Task**
    Generate an improved context for the compound image:  
        - Refer to the compound image's Caption, Inline Summary and Visual Perception Description. 
        - Exclude any content of the original Context that directly appears in the Answer.

    ### **Objective**
    1. **Accuracy**:  
        - Ensure the improved context is medically accurate and avoids including any information from the Answer.
    2. **Clarity**:  
        - Keep the consolidated context concise, clear, and professionally written.
    3. **Relevance**:  
        - Focus on overall findings, mechanisms, or clinical significance of the compound image without introducing unrelated or speculative details.
    4. **Answer Separation**:  
        - Ensure the improved context does not leak any information that directly appears in the Answer.

    ### **Instructions**
    1. **Use References**:  
        - Base the improved context on the provided information, but avoid including any information that overlaps directly with the Answer.  
        - Consolidate relevant medical background or broader insights related to the compound image.
    2. **Remove Answer Information**:  
        - Exclude any content from the Context that overlaps with the Answer.  
        - Provide general information about the compound image without revealing specific details from the Answer.
    3. **Focus on Compound Image**:  
        - Ensure the improved context provides information that is relevant to the compound image as a whole.
    4. **Ensure Clarity and Conciseness**:  
        - Write the improved context in a clear, concise, and professional manner, adhering to medical accuracy and educational value.

    ### **Output Format**
    - **Improved Context**: A consolidated and comprehensive context for the compound image, excluding information that overlaps with the Answer.

    ### **Example**
    #### **Input**
    - **Compound Image**:  
        - **Index**: 0  
        - **Caption**: A compound figure of brain MRI scans showing axial, coronal, and sagittal views of a lesion in the right temporal lobe.
        - **Inline Summary**: MRI scans reveal a hyperintense lesion in the right temporal lobe on T2-weighted imaging. The lesion shows no enhancement on post-contrast imaging, suggesting a non-malignant etiology. There is mild surrounding edema, and no midline shift is observed.
        - **Visual Perception Description**: The compound image contains three sub-images displaying an MRI lesion in the right temporal lobe from axial, coronal, and sagittal perspectives. The lesion appears hyperintense on T2-weighted imaging and does not show any enhancement with contrast.
    - **Specific Instruction**:  
        - **Context**: This axial MRI image shows a hyperintense lesion in the right temporal lobe on T2-weighted imaging. Brain lesions are assessed for their intensity, location, and enhancement patterns to determine their etiology.
        - **Question**: What imaging features help differentiate malignant from non-malignant brain lesions?
        - **Answer**: Malignant brain lesions often show irregular or ring-enhancing patterns on post-contrast imaging, associated with significant surrounding edema, mass effect, and midline shift. Non-malignant lesions, in contrast, typically lack enhancement, have smoother borders, and are associated with less severe edema or mass effect.
    #### **Output**
    - **Improved Context**: This compound figure includes MRI scans of a lesion in the right temporal lobe, displayed in axial, coronal, and sagittal views. Brain lesions are evaluated based on their intensity, location, and enhancement patterns on different imaging planes. These characteristics, combined with clinical findings, help assess the lesion's etiology and guide further management.
    """
    return SYSTEM_PROMPT.strip()

SYSTEM_PROMPT = generate_system_prompt()
# Load the input JSON file
# Example input path (commented out for reference)



def summarize_sample_images(sample):
    images_info = []

    # Main image
    main_image_info = {
        "image": sample["image"],
        "text": sample.get("caption", ""),
        "rewritten_caption": sample.get("rewritten_caption", ""),
        "visual perception description": sample.get("visual perception description", ""),
        "Medical Knowledge": sample.get("Medical Knowledge", ""),
        "keywords": sample.get("keywords", ""),
        "type": "main",
        "qa_pairs": sample.get("qa_pairs", "")
    }
    images_info.append(main_image_info)

    # Sub-images
    if "subcaptions" in sample:
        for subfigure in sample["subcaptions"]:
            sub_image_info = {
                "image": subfigure["image"],
                "text": subfigure.get("text", ""),
                "rewritten_caption": subfigure.get("rewritten_caption", ""),
                "visual perception description": subfigure.get("visual perception description", ""),
                "Medical Knowledge": subfigure.get("Medical Knowledge", ""),
                "keywords": subfigure.get("keywords", ""),
                "type": "sub",
                "sub_idx": subfigure["idx"]
            }
            images_info.append(sub_image_info)

    return images_info

def build_structured_user_input(entry, qa_pair):
    compound_info = {
        "index": "0",
        "inline summary": entry.get("Inline Summary", ""),
        "medical knowledge": entry.get("medical knowledge", ""),
        "visual perception description": entry.get("visual perception description", "")
    }

    image_path = qa_pair.get("image_path", "")
    all_images = summarize_sample_images(entry)
    matched_image = next((img for img in all_images if img["image"] == image_path), None)

    if matched_image and matched_image.get("type") == "sub":
        sub_index = str(matched_image.get("sub_idx", ""))
    else:
        sub_index = "0"

    subimage_info = {
        "index": sub_index,
        "context": qa_pair.get("context", ""),
        "question": qa_pair.get("question", ""),
        "answer": qa_pair.get("answer", "")
    }

    user_input = (
        f"#### **Input**\n"
        f"- **Compound Image**:\n  "
        f"    - **Index**: 0"
        f"    - **Inline Summary**: {compound_info.get("inline summary")}\n"
        f"    - **Medical Knowledge**: {compound_info.get("medical knowledge")}\n"
        f"    - **Visual Perception Description**: {compound_info.get("visual perception description")}\n"
        f"- **Specific Sub-Image**:\n  "
        f"    - **Index**: {subimage_info.get("index")}\n"
        f"    - **Context**: {subimage_info.get("context")}\n"
        f"    - **Question**: {subimage_info.get("question")}\n"
        f"    - **Answer**: {subimage_info.get("answer")}"

    )    



    return user_input


# === Generate improved context ===
def clean_context(entry, qa_pair):
    user_input_json = build_structured_user_input(entry, qa_pair)
    user_prompt = json.dumps(user_input_json, indent=4)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    #print(prompt_text)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(output_text)
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()

    if "**Improved Context**" in output_text:
        output_text = output_text.split("**Improved Context**", 1)[1].strip()

    return output_text



# --- Main Cleaning Loop ---
if __name__ == "__main__":
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load Model and Processor
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct-AWQ", torch_dtype="auto", device_map="auto", cache_dir="<YOUR_HF_CACHE_DIR>"
    )

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct-AWQ', cache_dir="<YOUR_HF_CACHE_DIR>")

    for entry in tqdm(data, desc="Generating Improved Contexts"):
        qa_pairs = entry.get("qa_pairs", [])
        if not qa_pairs:
            # Fallback to existing question/answer fields
            q_keys = ["question", "Question"]
            a_keys = ["answer", "Answer"]
        
            question_text = next((entry.get(k) for k in q_keys if k in entry), None)
            answer_text = next((entry.get(k) for k in a_keys if k in entry), None)
        
            if question_text and answer_text:
                qa_pairs = [{
                    "context": entry.get("context", ""),
                    "question": question_text,
                    "answer": answer_text,
                    "image_path": entry.get("image", ""),
                    "image_index": "Fig-0"
                }]
                entry["qa_pairs"] = qa_pairs  # insert into entry
        
        for qa in qa_pairs:
            try:
                improved_context = clean_context(entry, qa)
                qa["improved context"] = improved_context
            except Exception as e:
                print(f"[Warning] Failed to clean context for idx={entry.get('idx', 'unknown')}: {e}")



# Save the modified dataset back to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Descriptions generated and saved to: {output_file}")

    #print(f"Updated JSON saved to {output_path}")