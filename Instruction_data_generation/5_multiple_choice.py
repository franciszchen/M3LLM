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

# Argument Parser for Input File
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
args = parser.parse_args()

input_file = args.input_file  # Get the input JSON file path from the argument
output_dir = '<YOUR_OUTPUT_DIR>/PMC_Insturction_Data/stage5_subimage_option'  # Save output in the same directory as input
input_filename = os.path.basename(input_file)  # Extract filename

# Extract index from filename 
file_index = re.search(r'final_instruction_subimage_option_(\d+)\.json$', input_filename)
file_idx = file_index.group(1) if file_index else "unknown"

# Define Output Filename
output_file = os.path.join(output_dir, f"final_instruction_recontext_subimage_option_{file_idx}.json")

def generate_system_prompt():
    """
    Construct system prompt for improving multiple choice question context.
    """
    SYSTEM_PROMPT = """
    ### **Role**
    You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to revise and consolidate information to generate a single improved context for a multiple-choice question about a compound image or its sub-images. The improved context should exclude any information that directly reveals the correct answer while providing relevant medical background and insights.

    ### **Task**
    #### **Data Description**
    The input includes:  
    1. **Compound Image Information**:  
        - Caption: A description summarizing the compound image's content and clinical relevance.
        - Inline Summary: A concise summary of medical observations, analyses, and conclusions related to the compound image.
        - Medical Knowledge: Relevant background information about the compound image.
        - Visual Perception Description: Highlights visible findings in the compound image.  
    2. **Specific Sub-Image Information**:  
        - Index: The identifier of the selected sub-image(s).
        - Caption: Description of the sub-image's visual content.
        - Visual Perception Description: Observations of the sub-image's features.
    3. **Multiple Choice Question**:  
        - Question: A four-choice question about the image.
        - Options: Four answer options (A, B, C, D).
        - Correct Answer: The correct option letter.
    #### **Specific Task**
    Generate an improved comprehensive context for the multiple-choice question by consolidating:  
        - The compound image's Caption, Inline Summary, Medical Knowledge, and Visual Perception Description.  
        - The selected sub-image's information.  
    **Requirement**: Exclude any content that directly reveals or strongly hints at the correct answer, while maintaining sufficient information for the question to be answerable.

    ### **Objective**
    1. **Accuracy**:  
        - Ensure the improved context is medically accurate and avoids revealing the correct answer.
    2. **Clarity**:  
        - Keep the consolidated context concise, clear, and professionally written.
    3. **Relevance**:  
        - Focus on overall findings, mechanisms, or clinical significance relevant to the question without introducing unrelated or speculative details.
    4. **Answer Separation**:  
        - Ensure the improved context provides sufficient background for answering the question but does not leak information that directly identifies the correct answer.
    5. **Educational Value**:
        - Maintain educational value by providing enough context for clinical reasoning while preserving the challenge of the multiple-choice question.

    ### **Instructions**
    1. **Use References**:  
        - Base the improved context on the provided compound and sub-image information.
        - Avoid including specific details that make the correct answer obvious.  
        - Consolidate relevant medical background or broader insights related to the images.
    2. **Remove Answer-Revealing Information**:  
        - Identify and exclude any content that directly appears in or strongly implies the correct answer.  
        - Provide general information about the medical condition or imaging findings without revealing specific diagnostic conclusions present in the correct answer.
    3. **Focus on Educational Context**:  
        - Ensure the improved context provides medical background and clinical context that supports understanding of the question.
        - Include information about the imaging modality, anatomical location, and general clinical considerations.
    4. **Ensure Clarity and Conciseness**:  
        - Write the improved context in a clear, concise, and professional manner.
        - Maintain medical accuracy and educational value while avoiding data leakage.

    ### **Output Format**
    - **Improved Context**: A consolidated and comprehensive context for the multiple-choice question, excluding information that reveals the correct answer while maintaining sufficient background for clinical reasoning.

    ### **Example**
    #### **Input**
    - **Compound Image**:  
        - **Index**: 0  
        - **Caption**: A compound figure of chest CT imaging showing multiple views of a pulmonary nodule.
        - **Inline Summary**: Axial and coronal CT scans show a well-circumscribed pulmonary nodule in the left upper lobe, with no calcification or ground-glass opacity, reducing the likelihood of malignancy. Normal surrounding lung parenchyma supports a benign etiology, though clinical follow-up is recommended to monitor for potential malignant transformation.
        - **Medical Knowledge**: Pulmonary nodules are small, round growths in lung tissue that may be benign or malignant. Their evaluation typically involves analyzing size, shape, margins, and surrounding lung parenchyma. Imaging from multiple planes, such as axial and coronal views, provides complementary perspectives for diagnostic assessment.
        - **Visual Perception Description**: The compound figure contains multiple sub-images showing different CT views of a solitary pulmonary nodule in the left upper lobe from axial and coronal perspectives.
    - **Specific Sub-Image**:  
        - **Index**: Fig-1  
        - **Caption**: An axial CT image of the chest demonstrating a solitary pulmonary nodule in the left upper lobe.
        - **Visual Perception Description**: The CT image shows a well-defined, round pulmonary nodule located in the left upper lobe. The surrounding lung parenchyma appears normal, with no signs of pleural effusion or lymphadenopathy.
    - **Multiple Choice Question**:
        - **Question**: Based on Fig-1, which of the following features most strongly supports a benign diagnosis for the pulmonary nodule?
        - **Options**:  
        A. Presence of ground-glass opacity  
        B. Smooth, well-defined borders  
        C. Enlarged mediastinal lymph nodes  
        D. Irregular shape with spiculated margins
        - **Correct Answer**: B
    #### **Output**
    - **Improved Context**: This compound figure includes multiple CT views of a pulmonary nodule located in the left upper lobe. Pulmonary nodules are small, round growths in lung tissue that require careful evaluation to determine their nature. The assessment typically involves analyzing various imaging characteristics from different perspectives. Axial and coronal CT views provide complementary information about the nodule's morphology and its relationship to surrounding structures. Key features evaluated include the nodule's shape, margins, density characteristics, and the appearance of adjacent lung parenchyma. These imaging characteristics, combined with clinical context, help guide further diagnostic workup and management decisions.
    """
    return SYSTEM_PROMPT.strip()

SYSTEM_PROMPT = generate_system_prompt()


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
        "caption": entry.get("caption", ""),
        "inline summary": entry.get("Inline Summary", ""),
        "medical knowledge": entry.get("Medical Knowledge", ""),
        "visual perception description": entry.get("visual perception description", "")
    }

    # Get selected sub-image information
    selected_image_paths = qa_pair.get("selected_image_paths", [])
    image_indices = qa_pair.get("image_index", [])
    
    all_images = summarize_sample_images(entry)
    
    # Build sub-image information
    subimage_details = []
    for img_path, img_idx in zip(selected_image_paths, image_indices):
        matched_image = next((img for img in all_images if img["image"] == img_path), None)
        if matched_image:
            subimage_details.append({
                "index": img_idx,
                "caption": matched_image.get("text", ""),
                "visual_perception": matched_image.get("visual perception description", "")
            })

    # Build the user input prompt
    user_input = f"#### **Input**\n"
    user_input += f"- **Compound Image**:\n"
    user_input += f"    - **Index**: 0\n"
    user_input += f"    - **Caption**: {compound_info.get('caption')}\n"
    user_input += f"    - **Inline Summary**: {compound_info.get('inline summary')}\n"
    user_input += f"    - **Medical Knowledge**: {compound_info.get('medical knowledge')}\n"
    user_input += f"    - **Visual Perception Description**: {compound_info.get('visual perception description')}\n"
    
    if subimage_details:
        user_input += f"- **Specific Sub-Image**:\n"
        for sub_info in subimage_details:
            user_input += f"    - **Index**: {sub_info.get('index')}\n"
            user_input += f"    - **Caption**: {sub_info.get('caption')}\n"
            user_input += f"    - **Visual Perception Description**: {sub_info.get('visual_perception')}\n"
    
    user_input += f"- **Multiple Choice Question**:\n"
    user_input += f"    - **Question**: {qa_pair.get('question', '')}\n"
    user_input += f"    - **Options**: {qa_pair.get('option', '')}\n"
    user_input += f"    - **Correct Answer**: {qa_pair.get('answer', '')}"

    return user_input


# === Generate improved context ===
def clean_context(entry, qa_pair):
    user_input = build_structured_user_input(entry, qa_pair)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(output_text)
    
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()

    if "**Improved Context**" in output_text:
        output_text = output_text.split("**Improved Context**", 1)[1].strip()
        # Remove leading colons or other separators
        output_text = re.sub(r'^[:\s]+', '', output_text).strip()

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

    for entry in tqdm(data, desc="Generating Improved Contexts for Multiple Choice Questions"):
        qa_pairs = entry.get("qa_pairs", [])
        
        if not qa_pairs:
            print(f"[Warning] No qa_pairs found for entry idx={entry.get('idx', 'unknown')}")
            continue
        
        for qa in qa_pairs:
            try:
                improved_context = clean_context(entry, qa)
                qa["improved context"] = improved_context
            except Exception as e:
                print(f"[Warning] Failed to clean context for idx={entry.get('idx', 'unknown')}: {e}")

    # Save the modified dataset back to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Improved contexts generated and saved to: {output_file}")

