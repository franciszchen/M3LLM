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
from PIL import Image

# === Parse input arguments ===
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--classification_file", type=str, required=True, help="Path to the classification CSV file.")
args = parser.parse_args()

input_file = args.input_file
classification_csv = args.classification_file

# === Prepare output filename ===
output_dir = '<YOUR_OUTPUT_DIR>/PMC_Insturction_Data/stage4_boundingboxVQA_v1'
input_filename = os.path.basename(input_file)

# Match "random_chunk_000.json"
file_index = re.search(r'random_chunk_(\d+)\.json$', input_filename)
file_idx = file_index.group(1) if file_index else "unknown"

# Construct output file name using this index
output_file = os.path.join(output_dir, f"final_instruction_puretextQA_{file_idx}.json")


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
    You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to generate structured and concise **Question-Answer** outputs based on the relative positions of two selected sub-images from a compound medical image. I already pre-computed the relative position of two figures, please rephrase this positional information into a professional and medically accurate QA pair.

    ### **Task**
    #### **Data Description**  
    The input includes:  
    1. **Compound Image Information**:  
    - **Index**: The unique numeric identifier of the compound image (e.g., Fig-0).  
    - **Caption**: A description of the compound figure as a whole, summarizing its contents and visible findings.  
    2. **Selected Sub-Images**:  
    - **Indices**: The numeric identifiers of the two selected sub-images (e.g., Fig-1 and Fig-2).  
    3. **Specific Sub-Image Details**:  
    For each of the two selected sub-images:  
    - **Index**: The numeric identifier of the sub-image.  
    - **Caption**: A description of the visual content of the sub-image.  
    - **Visual Perception Description**: Observations of the sub-image’s visual features, including spatial orientation and key findings.  
    - **Bounding Box Information**: The coordinates of the central point of the sub-image bounding box in the compound image, provided as `(x,y)`. 
    -** Precomputed relative position**: The precomputed relative position of two selected figures.
    #### **Specific Task**  
    Generate a structured output consisting of:
    1. **Question**: A clear and specific question about the spatial relationship (literal physical placement) between the two selected sub-images.  
    2. **Answer**: A precise and accurate response describing the relative positions of the two selected sub-images. Please use the given center points of the given two subfigures.
    The outputs must focus **only on the spatial relationship between the two selected sub-images**.

    ### **Objective**  
    1. **Accuracy**:  
    - Ensure all outputs are accurate and based strictly on the provided references for the two selected sub-images.  
    2. **Clarity**:  
    - Write all outputs in a clear, concise, and professional manner.  
    3. **Relevance**:  
    - Focus on the spatial relationship between the two selected sub-images.  

    ### **Instructions**  
    1. **Use References**:  
    - Base outputs on the provided information for the two selected sub-images, ensuring the **Question** and **Answer** are supported by their details.  
    2. **Focus on Selected Sub-Images**:  
    - Ensure all outputs focus only on the two selected sub-images.  
    3. **Describe Relative Position**:  
    - please rephrase this positional information into a professional and medically accurate QA pair. No need to mention the reason, directly state the position.
    
    ### **Output Format**   
    - **Question**: A specific and clear question about the spatial relationship between the two selected sub-images.  
    - **Answer**: A precise and accurate response describing the relative positions of the two selected sub-images.

    ### **Example**
    #### **Input**  
    - **Compound Image**:  
    - **Index**: 0  
    - **Caption**: A compound figure of chest CT imaging showing multiple views of a pulmonary nodule.  
    - **Selected Sub-Images**:  
    - **Indices**: Fig-1, Fig-2  
    - **Specific Sub-Image Details**:  
    - **Index**: Fig-1  
        - **Caption**: An axial CT image of the chest demonstrating a solitary pulmonary nodule in the left upper lobe.  
        - **Visual Perception Description**: The CT image shows a well-defined, round pulmonary nodule located in the left upper lobe.  
        - **Bounding Box Information**: (100, 150)  
    - **Index**: Fig-2  
        - **Caption**: A coronal CT image showing the same pulmonary nodule in the left upper lobe from a different plane.  
        - **Visual Perception Description**: The coronal CT image demonstrates the nodule’s position relative to the mediastinum and surrounding lung structures, confirming its location in the left upper lobe.  
        - **Bounding Box Information**: (250, 100)  
    -** Precomputed relative position **: 
    The center of Fig-1 is (100, 150), and the center of Fig-2 is (250, 100). Therefore, Fig-1 is **to the left and below** Fig-2.

    #### **Output**  
    - **Question**: What is the spatial relationship between the Fig-1 and the Fig-2 of the pulmonary nodule in the compound image?  
    - **Answer**: Fig-1 is positioned to the left and below Fig-2 in the compound image.       
"""

    return SYSTEM_PROMPT.strip()

def load_bounding_boxes(compound_image_filename):
    root_dir = "<YOUR_BBOX_DIR>/subfig_bounding_boxes"
    subdirs = [f"predict{i}/labels" if i else "predict/labels" for i in range(0, 5)]
    bbox_filename = os.path.splitext(compound_image_filename)[0] + ".txt"
    print(bbox_filename)
    for subdir in subdirs:
        path = os.path.join(root_dir, subdir, bbox_filename)
        if os.path.isfile(path):
            with open(path, "r") as f:
                lines = f.readlines()
                bboxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, x_center, y_center, width, height = map(float, parts)
                    bboxes.append((x_center, y_center, width, height))
                return bboxes
    return []


def compute_relative_position(box1, box2, image_width, image_height, proportional_epsilon=0.02):
    """
    Compute relative spatial position of box1 to box2 with a dynamic epsilon
    based on image dimensions.

    Args:
        box1, box2: (x_center_pixel, y_center_pixel)
        image_width, image_height: dimensions of the compound image
        proportional_epsilon: float, tolerance as a proportion of image size
    """
    x1, y1 = box1
    x2, y2 = box2

    dx = x1 - x2
    dy = y1 - y2

    epsilon_x = proportional_epsilon * image_width
    epsilon_y = proportional_epsilon * image_height

    position = []

    if dx > epsilon_x:
        position.append("to the right")
    elif dx < -epsilon_x:
        position.append("to the left")

    if dy > epsilon_y:
        position.append("below")
    elif dy < -epsilon_y:
        position.append("above")

    if not position:
        return "at the same position"
    
    return " and ".join(position)




def yolo_to_xyxy(norm_box, image_width, image_height):
    x_c, y_c, w, h = norm_box
    x_min = int((x_c - w / 2) * image_width)
    y_min = int((y_c - h / 2) * image_height)
    x_max = int((x_c + w / 2) * image_width)
    y_max = int((y_c + h / 2) * image_height)
    return (x_min, y_min, x_max, y_max)

def yolo_to_center(norm_box, image_width, image_height):
    """
    Convert YOLO-format normalized bounding box to center point in pixel coordinates.

    Parameters:
        norm_box (tuple): (x_center_norm, y_center_norm, width_norm, height_norm)
        image_width (int): width of the image
        image_height (int): height of the image

    Returns:
        (x_center_pixel, y_center_pixel): tuple of center point in pixels
    """
    x_c_norm, y_c_norm, _, _ = norm_box
    x_c_pixel = int(x_c_norm * image_width)
    y_c_pixel = int(y_c_norm * image_height)
    return (x_c_pixel, y_c_pixel)

def extract_subfig_index(subfig_filename):
    try:
        return int(subfig_filename.split(".jpg_")[-1].split(".")[0])
    except:
        return None

def construct_user_prompt(sample, selected_subfigs):
    compound_caption = sample.get("caption", "").strip()
    medical_knowledge = sample.get("Medical_Knowledge", "").strip()
    inline_summary = sample.get("references", "")

    def get_idx(sf):
        raw_idx = sf.get("idx", sf.get("sub_idx", "unknown"))
        try:
            return int(raw_idx) + 1
        except:
            return raw_idx

    # === Relative position description ===
    if "bounding_box" in selected_subfigs[0] and "bounding_box" in selected_subfigs[1]:
        pos_info = compute_relative_position(
            selected_subfigs[0]["bounding_box"],
            selected_subfigs[1]["bounding_box"],
            image_width=width,
            image_height=height,
            proportional_epsilon=0.02  # or another value
        )
        idx0 = get_idx(selected_subfigs[0])
        idx1 = get_idx(selected_subfigs[1])
        rel_statement = (
            f"The center of Fig-{idx0} is {selected_subfigs[0]['bounding_box']}, "
            f"and the center of Fig-{idx1} is {selected_subfigs[1]['bounding_box']}. "
            f"Therefore, Fig-{idx0} is {pos_info} of Fig-{idx1}."
        )
    else:
        rel_statement = "The bounding box information is incomplete."

    lines = [
        "#### **Input**",
        "- **Compound Image**:",
        "- **Index**: 0",
        f"- **Caption**: {compound_caption}",
        # f"- **Inline Summary**: {inline_summary}",
        # f"- **Medical Knowledge**: {medical_knowledge}",
        "",
        "- **Specific Sub-Image**:"
    ]

    for subfig in selected_subfigs:
        idx = get_idx(subfig)
        lines.extend([
            f"- **Index**: Fig-{idx}",
            f"    - **Caption**: {subfig.get('text', '').strip()}",
            f"    - **Visual Perception Description**: {subfig.get('visual perception description', '').strip()}",
            f"    - **Bounding Box Information**: {subfig.get('bounding_box', '(unknown)')}",
            ""
        ])

    lines.append(f"- **Precomputed Relative Position**: {rel_statement}")
    lines.append("")
    lines.append("Please rephrase this positional information into a professional and medically accurate QA pair.")

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
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()
    #print(output_text)
    #context_match = re.search(r'- \*\*Context\*\*:\s*(.*?)(?=\n- \*\*Question\*\*:)', output_text, re.DOTALL)
    question_match = re.search(r'- \*\*Question\*\*:\s*(.*?)(?=\n- \*\*Answer\*\*:)', output_text, re.DOTALL)
    answer_match = re.search(r'- \*\*Answer\*\*:\s*(.*)', output_text, re.DOTALL)

    #context = context_match.group(1).strip() if context_match else ""
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    return {
        #"Context": context,
        "Question": question,
        "Answer": answer,
        "raw_output": output_text
    }

# === Load input data ===
with open(input_file, "r") as f:
    data = json.load(f)

results = []
for sample in tqdm(data):
    subfigs = sample.get("subcaptions", [])
    if len(subfigs) < 2:
        continue

    selected_indices = random.sample(range(len(subfigs)), 2)
    compound_img = sample.get("image", "")
    bbox_list = load_bounding_boxes(compound_img)

    # Get compound image size
    width, height = 1000, 1000  # fallback
    for group in range(1, 17):
        candidate = os.path.join(f"<YOUR_IMAGE_DIR>/mm_image_files/compound/group_{group}", compound_img)
        if os.path.exists(candidate):
            with Image.open(candidate) as img:
                width, height = img.size
            break

    selected = []
    for idx in selected_indices:
        subfig = subfigs[idx]
        subfig_index = extract_subfig_index(subfig["image"])
        if subfig_index is not None and subfig_index < len(bbox_list):
            abs_bbox = yolo_to_center(bbox_list[subfig_index], width, height)
            subfig["bounding_box"] = abs_bbox
        else:
            subfig["bounding_box"] = "(unknown)"
        selected.append(subfig)

    qa_result = generate_instruction_for_qa(sample, selected)
    qa_pairs = [{
        "question": qa_result["Question"],
        "answer": qa_result["Answer"],
        "selected_image_paths": [s["image"] for s in selected],
        "image_index": [f"Fig-{s.get('idx', '?')}" for s in selected],
        "raw_output": qa_result["raw_output"],
        "grouped_visual_word_count": len(qa_result["Answer"].split())
    }]

    sample["qa_pairs"] = qa_pairs
    results.append(sample)



# === Save result ===
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to: {output_file}")
