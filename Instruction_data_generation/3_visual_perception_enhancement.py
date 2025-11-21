import json
import os
import re
from cli import HuatuoChatbot
from tqdm import tqdm
import argparse
import time
from PIL import Image
import hashlib
import glob

# Define HuatuoGPT-Vision model path
#HUATUO_MODEL_PATH = "FreedomIntelligence/HuatuoGPT-Vision-7B"
HUATUO_MODEL_PATH = 'FreedomIntelligence/HuatuoGPT-Vision-34B'
# Initialize the chatbot
bot = HuatuoChatbot(HUATUO_MODEL_PATH)

# Argument Parser for Input File
parser = argparse.ArgumentParser(description="Process a JSON file with a given input file name.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
args = parser.parse_args()

input_file = args.input_file
output_dir = '<YOUR_OUTPUT_DIR>/PMC_instruction_data/stage1_3'
input_filename = os.path.basename(input_file)

# Updated regex to handle optional 'zc_' prefix
file_index_match = re.match(r'background_info_((?:zc_)?\d+_\d+)\.json$', input_filename)
file_idx = file_index_match.group(1) if file_index_match else "unknown"

output_file = os.path.join(output_dir, f"visual_component_{file_idx}.json")


def hash_image(image_path):
    try:
        with Image.open(image_path) as img:
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception as e:
        print(f"[Error loading image] {image_path}: {e}")
        return "ERROR"

# System Prompt
SYSTEM_PROMPT = """
### Role
You are a biomedical AI assistant specializing in medical imaging and clinical notes. Your task is to generate concise and detailed descriptions of medical images for clinical and AI training purposes.

### Task
#### **Input Description**
The input includes:  
1. **Image Captions**: Original and rewritten captions describing the medical image.  
2. Corresponding Image
#### **Specific Task**
Generate a professional description of the image, focusing on visible findings, their clinical relevance, and additional insights based on visual perception.

### Objective
1. **Accuracy**:  
   - Ensure all details are medically accurate and based on visible features or provided references.
2. **Conciseness**:  
   - Limit the description to 5 sentences, focusing on the most relevant and visually significant findings.
3. **Professionalism**:  
   - Use formal medical language and maintain scientific rigor.
4. **Visual Focus**:  
   - Provide insights grounded in the image's visual features, avoiding redundancy with the captions or medical knowledge.

### Instructions
1. **Use References**:  
   - Use the captions and medical knowledge for context, but focus on describing what is visually observable in the image.  
   - Avoid repeating or paraphrasing the provided references.
2. **Focus on Visible Details**:  
   - Highlight significant findings visible in the image, such as anatomical features, abnormalities, or patterns.  
   - Provide additional insights where possible.
3. **Highlight Clinical Relevance**:  
   - Emphasize visually derived details that are clinically meaningful and add diagnostic value.
4. **Avoid Hallucination**:  
   - Do not infer or assume details beyond what is visible in the image or explicitly supported by the references.

### Output Format
- **Visual Perception Description**: A concise and professional description of the medical image, limited to 5 sentences, emphasizing visual findings and clinical relevance.

### Example
#### **Input**
- **Original Caption**: CT scan of the chest showing a solitary pulmonary nodule in the right upper lobe.   
- Image Input
#### **Output**
- **Visual Perception Description**: The axial chest CT shows a solitary, well-defined pulmonary nodule in the right upper lobe with smooth margins and no visible calcifications. The nodule measures approximately 1.2 cm and is located adjacent to the pleural surface. No surrounding ground-glass opacity, lymphadenopathy, or pleural effusion is observed. The proximity of the nodule to the pleura may warrant further evaluation for any subtle pleural involvement. These features suggest a likely benign etiology, though clinical correlation is recommended.
"""

# Define folders containing images
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

# Load dataset
with open(input_file, "r") as f:
    data = json.load(f)

def construct_user_prompt(entry, fallback_entry=None):
    caption = entry.get("caption", entry.get("text", ""))
    
    if fallback_entry is None:
        keywords = entry.get("keywords", "")
        medical_knowledge = entry.get("medical knowledge", "")
    else:
        keywords = fallback_entry.get("keywords", "")
        medical_knowledge = fallback_entry.get("medical knowledge", "")
    
    user_input = (
        f"#### **Input**\n"
        f"**Original Caption**: {caption}\n"
        #f"**Keywords**: {keywords}\n"
        #f"**Medical Knowledge**: {medical_knowledge}"
    )
    
    return user_input



def extract_context(output_text):

    if not isinstance(output_text, str):
        return ""

    match = re.search(
        r"-?\s*\*{0,2}Visual Perception Description\*{0,2}[:：]?\s*(.*)",
        output_text,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()
    else:
        return output_text.strip()



# Function to generate background information
def generate_instruction(image_path, user_prompt):
    print(f"\n>>> Loading image: {image_path}")
    print("Image hash:", hash_image(image_path))  # Add this line

    query = f"{SYSTEM_PROMPT}\n\nUser Query: {user_prompt}"
    #query = 'Please describe this medical image'
    print('--------------Input:-----------')
    #print(query)
    #print(user_prompt)
    output_text = bot.inference(query, [image_path])
    if isinstance(output_text, list):
        output_text = output_text[0]

    output_text = output_text.strip()
    #print('-----------Output:------------')
    print(output_text)
    context = extract_context(output_text)
    return context

# Process each image-caption pair
for entry in tqdm(data):
    compound_image_path = find_compound_image_path(entry["image"])
    user_prompt = construct_user_prompt(entry)
    
    # Generate background info for the compound figure
    entry["visual perception description"] = generate_instruction(compound_image_path, user_prompt)

    # Generate descriptions for subfigures (if available)
    if "subcaptions" in entry and entry["subcaptions"]:
      for sub in entry.get("subcaptions", []):
         sub_user_prompt = construct_user_prompt(sub, fallback_entry=entry)
         subfigure_image_path = find_subfigure_image_path(sub["image"])
         #subfigure_image_path = os.path.join(SUBFIGURE_FOLDER, sub["image"])
         sub["visual perception description"] = generate_instruction(subfigure_image_path, sub_user_prompt)



with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Descriptions generated and saved to: {output_dir}")