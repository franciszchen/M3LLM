import json
import base64
from pathlib import Path
from openai import AzureOpenAI

# ==== Azure OpenAI Configuration ====
client = AzureOpenAI(
    api_key="apikey",
    api_version="apiversion",
    azure_endpoint="https://xxxx.openai.azure.com",
    azure_deployment="gpt-4o"
)

# ==== File paths ====
json_path = Path("yourpath/single_subimageVQA_benchmark.json")
image_folder = Path("./benchmark_images")
output_path = './gpt_answer/singleSubimageVQA_with_predictions.json'

# ==== Utility: Convert local image to base64 ====
def encode_image_to_base64(image_path: Path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ==== Load JSON ====
with open(json_path, "r") as f:
    data = json.load(f)

results = []

# ==== Iterate through samples ====
for i, sample in enumerate(data):
    question = sample["question"]
    images = sample["images"]
    image_indices = sample["image_indices"]

    image_pairs = []
    for idx, img_name in zip(image_indices, images):
        img_path = image_folder / img_name
        if not img_path.exists():
            print(f"⚠️ Missing image: {img_path}")
            continue

        b64_image = encode_image_to_base64(img_path)
        image_pairs.append({"type": "text", "text": f"{idx}:"})
        image_pairs.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
        })

    # ==== Construct prompt ====
    prompt_text = (
        "You are a medical expert who is good at single subimage VQA task. Please answer the question regarding a single sub-image. Please do not answer with a single word."
        f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical imaging assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}] + image_pairs}
    ]

    # ==== Run inference ====
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=512,
        )
        answer = response.choices[0].message.content
        print(answer)
        print(f"[{i+1}/{len(data)}] ✅ Done")

    except Exception as e:
        print(f"[{i+1}/{len(data)}] ❌ Error: {e}")
        answer = None

    # ==== Store result ====
    result_entry = {
        "question": question,
        "gt": sample.get("gt", ""),
        "images": images,
        "image_indices": image_indices,
        "prediction": answer
    }
    results.append(result_entry)

# ==== Save to output JSON ====
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ All results saved to: {output_path}")