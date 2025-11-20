import json
import re

input_path = "/Users/yihang/Desktop/PMC-codebase/benchmark/benchmark_data/multiplechoiceVQA_benchmark.json"
output_path = "/Users/yihang/Desktop/PMC-codebase/benchmark/benchmark_data/multiplechoiceVQA_benchmark_new.json"

def transform_question(q: str) -> str:
    """
    Convert:
      "Question text...\nA. option1\nB. option2..."
    to:
      "The question is Question text... The candidate options are A. option1\nB. option2..."
    """
    # Find the first occurrence of a newline followed by "A." or "A)"
    m = re.search(r"\nA[.)]", q)
    if not m:
        # If we can't find options, just return the original string
        return q

    split_idx = m.start()  # index of '\n' before A
    question_part = q[:split_idx].strip()
    options_part = q[split_idx + 1:].lstrip()  # skip the newline, keep "A."

    return f"The question is {question_part} The candidate options are: {options_part}"

# Load original JSON
with open(input_path, "r") as f:
    data = json.load(f)

# Transform each sample
for sample in data:
    if "question" in sample and isinstance(sample["question"], str):
        sample["question"] = transform_question(sample["question"])

# Save new JSON
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved converted file to {output_path}")
