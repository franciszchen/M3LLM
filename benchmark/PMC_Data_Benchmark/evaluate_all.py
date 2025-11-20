# evaluate_all.py
import os
import re
import json
import glob
import argparse
from tqdm import tqdm
from PIL import Image
import time
import json, gzip
from model_wrapper.vlm_manager import get_model
from task_config import TASKS
from eval_utils import (
    calculate_bleu_scores,
    calculate_rouge,
    calculate_em,
    calculate_f1,
    judge_multi_choice,
    parse_options_from_prompt,
    calculate_bertscore,
    calculate_bertsimilarity
)
from cider import Cider

from llm_as_judge import AzureGPT

COMPOUND_FOLDER_ROOT = "/gpfs/radev/home/hk832/pi/mm_image_files/compound"
SUBFIGURE_FOLDER = "/gpfs/radev/home/hk832/scratch/pmc_data/mm_image_files/subfigures"
RESULTS_DIR = "results"

# === Image Finders ===
def find_compound_image_path(filename):
    matches = glob.glob(os.path.join(COMPOUND_FOLDER_ROOT, "group_*", filename))
    if not matches:
        print(f"[Warning] Compound image not found: {filename}")
        return None
    return matches[0]

def find_subfigure_image_path(filename):
    matches = glob.glob(os.path.join(SUBFIGURE_FOLDER, "*", "subfigures", filename))
    if not matches:
        print(f"[Warning] Subfigure image not found: {filename}")
        return None
    return matches[0]

# === Base Args Class ===
class Args:
    temperature = 0
    top_p = 1.0
    repetition_penalty = 1.0
    max_new_tokens = 512

def load_samples(path):
    """Load a dataset from .json, .json.gz, .jsonl, or .jsonl.gz"""
    is_jsonl = path.endswith(".jsonl")
    opener = gzip.open if path.endswith(".gz") else open

    samples = []
    with opener(path, "rt", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                samples.append(json.loads(line))
        else:
            data = json.load(f)
            # Normalize to list
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                # common patterns
                samples = data.get("data") or data.get("samples") or [data]
            else:
                raise ValueError("Unsupported JSON structure")
    return samples


def evaluate(model_name, model_path, task_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if task_name not in TASKS:
        raise ValueError(f"Task '{task_name}' not found in task_config")

    task_cfg = TASKS[task_name]
    samples = load_samples(task_cfg["json_path"])

    model = get_model(model_name, model_path, Args())
    results: List[Dict[str, Any]] = []

    is_multi_choice = task_name.strip().lower() in {"multi-choice", "multiple-choice", "multichoice"}

    # Counters / aggregators
    mc_total = 0
    mc_correct_total = 0

    metric_keys = [
        "bleu1", "bleu2", "bleu3", "bleu4",
        "rouge-1", "rouge-2", "rouge-l",
        "em", "f1", "precision", "recall",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "bertsim", "cider"
    ]
    agg_sums = {k: 0.0 for k in metric_keys}
    agg_count = 0

    # For CIDEr (corpus-level)
    gts_cider: Dict[str, List[str]] = {}
    res_cider: Dict[str, List[str]] = {}

    for idx, sample in enumerate(tqdm(samples, desc=f"{task_name}-{model_name}")):
        question = sample.get("question", "")
        reference_answer = sample.get("gt", "")
        image_index = sample.get("image_indices", "")
        sid = str(sample.get("id", idx))  # stable id for mapping metrics

        # Load images (if any)
        image_filenames = sample.get("images", [])
        loaded_images = []
        for fn in image_filenames:
            full_path = os.path.join(task_cfg["image_root"], fn)
            if os.path.exists(full_path):
                try:
                    loaded_images.append(Image.open(full_path).convert("RGB"))
                except Exception as e:
                    print(f"[Warning] Failed to open {full_path}: {e}")
            else:
                print(f"[Warning] Missing image file: {full_path}")

        # Build messages for your model
        messages = {
            "system": task_cfg.get("system_prompt", ""),
            "prompt": question,
            "images": loaded_images,
            "image_indices": image_index
        }

        # Generate
        prediction = model.generate_output(messages)

        # Evaluate
        if is_multi_choice:
            choices = parse_options_from_prompt(question)
            if not choices:
                print(f"[Warn] No options parsed for id={sid}. Ensure A./B./C. appear in the prompt.")
                mc_correct = 0
            else:
                mc_correct = int(judge_multi_choice(choices, reference_answer, prediction))

            mc_total += 1
            mc_correct_total += mc_correct

            metrics = {"mc_correct": mc_correct}

        else:
            bleu = calculate_bleu_scores(reference_answer, prediction)
            rouge = calculate_rouge(reference_answer, prediction)
            em = calculate_em(prediction, reference_answer)
            f1, precision, recall = calculate_f1(prediction, reference_answer)
            bert = calculate_bertscore(prediction, reference_answer)
            bertsim = calculate_bertsimilarity(reference_answer, prediction)

            metrics = {
                **bleu,
                **rouge,
                "em": em,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                **bert,
                "bertsim": float(bertsim) if isinstance(bertsim, (int, float)) else json_safe(bertsim),
                # 'cider' will be filled after corpus computation
            }

            # accumulate for macro averages (skip cider for now)
            for k in metric_keys:
                if k == "cider":
                    continue
                if k in metrics and isinstance(metrics[k], (int, float)):
                    agg_sums[k] += float(metrics[k])
            agg_count += 1

            # Collect for corpus CIDEr
            gts_cider[sid] = [reference_answer]
            res_cider[sid] = [prediction]

        # Store per-sample record
        results.append({
            "id": sid,
            "task": task_name,
            "model": model_name,
            "question": question,
            "reference_answer": reference_answer,
            "prediction": prediction,
            "metrics": metrics
        })

    # --- Compute CIDEr (corpus) and attach per-item scores ---
    cider_corpus = 0.0
    if not is_multi_choice and gts_cider:
        try:
            cider = Cider(n=4, sigma=6.0)
            cider_corpus, cider_per_list = cider.compute_score(gts_cider, res_cider)
            sid_list = list(gts_cider.keys())  # insertion order preserved
            sid_to_cider = {sid: float(sc) for sid, sc in zip(sid_list, cider_per_list)}
            # write per-item + aggregate
            for rec in results:
                sid = rec.get("id")
                if sid in sid_to_cider and "metrics" in rec:
                    rec["metrics"]["cider"] = sid_to_cider[sid]
                    agg_sums["cider"] = agg_sums.get("cider", 0.0) + sid_to_cider[sid]
        except AssertionError as e:
            print(f"[WARN] CIDEr skipped due to input mismatch: {e}")
            cider_corpus = 0.0

    # Save per-sample results
    output_path = os.path.join(RESULTS_DIR, f"{task_name}_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_path}")

    # Save overall summary
    if is_multi_choice:
        mc_acc = (mc_correct_total / mc_total) if mc_total else 0.0
        mc_acc_pct = 100.0 * mc_acc
        print(f"[Multi-choice] Accuracy: {mc_correct_total}/{mc_total} = {mc_acc_pct:.2f}%")

        summary = {
            "task": task_name,
            "model": model_name,
            "num_samples": mc_total,
            "num_correct": mc_correct_total,
            "accuracy": mc_acc,              # 0-1
            "accuracy_pct": mc_acc_pct       # 0-100
        }
    else:
        averages = {k: (agg_sums[k] / agg_count if agg_count else 0.0) for k in metric_keys if k in agg_sums}
        # Prefer the standard corpus CIDEr in the averages (like COCO eval)
        averages["cider"] = float(cider_corpus)
        summary = {
            "task": task_name,
            "model": model_name,
            "num_samples": agg_count,
            "macro_avg": averages,
            "corpus": {"cider": float(cider_corpus)}
        }

    summary_path = os.path.join(RESULTS_DIR, f"{task_name}_{model_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation for a given model and task.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name key for inference wrapper")
    parser.add_argument("--model_path", type=str, required=True, help="Path to load the model from")
    parser.add_argument("--task", type=str, required=True, help="Task name defined in task_config")

    args = parser.parse_args()
    evaluate(args.model_name, args.model_path, args.task)