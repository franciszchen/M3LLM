#!/usr/bin/env python3
import os
import json
import gzip
import argparse
from typing import Any, Dict, List
from tqdm import tqdm

# ---- your existing metric utilities ----
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
from cider import Cider  # corpus-level CIDEr

RESULTS_DIR = "results"

# ---------- IO helpers ----------
def load_samples(path: str) -> List[Dict[str, Any]]:
    """Load .json / .json.gz / .jsonl / .jsonl.gz into a list of dicts."""
    is_jsonl = path.endswith(".jsonl") or path.endswith(".jsonl.gz")
    opener = gzip.open if path.endswith(".gz") else open
    samples: List[Dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                samples.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                samples = data.get("data") or data.get("samples") or [data]
            else:
                raise ValueError("Unsupported JSON structure")
    return samples

def json_safe(x: Any) -> Any:
    """Make values JSON-serializable (e.g., numpy/tensors)."""
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    if isinstance(x, (set,)):
        return list(x)
    return x

# ---------- main evaluation ----------
def evaluate_from_predictions(
    predictions_json: str,
    model_name: str = "azure_gpt4o_cached",
    task_name: str = "open-ended",
    output_dir: str = RESULTS_DIR,
):
    os.makedirs(output_dir, exist_ok=True)

    is_multi_choice = task_name.strip().lower() in {"multi-choice", "multiple-choice", "multichoice"}

    samples = load_samples(predictions_json)

    metric_keys = [
        "bleu1", "bleu2", "bleu3", "bleu4",
        "rouge-1", "rouge-2", "rouge-l",
        "em", "f1", "precision", "recall",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "bertsim", "cider"
    ]
    agg_sums = {k: 0.0 for k in metric_keys}
    agg_count = 0

    # for multi-choice accuracy
    mc_total = 0
    mc_correct_total = 0

    # for corpus CIDEr
    gts_cider: Dict[str, List[str]] = {}
    res_cider: Dict[str, List[str]] = {}

    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(tqdm(samples, desc=f"eval-{task_name}-{model_name}")):
        sid = str(sample.get("id", idx))
        question = sample.get("question", "")
        reference_answer = sample.get("gt", sample.get("reference_answer", ""))
        prediction = sample.get("prediction", sample.get("pred", "")) or ""

        if is_multi_choice:
            choices = parse_options_from_prompt(question)
            if not choices:
                mc_correct = 0
            else:
                mc_correct = int(judge_multi_choice(choices, reference_answer, prediction))
            mc_total += 1
            mc_correct_total += mc_correct
            metrics = {"mc_correct": mc_correct}

        else:
            # token/string-based metrics
            bleu = calculate_bleu_scores(reference_answer, prediction)
            rouge = calculate_rouge(reference_answer, prediction)
            em = calculate_em(prediction, reference_answer)
            f1, precision, recall = calculate_f1(prediction, reference_answer)

            # semantic metrics
            bert = calculate_bertscore(prediction, reference_answer)  # returns p/r/f1
            bertsim = calculate_bertsimilarity(reference_answer, prediction)

            metrics = {
                **bleu,
                **rouge,
                "em": em,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                **bert,                      # bertscore_p/r/f1
                "bertsim": float(bertsim) if isinstance(bertsim, (int, float)) else json_safe(bertsim),
                # 'cider' filled after corpus computation
            }

            # accumulate macro (skip 'cider' here; we add corpus CIDEr later)
            for k in metric_keys:
                if k == "cider":
                    continue
                if k in metrics and isinstance(metrics[k], (int, float)):
                    agg_sums[k] += float(metrics[k])
            agg_count += 1

            # collect for corpus CIDEr
            gts_cider[sid] = [reference_answer]
            res_cider[sid] = [prediction]

        results.append({
            "id": sid,
            "task": task_name,
            "model": model_name,
            "question": question,
            "reference_answer": reference_answer,
            "prediction": prediction,
            "metrics": metrics
        })

    # compute CIDEr corpus-level, attach per-item CIDEr
    cider_corpus = 0.0
    if not is_multi_choice and gts_cider:
        try:
            cider = Cider(n=4, sigma=6.0)
            cider_corpus, cider_per_list = cider.compute_score(gts_cider, res_cider)
            sid_list = list(gts_cider.keys())
            sid_to_cider = {sid: float(sc) for sid, sc in zip(sid_list, cider_per_list)}
            for rec in results:
                sid = rec.get("id")
                if sid in sid_to_cider:
                    rec["metrics"]["cider"] = sid_to_cider[sid]
                    agg_sums["cider"] = agg_sums.get("cider", 0.0) + sid_to_cider[sid]
        except AssertionError as e:
            print(f"[WARN] CIDEr skipped: {e}")
            cider_corpus = 0.0

    # write per-sample results
    base = f"{task_name}_{model_name}_frompreds"
    out_items = os.path.join(output_dir, base + ".json")
    with open(out_items, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved per-sample results → {out_items}")

    # write summary
    if is_multi_choice:
        mc_acc = (mc_correct_total / mc_total) if mc_total else 0.0
        mc_acc_pct = 100.0 * mc_acc
        summary = {
            "task": task_name,
            "model": model_name,
            "num_samples": mc_total,
            "num_correct": mc_correct_total,
            "accuracy": mc_acc,
            "accuracy_pct": mc_acc_pct
        }
    else:
        macro_avg = {}
        if agg_count:
            for k, v in agg_sums.items():
                if k == "cider":
                    continue  # report corpus CIDEr separately
                macro_avg[k] = v / agg_count
        summary = {
            "task": task_name,
            "model": model_name,
            "num_samples": agg_count,
            "macro_avg": macro_avg,
            "corpus": {"cider": float(cider_corpus)}
        }

    out_summary = os.path.join(output_dir, base + "_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary → {out_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate from an existing predictions JSON (with gt & prediction).")
    parser.add_argument(
        "--predictions_json",
        type=str,
        required=True,
        help="Path to JSON/JSONL (optionally .gz) that contains fields: question, gt, prediction."
    )
    parser.add_argument("--task", type=str, default="open-ended", help="open-ended | multi-choice")
    parser.add_argument("--model_name", type=str, default="azure_gpt4o_cached")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    evaluate_from_predictions(
        predictions_json=args.predictions_json,
        model_name=args.model_name,
        task_name=args.task,
        output_dir=args.output_dir,
    )
