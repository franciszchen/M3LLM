#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified MedGemma inference script for MIMIC-CXR tasks.
Supports both:
  - Binary disease classification (Yes/No)
  - Disease progression (worsening/stable/improving)

Usage:
  python medgemma_inference.py \
    --model_path google/medgemma-2b \
    --input_jsonl ../data/disease_binary.jsonl \
    --image_root /path/to/MIMIC-CXR-JPG-sorted \
    --output_dir ../outputs/medgemma_binary \
    --task_type auto
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

# Disable torch.compile/dynamo for stability
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.pop("TORCH_COMPILE", None)
os.environ.pop("TORCH_LOGS", None)


# ============================================================================
# Text Processing Helpers
# ============================================================================

def strip_image_tokens(text: str) -> str:
    """Remove <image> tokens and Fig-X: markers from prompt text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"Fig-\d+\s*:\s*<image>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<image(?:_\d+)?>", "", text, flags=re.IGNORECASE)
    t = text.strip()
    if t.lower().startswith("question:"):
        t = t[len("question:"):].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        t = t[1:-1]
    return t.strip()


def first_human_turn(convs: List[Dict[str, Any]]) -> str:
    """Extract the first human message from conversations list."""
    if not isinstance(convs, list):
        return ""
    for turn in convs:
        if isinstance(turn, dict) and turn.get("from") == "human":
            return str(turn.get("value") or "").strip()
    return ""


# ============================================================================
# Answer Normalization
# ============================================================================

def normalize_yesno(text: str) -> Optional[str]:
    """Return 'Yes' or 'No' if found, else None."""
    if not isinstance(text, str):
        return None
    s = text.strip().lower()
    y = re.search(r"\b(yes|y|true|present|positive|\+|1)\b", s)
    n = re.search(r"\b(no|n|false|absent|negative|\-|0)\b", s)
    if y and (not n or y.start() < n.start()):
        return "Yes"
    if n:
        return "No"
    return None


def normalize_progression(text: str) -> Optional[str]:
    """Return 'worsening', 'stable', or 'improving' if found, else None."""
    if not isinstance(text, str):
        return None
    s = text.strip().lower()
    w = re.search(r"\b(worsen(?:ing|ed)?|worse|deteriorat(?:ing|ed)?)\b", s)
    st = re.search(r"\b(stable|unchanged|no change|same)\b", s)
    i = re.search(r"\b(improv(?:ing|ed)?|better)\b", s)
    
    matches = []
    if w:
        matches.append((w.start(), "worsening"))
    if st:
        matches.append((st.start(), "stable"))
    if i:
        matches.append((i.start(), "improving"))
    
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]
    return None


# ============================================================================
# Image Resolution
# ============================================================================

def resolve_image_paths(image_field, image_root: str) -> List[str]:
    """Resolve image paths from JSONL field."""
    names: List[str] = []
    if isinstance(image_field, (list, tuple)):
        names = [str(x) for x in image_field]
    elif isinstance(image_field, str):
        names = [s.strip() for s in image_field.split("||")] if "||" in image_field else [image_field]
    
    out = []
    for n in names:
        if not n:
            continue
        if os.path.isabs(n) and os.path.exists(n):
            out.append(os.path.normpath(n))
        elif os.path.exists(os.path.join(image_root, os.path.basename(n))):
            out.append(os.path.normpath(os.path.join(image_root, os.path.basename(n))))
        elif os.path.exists(os.path.join(image_root, n)):
            out.append(os.path.normpath(os.path.join(image_root, n)))
    return out


def pil_open_rgb(path: str) -> Image.Image:
    """Open image and convert to RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(df: pd.DataFrame, task_type: str) -> Dict[str, Any]:
    """Compute accuracy metrics overall and per-disease."""
    out: Dict[str, Any] = {}
    if df is None or df.empty:
        return {"samples": 0, "accuracy": 0.0, "per_disease": {}}
    
    valid = df.dropna(subset=["pred_norm", "gt_norm"]).copy()
    if valid.empty:
        return {"samples": 0, "accuracy": 0.0, "per_disease": {}}
    
    valid["correct"] = (valid["pred_norm"] == valid["gt_norm"]).astype(int)
    out["samples"] = int(len(valid))
    out["accuracy"] = float(valid["correct"].mean())
    
    if "disease" not in valid.columns or valid["disease"].isna().all():
        if valid["id"].astype(str).str.contains("::").any():
            valid["disease"] = valid["id"].astype(str).str.split("::").str[-1]
    
    per: Dict[str, Dict[str, float]] = {}
    if "disease" in valid.columns and not valid["disease"].isna().all():
        g = valid.groupby("disease")["correct"].agg(["count", "mean"]).reset_index()
        for _, r in g.iterrows():
            per[str(r["disease"])] = {"n": int(r["count"]), "acc": float(r["mean"])}
    out["per_disease"] = per
    return out


# ============================================================================
# MedGemma Message Builder
# ============================================================================

def build_messages_medgemma(prompt_text: str, image_paths: List[str], system_prompt: Optional[str]):
    """
    Build MedGemma chat-style messages.
    Format: [{"role":"user","content":[{"type":"image","image": PIL}, ..., {"type":"text","text": prompt}]}]
    """
    user_content: List[Dict[str, Any]] = []
    for p in image_paths:
        user_content.append({"type": "image", "image": pil_open_rgb(p)})
    user_content.append({"type": "text", "text": prompt_text})
    
    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({"role": "user", "content": user_content})
    return msgs


# ============================================================================
# Main Inference Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified MedGemma inference for MIMIC-CXR")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model path (e.g., google/medgemma-2b)")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--task_type", type=str, default="auto",
                        choices=["auto", "binary", "progression"],
                        help="Task type: auto-detect, binary (Yes/No), or progression")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--system_prompt", type=str, default="",
                        help="Optional system prompt to prepend")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load model
    print(f"Loading MedGemma model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        cache_dir=os.environ.get("HF_HOME")
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=os.environ.get("HF_HOME")
    )
    model.eval()
    print("Model loaded successfully!")
    
    # Load JSONL data
    print(f"Loading data from {args.input_jsonl}...")
    samples = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] Failed to parse line: {e}")
    
    # Auto-detect task type
    detected_task = args.task_type
    if detected_task == "auto" and samples:
        meta = samples[0].get("meta", {})
        task_meta = meta.get("task", "")
        if "binary" in task_meta or "presence" in task_meta:
            detected_task = "binary"
        elif "progression" in task_meta:
            detected_task = "progression"
        else:
            ans = samples[0].get("answer", "").strip().lower()
            if ans in ["yes", "no"]:
                detected_task = "binary"
            elif ans in ["worsening", "stable", "improving"]:
                detected_task = "progression"
            else:
                detected_task = "binary"
    
    print(f"Task type: {detected_task}")
    print(f"Total samples: {len(samples)}")
    
    # Inference loop
    results = []
    skipped = 0
    processed = 0
    
    out_jsonl = os.path.join(args.output_dir, "predictions.jsonl")
    out_csv = os.path.join(args.output_dir, "predictions.csv")
    
    with open(out_jsonl, "w", encoding="utf-8") as fj:
        for sample in tqdm(samples, desc="Processing"):
            if args.limit and processed >= args.limit:
                break
            
            sid = sample.get("id", processed)
            img_paths = resolve_image_paths(sample.get("image", []), args.image_root)
            
            if not img_paths:
                skipped += 1
                continue
            
            # Get prompt
            convs = sample.get("conversations", [])
            raw_prompt = first_human_turn(convs)
            if not raw_prompt:
                skipped += 1
                continue
            
            prompt = strip_image_tokens(raw_prompt)
            
            # Build messages
            messages = build_messages_medgemma(prompt, img_paths, args.system_prompt)
            
            # Process and generate
            try:
                text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=text_prompt, images=[msg["content"] for msg in messages if msg["role"] == "user"][0], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature if args.temperature > 0 else None,
                        top_p=args.top_p if args.temperature > 0 else None,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=(args.temperature > 0),
                    )
                
                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                
            except Exception as e:
                print(f"[ERROR] Generation failed for id={sid}: {e}")
                skipped += 1
                continue
            
            # Normalize answers
            gt_raw = sample.get("answer", "")
            if detected_task == "binary":
                pred_norm = normalize_yesno(response)
                gt_norm = normalize_yesno(gt_raw)
            else:
                pred_norm = normalize_progression(response)
                gt_norm = normalize_progression(gt_raw)
            
            # Extract disease
            disease = None
            meta = sample.get("meta", {})
            if isinstance(meta, dict):
                disease = meta.get("disease")
            if not disease and isinstance(sid, str) and "::" in str(sid):
                disease = str(sid).split("::")[-1]
            
            # Record result
            rec = {
                "id": sid,
                "prompt": prompt,
                "response": response,
                "gt_answer": gt_raw,
                "pred_norm": pred_norm,
                "gt_norm": gt_norm,
                "disease": disease,
                "images": img_paths,
                "task_type": detected_task,
            }
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            results.append({
                "id": sid,
                "response": response,
                "gt_answer": gt_raw,
                "pred_norm": pred_norm,
                "gt_norm": gt_norm,
                "disease": disease,
            })
            processed += 1
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    
    # Compute metrics
    metrics = compute_metrics(df, detected_task)
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Samples processed: {metrics.get('samples', 0)}")
    print(f"Overall accuracy: {metrics.get('accuracy', 0.0):.4f}")
    
    if metrics.get("per_disease"):
        print("\nPer-disease accuracy:")
        for disease, info in sorted(metrics["per_disease"].items()):
            print(f"  {disease:20s}: n={info['n']:<4d} acc={info['acc']:.4f}")
    
    print(f"\nOutputs saved to:")
    print(f"  JSONL: {out_jsonl}")
    print(f"  CSV:   {out_csv}")
    print(f"\nSkipped: {skipped}")
    print("="*60)


if __name__ == "__main__":
    main()

