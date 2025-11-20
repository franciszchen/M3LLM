from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import re
import difflib
import evaluate
from evaluate import load
from sentence_transformers import SentenceTransformer
from cider import Cider
from tokenizer.ptbtokenizer import PTBTokenizer
from typing import Dict, List, Tuple

bertscore = load("bertscore")
_bertscore = None
sbertmodel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
rouge_scorer = Rouge()

def tokenize(text):
    return text.lower().replace(".", " .").split()

def _prepare_cider_inputs(results: List[Dict], id_key="id", ref_key="reference_answer", pred_key="prediction"):
    """Build gts/res dicts: {id: [str, ...]} expected by Cider."""
    gts, res = {}, {}
    for rec in results:
        sid = str(rec[id_key])
        gts[sid] = [rec.get(ref_key, "")]
        res[sid] = [rec.get(pred_key, "")]
    return gts, res

def compute_cider_corpus(results: List[Dict],
                         id_key="id", ref_key="reference_answer", pred_key="prediction",
                         n: int = 4, sigma: float = 6.0,
                         use_ptb_tokenizer: bool = False) -> Tuple[float, Dict[str, float]]:
    """
    Returns (corpus_cider, per_item) and does COCO-style tokenization if requested.
    """
    gts, res = _prepare_cider_inputs(results, id_key, ref_key, pred_key)

    if use_ptb_tokenizer:
        if not _HAS_PTB:
            raise RuntimeError("PTBTokenizer not found; set use_ptb_tokenizer=False or fix the import.")
        tok = PTBTokenizer()
        gts = tok.tokenize(gts)
        res = tok.tokenize(res)

    cider = Cider(n=n, sigma=sigma)
    corpus, per_list = cider.compute_score(gts, res)   # corpus: float, per_list: [float] aligned with keys order
    sid_list = list(gts.keys())
    per_item = {sid: float(sc) for sid, sc in zip(sid_list, per_list)}
    return float(corpus), per_item

def add_cider_to_results(results: List[Dict],
                         agg_sums: Dict[str, float],
                         metric_key: str = "cider",
                         **kwargs) -> float:
    """
    Compute corpus CIDEr, write each sample's CIDEr into rec['metrics'][metric_key],
    add to agg_sums, and return the corpus score.
    """
    corpus, per_item = compute_cider_corpus(results, **kwargs)
    for rec in results:
        sid = str(rec.get(kwargs.get("id_key", "id")))
        if "metrics" in rec and sid in per_item:
            rec["metrics"][metric_key] = per_item[sid]
            agg_sums[metric_key] = agg_sums.get(metric_key, 0.0) + per_item[sid]
    return corpus


def calculate_bleu_scores(reference, prediction):
    reference_tokens = tokenize(reference)
    prediction_tokens = tokenize(prediction)
    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n) + (0.0,) * (4 - n)
        scores[f"bleu{n}"] = sentence_bleu([reference_tokens], prediction_tokens, weights=weights)
    return scores

def calculate_bertscore(pred, ref, lang="en", model_type="microsoft/deberta-xlarge-mnli"):
    """
    Accepts str (or list[str]) and returns flat floats:
      {"bertscore_p": float, "bertscore_r": float, "bertscore_f1": float}
    """
    global _bertscore
    if _bertscore is None:
        _bertscore = evaluate.load("bertscore")

    preds = [pred] if isinstance(pred, str) else list(pred)
    refs  = [ref]  if isinstance(ref, str)  else list(ref)

    out = _bertscore.compute(
        predictions=preds,
        references=refs,
        lang=lang,
        model_type=model_type
    )
    # out["precision"], out["recall"], out["f1"] are lists of floats
    p = float(sum(out["precision"]) / len(out["precision"]))
    r = float(sum(out["recall"])    / len(out["recall"]))
    f = float(sum(out["f1"])        / len(out["f1"]))
    return {"bertscore_p": p, "bertscore_r": r, "bertscore_f1": f}

def calculate_bertsimilarity(reference,prediction):
    embeddings_reference = sbertmodel.encode(reference)
    embeddings_prediction = sbertmodel.encode(prediction)
    similarities = sbertmodel.similarity(embeddings_reference, embeddings_prediction)
    #print(float(similarities.squeeze().item()))
    return float(similarities.squeeze().item())

def calculate_rouge(reference, prediction):
    try:
        scores = rouge_scorer.get_scores(prediction.lower(), reference.lower())[0]
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }
    except:
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

def calculate_em(pred, ref):
    return int(pred.strip().lower() == ref.strip().lower())

def calculate_f1(prediction, ground_truth):
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    common = pred_tokens & gt_tokens
    if not pred_tokens or not gt_tokens:
        return 0, 0, 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

def evaluate_sample(pred, target):
    result = {
        "bleu1": bleu(pred, target, 1),
        "bleu2": bleu(pred, target, 2),
        "bleu3": bleu(pred, target, 3),
        "bleu4": bleu(pred, target, 4),
        "em": int(pred.strip().lower() == target.strip().lower()),
    }
    rouge_result = rouge_scores(pred, target)
    result.update({
        "rouge1": rouge_result["rouge-1"]["f"],
        "rouge2": rouge_result["rouge-2"]["f"],
        "rougel": rouge_result["rouge-l"]["f"],
    })
    return result



def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    most_similar_index = 0
    highest_similarity = 0.0
    for i, s in enumerate(str_list):
        sim = str_similarity(s, target_str)
        if sim > highest_similarity:
            most_similar_index = i
            highest_similarity = sim
    return most_similar_index


def extract(text: str, key: str) -> str:
    m = re.search(rf"<{key}>(.*?)</{key}>", text, flags=re.S|re.I)
    if m:
        return m.group(1)
    m = re.search(rf"{key}\s*[:：]\s*(.*)", text, flags=re.I)
    return m.group(1) if m else text

def parse_response(response):
    response = response.lower()
    if "boxed" in response:
        response = extract(response,"answer")
    elif "<answer>" in response:
        response = extract(response,"answer")
    answer_patterns = [
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is",
        "answer:",
        "答案:",
        "final answer",
        "final answer is"
    ]
    for pat in answer_patterns:
        if pat in response:
            response = response.split(pat)[-1]
    return response

def parse_options_from_prompt(prompt: str):
    """
    Extract multiple-choice options from the question text.
    Works when options:
      - start right after the question: "...?A. foo"
      - are on new lines: "A. foo\nB. bar\n..."
      - use A/B/C with ., ), or :
    Returns lowercase option texts without the leading letters.
    """
    text = prompt or ""
    lines = re.split(r"\n+", text)
    opts, buf, current_letter = [], "", None

    def flush():
        nonlocal buf, current_letter, opts
        if current_letter is not None:
            cleaned = re.sub(r"^[A-F]\s*[\.\):\-]\s*", "", buf.strip(), flags=re.I)
            if cleaned:
                opts.append(cleaned.strip().lower())
        buf, current_letter = "", None

    for line in lines:
        m = re.match(r"^\s*([A-F])\s*[\.\):\-]\s*(.*)", line, flags=re.I)
        if m:
            flush()
            current_letter = m.group(1).upper()
            buf = m.group(2)
        else:
            if current_letter is not None:
                buf = (buf + " " + line.strip()).strip()
    flush()
    if opts:
        return opts
    
    parts = re.split(r"(?<![A-Za-z])([A-F])\s*[\.\):\-]\s*", text)
    merged = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            merged.append(parts[i+1].strip().lower())
    return merged

def judge_multi_choice(choices, answer, response, alphas=None):
    response = response.lower()
    if response.split("\n\n")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[0]
    elif response.split("\n\n")[-1].split(".")[0] in [chr(ord('a') + i) for i in range(len(choices))]:
        response = response.split("\n\n")[-1].split(".")[0]
    
    response = parse_response(response)
    alphas = [chr(ord('a') + i) for i in range(len(choices))]
    choices = [choice.lower() for choice in choices]
    flag = False
    response = response.strip().lower()
    response = response.replace("\n","")
    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1]
    answer = answer.strip().lower()
    
    if len(split_response) > 300:
        flag = False
    # letter,letter.  choice,choice
    if split_response == answer:
        flag = True
    
    # letter,choice
    elif split_response in alphas:
        if choices[ord(split_response)-ord("a")]== answer:
            flag = True
    
    # choice letter
    elif split_response in choices:
        if answer in alphas and split_response == choices[ord(answer)-ord("a")]:
            flag = True
    # unparsed
    else:
        index = find_most_similar_index(choices,response)
        if alphas[index] == answer or choices[index] == answer:
            flag = True
    return flag
