import json
from collections import Counter

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
    SK = True
except Exception:
    SK = False

LETTER_ORDER = ["A", "B", "C", "D", "E", "F"]  # extend if needed

def load_labels(path, task_filter="multi-choice"):
    with open(path, "r") as f:
        data = json.load(f)
    y_true, y_pred, ids = [], [], []
    for ex in data:
        if ex.get("task") != task_filter:
            continue
        gt = (ex.get("reference_answer") or "").strip()
        pd = (ex.get("prediction") or "").strip()
        if gt in LETTER_ORDER and pd in LETTER_ORDER:
            y_true.append(gt)
            y_pred.append(pd)
            ids.append(ex.get("id"))
    return y_true, y_pred, ids

def evaluate_mc(y_true, y_pred, labels=None):
    labels = labels or sorted(set(y_true) | set(y_pred), key=LETTER_ORDER.index)
    acc = sum(t==p for t,p in zip(y_true,y_pred)) / max(1, len(y_true))

    out = {"n": len(y_true), "accuracy": acc, "labels": labels}

    if SK:
        # Macro/weighted/micro precision/recall/F1
        P_macro, R_macro, F1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        P_weight, R_weight, F1_weight, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        P_micro, R_micro, F1_micro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="micro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        out.update({
            "precision_macro": P_macro,
            "recall_macro": R_macro,
            "f1_macro": F1_macro,
            "precision_weighted": P_weight,
            "recall_weighted": R_weight,
            "f1_weighted": F1_weight,
            "precision_micro": P_micro,   
            "recall_micro": R_micro,
            "f1_micro": F1_micro,
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist()
            },
            "per_class_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
        })
    else:
        labels = sorted(set(y_true) | set(y_pred), key=LETTER_ORDER.index)
        stats = {}
        for c in labels:
            tp = sum((t==c and p==c) for t,p in zip(y_true,y_pred))
            fp = sum((t!=c and p==c) for t,p in zip(y_true,y_pred))
            fn = sum((t==c and p!=c) for t,p in zip(y_true,y_pred))
            P = tp/(tp+fp) if (tp+fp)>0 else 0.0
            R = tp/(tp+fn) if (tp+fn)>0 else 0.0
            F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
            stats[c] = {"precision": P, "recall": R, "f1": F1}
        P_macro = sum(s["precision"] for s in stats.values())/len(stats)
        R_macro = sum(s["recall"] for s in stats.values())/len(stats)
        F1_macro = sum(s["f1"] for s in stats.values())/len(stats)
        out.update({
            "precision_macro": P_macro,
            "recall_macro": R_macro,
            "f1_macro": F1_macro,
            "note": "Install scikit-learn for weighted/micro metrics and confusion matrix."
        })
    return out

if __name__ == "__main__":
    y_true, y_pred, ids = load_labels("/gpfs/radev/pi/q_chen/yf329/PMC_instruction_data/benchmark/PMC_Data_Benchmark/results/multi-choice_llava.json")
    metrics = evaluate_mc(y_true, y_pred)
    from pprint import pprint
    pprint(metrics)
