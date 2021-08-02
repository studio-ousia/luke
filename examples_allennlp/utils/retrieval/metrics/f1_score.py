from typing import List, Dict


def compute_f1_score(prediction: List, gold: List) -> Dict[str, float]:
    prediction = set(prediction)
    gold = set(gold)

    tp = len(prediction & gold)

    if tp > 0:
        precision = tp / len(prediction)
        recall = tp / len(gold)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        precision = recall = f1 = 0
    return {"f1": f1, "precision": precision, "recall": recall}
