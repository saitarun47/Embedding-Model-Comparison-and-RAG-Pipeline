from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.metrics import f1_score

def compute_metrics(prediction, ground_truth):
    # Exact Match
    em = int(prediction.strip().lower() == ground_truth.strip().lower())

    # F1 Score (token level)
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    f1 = (2 * len(common)) / (len(pred_tokens) + len(gt_tokens)) if (len(pred_tokens) + len(gt_tokens)) > 0 else 0

    # BLEU Score
    bleu = sentence_bleu([gt_tokens], pred_tokens)

    # ROUGE-L
    rouge = Rouge()
    scores = rouge.get_scores(prediction, ground_truth, avg=True)
    rouge_l = scores['rouge-l']['f']

    return {"EM": em, "F1": round(f1, 4), "BLEU": round(bleu, 4), "ROUGE": round(rouge_l, 4)}
