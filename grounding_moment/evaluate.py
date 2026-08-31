"""
evaluate.py
Span IoU evaluation utilities for Grounding MOMENT.
"""

import torch


def span_iou_1d(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    """Compute 1D IoU between two integer-indexed spans."""
    inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / (union + 1e-6) if union > 0 else 0.0


def evaluate_grounding(
    model,
    test_samples: list[dict],
    tokenizer,
    device: torch.device,
    iou_thresholds: list[float] | None = None,
    seq_len: int = 512,
) -> dict:
    """
    Evaluate model on test samples.

    Each test sample is a dict with:
        time_series:   list[float], length 512 (1-D)
        text:          str (single query phrase)
        expected_span: dict with "start" and "end" keys (integer indices)

    Returns:
        dict with mean_iou and recall@K for each threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5, 0.7]

    model.eval()
    ious        = []
    recall_hits = {t: 0 for t in iou_thresholds}

    with torch.no_grad():
        for sample in test_samples:
            # Shape: (1, 512, 1) — 2-D univariant time series
            ts = (torch.tensor(sample["time_series"], dtype=torch.float32)
                  .unsqueeze(0)   # (1, 512)
                  .unsqueeze(-1)  # (1, 512, 1)
                  .to(device))

            enc = tokenizer(
                sample["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            results = model.predict(ts, input_ids, attention_mask, seq_len=seq_len)
            pred    = results[0]  # single batch item

            gt_start = sample["expected_span"]["start"]
            gt_end   = sample["expected_span"]["end"]

            if pred["spans"].shape[0] == 0:
                ious.append(0.0)
                continue

            best_idx = pred["scores"].argmax().item()
            p_start  = pred["spans"][best_idx, 0].item()
            p_end    = pred["spans"][best_idx, 1].item()

            iou = span_iou_1d(int(p_start), int(p_end), gt_start, gt_end)
            ious.append(iou)

            for thr in iou_thresholds:
                if iou >= thr:
                    recall_hits[thr] += 1

    n = len(test_samples)
    metrics = {"mean_iou": sum(ious) / n if n else 0.0}
    for thr in iou_thresholds:
        metrics[f"recall@{thr}"] = recall_hits[thr] / n if n else 0.0

    return metrics
