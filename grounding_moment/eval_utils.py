"""
eval_utils.py
Shared evaluation helpers for eval.py and pred.py.

The model outputs:
  prim_spans / prim_scores / prim_token_idxs  — flat primitive detections
  event_spans / event_scores                  — event-level detections
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_1d(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    """1-D IoU between two spans expressed as real-valued (start, end)."""
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = (a_e - a_s) + (b_e - b_s) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Description filter (mirrors GroundingMomentDataset.__init__)
# ---------------------------------------------------------------------------

def is_desc_kept(
    desc: dict,
    use_pos_variant: bool = True,
    use_neg_variant: bool = True,
) -> bool:
    """Return True iff this description record should be included."""
    kind    = desc.get("kind", "")
    verdict = desc.get("critic_verdict", "accept")
    if kind == "positive_base":
        return True
    if kind == "positive_variant" and use_pos_variant and verdict == "accept":
        return True
    if kind == "negative" and use_neg_variant and verdict == "accept":
        return True
    return False


# ---------------------------------------------------------------------------
# GT primitive flattening
# ---------------------------------------------------------------------------

def get_gt_primitives(desc: dict, seq_len: int = 512) -> list[tuple[int, int]]:
    """
    Flatten all (event_occurrence, phrase) GT primitive spans from a
    description record into a list of integer (start, end) spans.

    Filters zero-length spans (padded entries).
    """
    gt_event_spans: list = desc.get("event_spans", [])
    grounding: list      = desc.get("grounding", [])
    K = len(gt_event_spans)

    prims: list[tuple[int, int]] = []
    for phrase in grounding:
        ts_spans = phrase.get("ts_spans", [])
        for k in range(K):
            if k >= len(ts_spans):
                continue
            ps, pe = ts_spans[k]
            if pe > ps:
                prims.append((int(ps), int(pe)))
    return prims


def get_gt_events(desc: dict) -> list[tuple[int, int]]:
    """
    Return GT event spans from a description record as integer (start, end) pairs.

    ``event_spans`` in the described JSON is a list of [start, end] in timestep indices.
    """
    raw: list = desc.get("event_spans", [])
    events: list[tuple[int, int]] = []
    for span in raw:
        ps, pe = int(span[0]), int(span[1])
        if pe > ps:
            events.append((ps, pe))
    return events


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7]


def compute_sample_metrics(
    pred: dict,
    desc: dict,
    seq_len: int = 512,
    iou_thresholds: list[float] | None = None,
) -> dict:
    """
    Compute primitive and event evaluation metrics for one (prediction, description) pair.

    Parameters
    ----------
    pred:
        Output dict from model.predict(...)[b]:
            prim_spans      – (Kp, 2) int tensor, absolute timestep [start, end]
            prim_scores     – (Kp,) float tensor
            prim_token_idxs – list[Kp] of 1D LongTensors
            event_spans     – (Ke, 2) int tensor, absolute timestep [start, end]
            event_scores    – (Ke,) float tensor

        Legacy keys ``spans`` / ``scores`` are also accepted (primitives only,
        no event metrics in that case).
    desc:
        One description record from sample_XXX_described.json.

    Returns
    -------
    dict with:
        is_negative_gt         – bool
        is_negative_pred       – bool
        classification_correct – bool
        prim_mean_iou          – float
        prim_best_ious         – list[float], one per GT primitive
        prim_recall_hits       – dict {thr: int}
        event_mean_iou         – float
        event_best_ious        – list[float], one per GT event
        event_recall_hits      – dict {thr: int}
    """
    if iou_thresholds is None:
        iou_thresholds = _DEFAULT_THRESHOLDS

    kind      = desc.get("kind", "positive_base")
    is_neg_gt = (kind == "negative")

    # ── Primitive predictions ────────────────────────────────────────────────
    # Accept both new key (prim_spans) and legacy key (spans)
    if "prim_spans" in pred:
        prim_tensor = pred["prim_spans"]
    else:
        prim_tensor = pred.get("spans", torch.zeros(0, 2, dtype=torch.long))

    pred_prim_spans: list[tuple[int, int]] = []
    if prim_tensor.numel() > 0:
        for row in prim_tensor.tolist():
            pred_prim_spans.append((int(row[0]), int(row[1])))
    is_neg_pred = (len(pred_prim_spans) == 0)
    classification_correct = (is_neg_gt == is_neg_pred)

    # ── GT primitives ────────────────────────────────────────────────────────
    gt_prims = get_gt_primitives(desc, seq_len)

    prim_best_ious:  list[float] = []
    prim_recall_hits = {t: 0 for t in iou_thresholds}

    if gt_prims and not is_neg_gt:
        used_pred = set()
        for gt_ps, gt_pe in gt_prims:
            best_iou  = 0.0
            best_pidx = -1
            for pidx, (ps, pe) in enumerate(pred_prim_spans):
                if pidx in used_pred:
                    continue
                iou = iou_1d(float(ps), float(pe), float(gt_ps), float(gt_pe))
                if iou > best_iou:
                    best_iou  = iou
                    best_pidx = pidx
            prim_best_ious.append(best_iou)
            for thr in iou_thresholds:
                if best_iou >= thr:
                    prim_recall_hits[thr] += 1
            if best_pidx >= 0:
                used_pred.add(best_pidx)

    prim_mean_iou = (
        sum(prim_best_ious) / len(prim_best_ious)
        if prim_best_ious else 0.0
    )

    # ── Event predictions ────────────────────────────────────────────────────
    event_tensor = pred.get("event_spans", torch.zeros(0, 2, dtype=torch.long))
    pred_event_spans: list[tuple[int, int]] = []
    if event_tensor.numel() > 0:
        for row in event_tensor.tolist():
            pred_event_spans.append((int(row[0]), int(row[1])))

    gt_events = get_gt_events(desc)

    event_best_ious:  list[float] = []
    event_recall_hits = {t: 0 for t in iou_thresholds}

    if gt_events and not is_neg_gt:
        used_ev = set()
        for gt_es, gt_ee in gt_events:
            best_iou  = 0.0
            best_eidx = -1
            for eidx, (es, ee) in enumerate(pred_event_spans):
                if eidx in used_ev:
                    continue
                iou = iou_1d(float(es), float(ee), float(gt_es), float(gt_ee))
                if iou > best_iou:
                    best_iou  = iou
                    best_eidx = eidx
            event_best_ious.append(best_iou)
            for thr in iou_thresholds:
                if best_iou >= thr:
                    event_recall_hits[thr] += 1
            if best_eidx >= 0:
                used_ev.add(best_eidx)

    event_mean_iou = (
        sum(event_best_ious) / len(event_best_ious)
        if event_best_ious else 0.0
    )

    return {
        "is_negative_gt":       is_neg_gt,
        "is_negative_pred":     is_neg_pred,
        "classification_correct": classification_correct,
        "prim_mean_iou":        round(prim_mean_iou, 6),
        "prim_best_ious":       [round(v, 6) for v in prim_best_ious],
        "prim_recall_hits":     prim_recall_hits,
        "event_mean_iou":       round(event_mean_iou, 6),
        "event_best_ious":      [round(v, 6) for v in event_best_ious],
        "event_recall_hits":    event_recall_hits,
    }


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def aggregate_metrics(
    records: list[dict],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """
    Aggregate a list of per-record dicts (output of compute_sample_metrics).

    Returns a nested dict with overall + per-kind breakdown.
    """
    if iou_thresholds is None:
        iou_thresholds = _DEFAULT_THRESHOLDS

    def _agg(recs: list[dict]) -> dict:
        if not recs:
            return {}
        n = len(recs)

        pr_ious = [v for r in recs for v in r["prim_best_ious"]]
        ev_ious = [v for r in recs for v in r["event_best_ious"]]

        pr_recall = {}
        ev_recall = {}
        for thr in iou_thresholds:
            total_pr_gt = sum(len(r["prim_best_ious"]) for r in recs)
            pr_hits = sum(r["prim_recall_hits"][thr] for r in recs)
            pr_recall[f"prim_recall@{thr}"] = pr_hits / total_pr_gt if total_pr_gt else 0.0

            total_ev_gt = sum(len(r["event_best_ious"]) for r in recs)
            ev_hits = sum(r["event_recall_hits"][thr] for r in recs)
            ev_recall[f"event_recall@{thr}"] = ev_hits / total_ev_gt if total_ev_gt else 0.0

        # Classification (positive = has gt primitives → not negative)
        tp = sum(1 for r in recs if not r["is_negative_gt"] and not r["is_negative_pred"])
        fp = sum(1 for r in recs if     r["is_negative_gt"] and not r["is_negative_pred"])
        fn = sum(1 for r in recs if not r["is_negative_gt"] and     r["is_negative_pred"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        out = {
            "n_records":      n,
            "prim_mean_iou":  round(sum(pr_ious) / len(pr_ious), 6) if pr_ious else 0.0,
            "event_mean_iou": round(sum(ev_ious) / len(ev_ious), 6) if ev_ious else 0.0,
            "cls_precision":  round(precision, 6),
            "cls_recall":     round(recall, 6),
            "cls_f1":         round(f1, 6),
        }
        out.update({k: round(v, 6) for k, v in pr_recall.items()})
        out.update({k: round(v, 6) for k, v in ev_recall.items()})
        return out

    kinds = ["positive_base", "positive_variant", "negative"]
    breakdown = {}
    for kind in kinds:
        subset = [r for r in records if r.get("kind") == kind]
        if subset:
            breakdown[kind] = _agg(subset)

    result = _agg(records)
    result["breakdown"] = breakdown
    return result
