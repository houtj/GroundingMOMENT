"""
dataset.py
Dataset for the Grounding MOMENT fractional-splitting pipeline.

Data layout on disk (one sample = two JSON files):
    {data_dir}/sample_{id}.json           — time series + event structure
    {data_dir}/sample_{id}_described.json — natural-language descriptions + grounding

For each sample, multiple description records are iterated.  A record is kept when:
  • kind == "positive_base"  (always)
  • kind == "positive_variant" and config.use_pos_variant and critic_verdict == "accept"
  • kind == "negative"       and config.use_neg_variant  and critic_verdict == "accept"

Each kept (sample, description) pair becomes one dataset item.

Tensor fields per item (P = config.max_phrases, T = max_text_len, K = num_events):

    time_series        (512, 1)          signal_noisy from the sample file
    input_ids          (T,)              BERT token ids
    attention_mask     (T,)              1 = real token, 0 = padding
    num_values         (T,) float        raw numeric values at [NUM] positions (Variant-3)
    num_mask           (T,) bool         True at [NUM] positions
    is_negative        bool              True iff kind == "negative"
    has_events         bool              num_events > 0
    num_events         int               number of event occurrences (K), 0 for negatives
    num_phrases        int               number of valid phrases in this description (≤ P)
    phrase_tok_start   (P,) long         inclusive BERT token index of phrase i; 0 for padded
    phrase_tok_end     (P,) long         exclusive BERT token index of phrase i; 0 for padded
    phrase_token_masks (P, T) bool       True at BERT positions belonging to phrase i
    event_pos_map      (T,) float        union of phrase_token_masks[:num_phrases] (for focal loss)
    event_spans        (K, 2) float      macro event (start, width) normalised to [0, 1]
    primitive_spans    (K, P, 2) float   primitive (start, width) per occurrence per phrase;
                                         padded phrases have zero spans

K_max is padded to the per-batch maximum in collate_fn.
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


SEQ_LEN = 512

# ---------------------------------------------------------------------------
# Number substitution helpers (Variant-3 numeric tokens)
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"(?<=\s)[-+]?\d+(?:\.\d+)?(?=\s)")


def _substitute_numbers(text: str, num_token: str) -> tuple[str, list[float]]:
    values: list[float] = []

    def _repl(m: re.Match) -> str:
        values.append(float(m.group(0)))
        return num_token

    return NUM_RE.sub(_repl, text), values


# ---------------------------------------------------------------------------
# Span normalisation helpers
# ---------------------------------------------------------------------------

def _span_to_start_width(start: int, end: int, seq_len: int = SEQ_LEN) -> tuple[float, float]:
    """Convert integer [start, end) indices to normalised (start, width) in [0, 1]."""
    s = start / seq_len
    w = (end - start) / seq_len
    return s, w


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GroundingMomentDataset(Dataset):
    """
    Loads paired sample_XXX.json + sample_XXX_described.json files.

    Args:
        data_dir:        Directory containing the JSON pairs.
        tokenizer:       Pre-loaded BertTokenizerFast.
        split:           "train" or "val".
        split_ratio:     Fraction of *samples* (not records) used for training.
        max_text_len:    BERT max token length.
        max_phrases:     Maximum number of phrases per description (P).
        use_pos_variant:            Include accepted positive-variant descriptions.
        use_neg_variant:            Include accepted negative descriptions.
        max_pos_variants_per_sample: Cap on accepted positive-variant descriptions per
                                     source sample. 0 = keep all.
        max_neg_variants_per_sample: Cap on accepted negative descriptions per source
                                     sample. 0 = keep all.
        num_token:                  Sentinel string for numeric token injection (Variant-3).
                                    When None the numeric path is fully disabled.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: BertTokenizerFast,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        max_text_len: int = 128,
        max_phrases: int = 7,
        use_pos_variant: bool = True,
        use_neg_variant: bool = True,
        max_pos_variants_per_sample: int = 0,
        max_neg_variants_per_sample: int = 0,
        num_token: str | None = None,
    ):
        self.data_dir       = Path(data_dir)
        self.tokenizer      = tokenizer
        self.max_text_len   = max_text_len
        self.max_phrases    = max_phrases
        self.use_pos_variant = use_pos_variant
        self.use_neg_variant = use_neg_variant
        self.max_pos_variants_per_sample = max_pos_variants_per_sample
        self.max_neg_variants_per_sample = max_neg_variants_per_sample
        self.num_token      = num_token
        self.num_token_id   = (
            tokenizer.convert_tokens_to_ids(num_token)
            if num_token is not None
            else None
        )

        # Discover all sample IDs by scanning for *_described.json files.
        described_files = sorted(self.data_dir.glob("sample_*_described.json"))
        sample_ids = []
        for df in described_files:
            # stem is "sample_XXX_described"; strip "_described"
            stem = df.stem  # e.g. "sample_000000_described"
            sid  = stem[: stem.rfind("_described")]  # "sample_000000"
            sample_path = self.data_dir / f"{sid}.json"
            if sample_path.exists():
                sample_ids.append(sid)

        # Randomly shuffle sample IDs with a fixed seed so train and val
        # datasets constructed with the same seed agree on the split.
        rng = random.Random(seed)
        rng.shuffle(sample_ids)

        n_total = len(sample_ids)
        n_train = int(n_total * split_ratio)
        if split == "train":
            selected = sample_ids[:n_train]
        else:
            selected = sample_ids[n_train:]

        # Build a lightweight index: (sample_path, described_path, desc_index).
        # JSON loading and tokenisation are deferred to __getitem__.
        self.index: list[tuple[Path, Path, int]] = []
        for sid in selected:
            sample_path    = self.data_dir / f"{sid}.json"
            described_path = self.data_dir / f"{sid}_described.json"
            with open(described_path) as f:
                described_data = json.load(f)
            n_pos = 0  # accepted positive_variant records kept for this sample
            n_neg = 0  # accepted negative records kept for this sample
            for i, desc in enumerate(described_data["descriptions"]):
                kind    = desc.get("kind", "")
                verdict = desc.get("critic_verdict", "accept")
                if kind == "positive_base":
                    pass
                elif (kind == "positive_variant"
                        and self.use_pos_variant
                        and verdict == "accept"):
                    if (self.max_pos_variants_per_sample > 0
                            and n_pos >= self.max_pos_variants_per_sample):
                        continue
                    n_pos += 1
                elif kind == "negative" and self.use_neg_variant and verdict == "accept":
                    if (self.max_neg_variants_per_sample > 0
                            and n_neg >= self.max_neg_variants_per_sample):
                        continue
                    n_neg += 1
                else:
                    continue
                self.index.append((sample_path, described_path, i))

        print(f"Dataset ({split}): {len(self.index)} records from {len(selected)} samples in {self.data_dir}")

    # ------------------------------------------------------------------
    # Record conversion
    # ------------------------------------------------------------------

    def _convert(self, signal: np.ndarray, desc: dict) -> dict | None:
        """Convert one (signal, description) pair to a sample dict."""
        kind       = desc.get("kind", "")
        is_negative = (kind == "negative")
        text        = desc["text"]
        grounding   = desc.get("grounding", [])       # list of {text_span, ts_spans}
        event_spans_raw = desc.get("event_spans", []) # list of [start, end] in timestep indices

        # ------------------------------------------------------------------
        # Time series tensor
        # ------------------------------------------------------------------
        ts = torch.from_numpy(signal).unsqueeze(-1)   # (512, 1)

        # ------------------------------------------------------------------
        # Numeric token substitution (Variant-3)
        # ------------------------------------------------------------------
        if self.num_token is not None:
            text_enc, num_values_list = _substitute_numbers(text, self.num_token)
        else:
            text_enc        = text
            num_values_list = []

        # ------------------------------------------------------------------
        # Tokenise — always request offset_mapping so we can map char spans
        # ------------------------------------------------------------------
        enc = self.tokenizer(
            text_enc,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids      = enc["input_ids"].squeeze(0)        # (T,)
        attention_mask = enc["attention_mask"].squeeze(0)   # (T,)
        offset_mapping = enc["offset_mapping"].squeeze(0)   # (T, 2)

        # ------------------------------------------------------------------
        # Char → token mapping helper
        # ------------------------------------------------------------------
        def _char_to_tok(char_start: int, char_end: int) -> tuple[int, int]:
            tok_s = tok_e = 0
            found_start = False
            for i, (cs, ce) in enumerate(offset_mapping.tolist()):
                if cs == 0 and ce == 0:
                    continue
                if not found_start and cs >= char_start:
                    tok_s = i
                    found_start = True
                if found_start and ce >= char_end:
                    tok_e = i + 1
                    break
            return tok_s, min(tok_e, self.max_text_len)

        # ------------------------------------------------------------------
        # Phrase token bounds  (P,)
        # ------------------------------------------------------------------
        P = self.max_phrases
        T = self.max_text_len

        num_phrases = len(grounding)
        if num_phrases > P:
            # Truncate silently; the model only handles P phrases
            grounding = grounding[:P]
            num_phrases = P

        phrase_tok_start = torch.zeros(P, dtype=torch.long)
        phrase_tok_end   = torch.zeros(P, dtype=torch.long)
        phrase_token_masks = torch.zeros(P, T, dtype=torch.bool)

        for i, g in enumerate(grounding):
            char_s, char_e = g["text_span"]
            ts_i, te_i     = _char_to_tok(char_s, char_e)
            phrase_tok_start[i] = ts_i
            phrase_tok_end[i]   = te_i
            if te_i > ts_i:
                phrase_token_masks[i, ts_i:te_i] = True

        # event_pos_map: union of all valid phrase masks
        if num_phrases > 0:
            event_pos_map = phrase_token_masks[:num_phrases].any(dim=0).float()
        else:
            event_pos_map = torch.zeros(T, dtype=torch.float32)

        # ------------------------------------------------------------------
        # Event spans and primitive spans
        # For each event occurrence j, grounding[i].ts_spans[j] gives the
        # timestep-index [start, end) for phrase i in that occurrence.
        # ------------------------------------------------------------------
        K = len(event_spans_raw)  # number of occurrences

        if K > 0 and not is_negative:
            ev_spans = torch.zeros(K, 2, dtype=torch.float32)
            for j, (es, ee) in enumerate(event_spans_raw):
                s, w = _span_to_start_width(es, ee)
                ev_spans[j] = torch.tensor([s, w])

            prim_spans = torch.zeros(K, P, 2, dtype=torch.float32)
            for i, g in enumerate(grounding):
                ts_spans_i = g["ts_spans"]
                for j, (ps, pe) in enumerate(ts_spans_i):
                    if j >= K:
                        break
                    s, w = _span_to_start_width(ps, pe)
                    prim_spans[j, i] = torch.tensor([s, w])
        else:
            # Negative or no events
            K = 0
            ev_spans   = torch.zeros(0, 2, dtype=torch.float32)
            prim_spans = torch.zeros(0, P, 2, dtype=torch.float32)

        # ------------------------------------------------------------------
        # Numeric values / mask (Variant-3)
        # ------------------------------------------------------------------
        if self.num_token is not None:
            num_values_tensor = torch.zeros(T, dtype=torch.float32)
            num_mask_tensor   = torch.zeros(T, dtype=torch.bool)
            val_iter = iter(num_values_list)
            for pos, tok_id in enumerate(input_ids.tolist()):
                if tok_id == self.num_token_id:
                    v = next(val_iter, 0.0)
                    num_values_tensor[pos] = float(v)
                    num_mask_tensor[pos]   = True
        else:
            num_values_tensor = None
            num_mask_tensor   = None

        # ------------------------------------------------------------------
        # Pack
        # ------------------------------------------------------------------
        sample: dict = {
            "time_series":       ts,                  # (512, 1)
            "input_ids":         input_ids,           # (T,)
            "attention_mask":    attention_mask,      # (T,)
            "is_negative":       is_negative,
            "has_events":        K > 0,
            "num_events":        K,
            "num_phrases":       num_phrases,
            "phrase_tok_start":  phrase_tok_start,    # (P,)
            "phrase_tok_end":    phrase_tok_end,      # (P,)
            "phrase_token_masks": phrase_token_masks, # (P, T)
            "event_pos_map":     event_pos_map,       # (T,)
            "event_spans":       ev_spans,            # (K, 2)
            "primitive_spans":   prim_spans,          # (K, P, 2)
        }
        if num_values_tensor is not None:
            sample["num_values"] = num_values_tensor  # (T,)
            sample["num_mask"]   = num_mask_tensor    # (T,)

        return sample

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        sample_path, described_path, desc_idx = self.index[idx]
        with open(sample_path) as f:
            sample_data = json.load(f)
        with open(described_path) as f:
            described_data = json.load(f)
        signal = np.array(sample_data["signal_noisy"], dtype=np.float32)
        desc   = described_data["descriptions"][desc_idx]
        return self._convert(signal, desc)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Custom collation:
          - Tensors with fixed shape (time_series, input_ids, etc.) are stacked.
          - event_spans / primitive_spans are zero-padded to K_max (max occurrences
            in the batch) before stacking.
          - Scalar fields (is_negative, has_events, num_events, num_phrases) are
            kept as lists.
          - num_values / num_mask are stacked when present.
        """
        K_max = max(s["num_events"] for s in batch)

        result: dict = {
            "time_series":       torch.stack([s["time_series"]       for s in batch]),
            "input_ids":         torch.stack([s["input_ids"]         for s in batch]),
            "attention_mask":    torch.stack([s["attention_mask"]    for s in batch]),
            "phrase_tok_start":  torch.stack([s["phrase_tok_start"]  for s in batch]),
            "phrase_tok_end":    torch.stack([s["phrase_tok_end"]    for s in batch]),
            "phrase_token_masks":torch.stack([s["phrase_token_masks"] for s in batch]),
            "event_pos_map":     torch.stack([s["event_pos_map"]     for s in batch]),
            "is_negative":       [s["is_negative"]  for s in batch],
            "has_events":        [s["has_events"]   for s in batch],
            "num_events":        [s["num_events"]   for s in batch],
            "num_phrases":       [s["num_phrases"]  for s in batch],
        }

        # Pad event_spans and primitive_spans to K_max
        ev_padded   = []
        prim_padded = []
        P = batch[0]["primitive_spans"].shape[1] if batch[0]["primitive_spans"].ndim == 3 else batch[0]["phrase_tok_start"].shape[0]
        for s in batch:
            K = s["num_events"]
            pad = K_max - K
            ev   = s["event_spans"]     # (K, 2)
            prim = s["primitive_spans"] # (K, P, 2)
            if pad > 0:
                ev   = torch.cat([ev,   torch.zeros(pad, 2,    dtype=ev.dtype)],   dim=0)
                prim = torch.cat([prim, torch.zeros(pad, P, 2, dtype=prim.dtype)], dim=0)
            elif K_max == 0:
                ev   = torch.zeros(0, 2,    dtype=torch.float32)
                prim = torch.zeros(0, P, 2, dtype=torch.float32)
            ev_padded.append(ev)
            prim_padded.append(prim)

        if K_max > 0:
            result["event_spans"]     = torch.stack(ev_padded)    # (B, K_max, 2)
            result["primitive_spans"] = torch.stack(prim_padded)  # (B, K_max, P, 2)
        else:
            B = len(batch)
            result["event_spans"]     = torch.zeros(B, 0, 2,    dtype=torch.float32)
            result["primitive_spans"] = torch.zeros(B, 0, P, 2, dtype=torch.float32)

        if "num_values" in batch[0]:
            result["num_values"] = torch.stack([s["num_values"] for s in batch])
            result["num_mask"]   = torch.stack([s["num_mask"]   for s in batch])

        return result
