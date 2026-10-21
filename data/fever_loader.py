"""
FEVER dataset loader and preprocessor.

Downloads from HuggingFace (copenlu/fever_gold_evidence), normalizes
records to the internal ClaimRecord schema, creates a stratified
development subset, and caches the result locally as JSONL.

Usage:
    python -m data.fever_loader            # Run with defaults
    python -m data.fever_loader --help     # Show options
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

from data.schema import ClaimRecord, EvidencePiece, save_records, load_records
from utils.logging_setup import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_DATASET = "copenlu/fever_gold_evidence"
DEFAULT_DEV_SIZE = 750  # 250 per label for balanced dev subset
LABELS_3WAY = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]


# ---------------------------------------------------------------------------
# Raw → Normalized conversion
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Light cleanup of FEVER evidence text (e.g. -LRB- → '(')."""
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    text = text.replace("-COLON-", ":")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_evidence_id(doc_title: str, sentence_idx) -> str:
    """Create a deterministic evidence ID from doc title and sentence index."""
    return f"{doc_title}_{sentence_idx}"


def _normalize_record(raw: dict, split: str = "train") -> Optional[ClaimRecord]:
    """Convert a raw HuggingFace FEVER record to ClaimRecord.

    Returns None if the record is invalid (empty claim or missing fields).
    """
    claim_text = (raw.get("claim") or "").strip()
    if not claim_text:
        return None

    label_raw = (raw.get("label") or "").strip().upper().replace(" ", "_")
    if label_raw not in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}:
        return None

    claim_id = raw.get("id", "")
    if not claim_id:
        # Fallback: hash the claim text
        claim_id = hashlib.md5(claim_text.encode("utf-8")).hexdigest()

    # Parse evidence: list of [doc_title, sentence_idx, evidence_text]
    evidence_pieces: List[EvidencePiece] = []
    seen_ids = set()
    for ev_group in raw.get("evidence", []):
        if not ev_group or len(ev_group) < 3:
            continue
        doc_title = str(ev_group[0])
        sentence_idx = int(ev_group[1]) if ev_group[1] is not None else 0
        ev_text = _clean_text(str(ev_group[2]))

        if not ev_text:
            continue

        ev_id = _make_evidence_id(doc_title, sentence_idx)
        if ev_id in seen_ids:
            continue  # skip duplicate evidence within same claim
        seen_ids.add(ev_id)

        evidence_pieces.append(
            EvidencePiece(
                evidence_id=ev_id,
                text=ev_text,
                doc_title=doc_title,
                sentence_idx=sentence_idx,
                source="fever",
                source_type="wikipedia",
            )
        )

    return ClaimRecord(
        claim_id=claim_id,
        claim_text=claim_text,
        label=label_raw,
        evidence=evidence_pieces,
        original_id=raw.get("original_id"),
        verifiable=raw.get("verifiable"),
        dataset="fever",
        split=split,
    )


# ---------------------------------------------------------------------------
# Download + preprocess
# ---------------------------------------------------------------------------

def load_fever_split(
    split: str = "train",
    max_records: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[ClaimRecord]:
    """Load and normalize a FEVER split from HuggingFace.

    Args:
        split: HuggingFace split name (train / validation).
        max_records: If set, stop after this many valid records.
        cache_dir: Directory to cache the raw HuggingFace download.

    Returns:
        List of normalized ClaimRecord objects.
    """
    from datasets import load_dataset

    logger.info(f"Loading FEVER split='{split}' from HuggingFace ({HF_DATASET})...")

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    ds = load_dataset(HF_DATASET, split=split, streaming=True, **kwargs)

    records: List[ClaimRecord] = []
    skipped = 0
    seen_ids = set()

    for raw in ds:
        rec = _normalize_record(raw, split=split)
        if rec is None:
            skipped += 1
            continue
        if rec.claim_id in seen_ids:
            skipped += 1
            continue
        seen_ids.add(rec.claim_id)
        records.append(rec)

        if max_records and len(records) >= max_records:
            break

    logger.info(
        f"Loaded {len(records)} records (skipped {skipped}) from FEVER {split}."
    )
    return records


# ---------------------------------------------------------------------------
# Stratified dev subset
# ---------------------------------------------------------------------------

def create_dev_subset(
    records: List[ClaimRecord],
    per_label: int = 250,
    seed: int = 42,
) -> List[ClaimRecord]:
    """Create a balanced development subset with equal claims per label.

    Args:
        records: Full list of ClaimRecords.
        per_label: Number of claims to sample per label.
        seed: Random seed for reproducibility.

    Returns:
        Stratified subset of ClaimRecords.
    """
    set_seed(seed)

    by_label: dict[str, list] = {lbl: [] for lbl in LABELS_3WAY}
    for rec in records:
        if rec.label in by_label:
            by_label[rec.label].append(rec)

    subset: List[ClaimRecord] = []
    for lbl in LABELS_3WAY:
        pool = by_label[lbl]
        n = min(per_label, len(pool))
        sampled = random.sample(pool, n)
        # Tag as dev split
        for rec in sampled:
            rec.split = "dev"
        subset.extend(sampled)
        logger.info(f"  {lbl}: sampled {n}/{len(pool)}")

    random.shuffle(subset)
    logger.info(f"Dev subset: {len(subset)} claims total.")
    return subset


# ---------------------------------------------------------------------------
# Evidence corpus extraction
# ---------------------------------------------------------------------------

def extract_evidence_corpus(records: List[ClaimRecord]) -> List[dict]:
    """Extract a deduplicated evidence corpus from claim records.

    Returns a list of dicts with evidence_id, text, doc_title, sentence_idx,
    suitable for building a retrieval index.
    """
    corpus = {}
    for rec in records:
        for ev in rec.evidence:
            if ev.evidence_id not in corpus:
                corpus[ev.evidence_id] = {
                    "evidence_id": ev.evidence_id,
                    "text": ev.text,
                    "doc_title": ev.doc_title,
                    "sentence_idx": ev.sentence_idx,
                    "source": ev.source,
                    "source_type": ev.source_type,
                }
    result = list(corpus.values())
    logger.info(f"Evidence corpus: {len(result)} unique pieces from {len(records)} claims.")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_fever_data(
    output_dir: str = "data",
    max_records: int = 5000,
    per_label: int = 250,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> dict:
    """Full FEVER data preparation pipeline.

    1. Downloads / streams FEVER from HuggingFace.
    2. Normalizes to internal schema.
    3. Creates a stratified dev subset.
    4. Extracts evidence corpus.
    5. Saves all artifacts as JSONL / JSON.

    Returns dict with paths to created files.
    """
    set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load and normalize
    records = load_fever_split(
        split="train", max_records=max_records, cache_dir=cache_dir
    )

    # 2. Save full processed records
    full_path = str(out / "fever_full.jsonl")
    save_records(records, full_path)
    logger.info(f"Saved full dataset: {full_path}")

    # 3. Create and save dev subset
    dev_subset = create_dev_subset(records, per_label=per_label, seed=seed)
    dev_path = str(out / "fever_dev.jsonl")
    save_records(dev_subset, dev_path)
    logger.info(f"Saved dev subset: {dev_path}")

    # 4. Extract and save evidence corpus (from dev subset for now)
    corpus = extract_evidence_corpus(dev_subset)
    corpus_path = str(out / "evidence_corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved evidence corpus: {corpus_path}")

    # 5. Save dataset metadata
    label_dist = Counter(r.label for r in dev_subset)
    meta = {
        "dataset": "fever",
        "source": HF_DATASET,
        "full_records": len(records),
        "dev_records": len(dev_subset),
        "evidence_corpus_size": len(corpus),
        "label_distribution": dict(label_dist),
        "per_label": per_label,
        "seed": seed,
    }
    meta_path = str(out / "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")

    return {
        "full_path": full_path,
        "dev_path": dev_path,
        "corpus_path": corpus_path,
        "meta_path": meta_path,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FEVER dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--max-records", type=int, default=5000,
                        help="Max records to load from FEVER train")
    parser.add_argument("--per-label", type=int, default=250,
                        help="Claims per label in dev subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache dir")
    args = parser.parse_args()

    result = prepare_fever_data(
        output_dir=args.output_dir,
        max_records=args.max_records,
        per_label=args.per_label,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    print(f"\nDataset preparation complete.")
    print(f"  Full records: {result['meta']['full_records']}")
    print(f"  Dev subset:   {result['meta']['dev_records']}")
    print(f"  Evidence:     {result['meta']['evidence_corpus_size']}")
    print(f"  Label dist:   {result['meta']['label_distribution']}")
