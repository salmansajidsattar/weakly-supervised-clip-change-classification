"""Caption-derived weak labeling for LEVIR-CC (binary).

After inspecting the actual caption distribution we collapsed to 2 classes.
The original site_preparation class had ~58 train and ~11 test samples — too
few to train or evaluate honestly. Pure demolition without follow-up
construction is genuinely rare in LEVIR-CC. We treat it as future work and
focus on the cleaner binary task.

Classes:
  0 = no_change   — captions describing no/negligible change
  1 = completed   — new buildings / houses / villas have been built or appeared

Voting rule:
  Each of the 5 captions per pair votes for one of the two classes (or
  "no match"). When a caption matches both classes (rare), completed wins.
  A pair receives a label only when >= min_votes captions agree; otherwise
  -1 (ignored). Pairs whose dominant signal is removal-without-new-build
  (a small minority in LEVIR-CC) end up as ignored under this rule, which
  is acceptable: we report the ignore rate in label_coverage.json.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

CLASS_NAMES = ["no_change", "completed"]

CLASS_TEMPLATES = {
    "no_change": [
        "the scene is the same as before",
        "almost no changes occurred between the two images",
        "nothing has changed in the area",
    ],
    "completed": [
        "new buildings or houses have been built in the area",
        "many villas appeared on previously bare land",
        "the bareland is now covered with new structures and roads",
    ],
}

# --- Patterns ---------------------------------------------------------------

_BUILT_NOUN = (
    r"(houses?|villas?|buildings?|residences?|structures?|apartments?|"
    r"factories?|warehouses?|storage tanks?|tanks?|sheds?|towers?)"
)

_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "no_change": [
        re.compile(p) for p in [
            r"\bno (change|changes|difference|differences)\b",
            r"\bnothing (has |is )?changed\b",
            r"\b(almost|nearly|essentially|virtually) (no|the same)\b",
            r"\b(the |two )?(scene|area|areas|image|images|scenes) "
            r"(is|are|seem|seems|look|looks) "
            r"(the )?(same|identical|unchanged|basically the same)\b",
            r"\bidentical\b",
            r"\bunchanged\b",
            r"\bremain(s|ed)? (the )?same\b",
            r"\bsame as before\b",
            r"\bnot changed\b",
        ]
    ],
    "completed": [
        re.compile(p) for p in [
            rf"\b{_BUILT_NOUN} "
            r"(are|is|have been|has been|been|got) "
            r"(built|constructed|added|installed|placed|erected|set up|"
            r"rebuilt|reconstructed|finished|completed)\b",
            rf"\b{_BUILT_NOUN} (have |has )?"
            r"(appear|appears|appeared)\b",
            r"\b(built|constructed) (on|in|along|beside|near|across|"
            r"around|at|between|next to) (the |this |a |both )?",
            r"\bappear(s|ed)? (along|in|on|at|next to|beside|near|across|"
            r"around|between|on both sides of)",
            r"\breplace(s|d)? .*(bareland|bare land|wasteland|meadow|"
            r"trees|forest|plants|vegetation|clearing)",
            r"\b(bareland|bare land|wasteland|meadow|clearing|bare ground) "
            r"(becomes|become|is now|are now|has been replaced|have been replaced)",
            r"\b(many|more|some|several|a few|lots of|rows of|massive|"
            r"a number of|a group of|a row of|two|three|four|five) "
            rf"{_BUILT_NOUN}\b",
            r"\bcompleted (villas?|houses?|buildings?)\b",
            r"\b(roads? .* (and|with) .* "
            r"(houses?|villas?|buildings?))\b",
            r"\b((houses?|villas?|buildings?) .* (and|with) .* roads?)\b",
        ]
    ],
}


def _vote_one(caption: str) -> int:
    """Return class index for a single caption, or -1 if no rule fires.

    When both classes match, completed wins (the new end-state dominates).
    """
    caption = caption.lower().strip()
    matches: list[int] = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if any(p.search(caption) for p in _PATTERNS[cls_name]):
            matches.append(cls_idx)
    if not matches:
        return -1
    # completed (1) wins over no_change (0)
    if 1 in matches:
        return 1
    return 0


def vote_label(captions: Iterable[str], min_votes: int = 3) -> int:
    """Vote across captions; return class with >= min_votes, else -1."""
    votes = [_vote_one(c) for c in captions]
    counts = Counter(v for v in votes if v >= 0)
    if not counts:
        return -1
    top_class, top_count = counts.most_common(1)[0]
    if top_count >= min_votes:
        return top_class
    return -1


def build_weak_labels(
    captions_json: str | Path,
    out_dir: str | Path,
    min_votes: int = 3,
) -> dict[str, dict[str, int]]:
    """Read LevirCCcaptions.json, derive labels, write per-split JSON files."""
    captions_json = Path(captions_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reading captions from %s", captions_json)
    with open(captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["images"] if isinstance(data, dict) and "images" in data else data

    results: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    rule_fires: dict[str, Counter[str]] = {
        "train": Counter(), "val": Counter(), "test": Counter(),
    }
    ignore_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for item in items:
        split = item.get("split") or item.get("filepath")
        if split not in results:
            continue
        fname = item.get("filename") or item.get("imgid") or item.get("cocoid")
        sents = item.get("sentences", [])
        captions = [s.get("raw", "") for s in sents]
        label = vote_label(captions, min_votes=min_votes)
        results[split][str(fname)] = label
        if label < 0:
            ignore_counts[split] += 1
            rule_fires[split]["ignore"] += 1
        else:
            rule_fires[split][CLASS_NAMES[label]] += 1

    for split, mapping in results.items():
        out_path = out_dir / f"{split}_labels.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        log.info(
            "Split %s: %d pairs, %d ignored (%.1f%%)",
            split, len(mapping), ignore_counts[split],
            100.0 * ignore_counts[split] / max(1, len(mapping)),
        )
        for cls in CLASS_NAMES + ["ignore"]:
            c = rule_fires[split][cls]
            log.info("  class %-16s : %5d (%.1f%%)",
                     cls, c, 100.0 * c / max(1, len(mapping)))

    coverage_path = out_dir / "label_coverage.json"
    coverage = {
        "min_votes": min_votes,
        "class_names": CLASS_NAMES,
        "counts_per_split": {s: dict(rule_fires[s]) for s in rule_fires},
        "ignore_per_split": ignore_counts,
        "totals": {s: len(m) for s, m in results.items()},
    }
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)
    log.info("Coverage report: %s", coverage_path)
    return results
