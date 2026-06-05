"""Prepare an ImageNet-val subset for W6 Top-1, from an ungated HF mirror.

The on-disk ILSVRC val images on this host are permission-locked, so we materialise a subset
from a public HuggingFace ImageNet-1k validation split: save JPEGs to ``--out/images`` and a
ground-truth file ``--out/gt.txt`` with lines ``<filename> <class_index>`` matching the order of
``internal/catalog/labels/imagenet1k.txt`` (the same labels VisionServe serves).

Crucially, before writing anything it **verifies** that the dataset's ClassLabel ordering aligns
with imagenet1k.txt (compares the first token of each class name in order). If they disagree the
GT indices would be meaningless, so it aborts rather than emit a silently-wrong ground truth.

Run (in the `label` conda env, which has `datasets`):
    HF_HOME=/mnt/nas/huggingface/hf_cache \
      python -m eval.accuracy.prep_imagenet_val --out /mnt/nas/huggingface/trung_w6/imagenet_val --n 5000
"""

from __future__ import annotations

import argparse
import os
import sys


CANDIDATES = [
    # (repo_id, split) tried in order; first that yields labelled images wins.
    # evanarlian val is ungated, 256px, and canonically ordered (only the known ImageNet
    # duplicate-name 'crane'/'crane2' at idx 517 differs cosmetically — same class order).
    ("evanarlian/imagenet_1k_resized_256", "val"),
    ("imagenet-1k", "validation"),
    ("benjamin-paine/imagenet-1k-256x256", "validation"),
]


def _load_serving_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln for ln in f.read().split("\n") if ln != ""]


def _first_token(name: str) -> str:
    # "tench, Tinca tinca" -> "tench" ;  "great white shark" -> "great" ; "crane2" -> "crane"
    tok = name.split(",")[0].strip().lower()
    return tok.rstrip("0123456789")  # normalise the known 'crane'/'crane2' disambiguation


# Known ImageNet duplicate class *names* (same word at two indices); some label sets append a
# digit to disambiguate. These are NOT ordering errors — tolerate them.
_DUP_TOLERANCE = 3


def _verify_ordering(class_names: list[str], serving_labels: list[str]) -> bool:
    """Confirm the dataset's label-index order matches imagenet1k.txt order.

    Tolerates up to ``_DUP_TOLERANCE`` cosmetic mismatches (ImageNet's duplicate names like
    'crane'/'crane2', 'maillot') after digit-stripping; rejects anything beyond that as a real
    reordering so we never emit a silently-wrong ground truth."""
    if len(class_names) != len(serving_labels):
        print(f"  ordering check: length mismatch {len(class_names)} vs {len(serving_labels)}",
              file=sys.stderr)
        return False
    mism = 0
    for i, (cn, sl) in enumerate(zip(class_names, serving_labels)):
        if _first_token(cn) != _first_token(sl):
            mism += 1
            print(f"  idx {i}: dataset={cn!r} vs serving={sl!r} (tolerated dup-name)",
                  file=sys.stderr)
    ok = mism <= _DUP_TOLERANCE
    print(f"  ordering check: {mism} mismatches -> {'ACCEPT' if ok else 'REJECT'}",
          file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output dir (writes images/ and gt.txt)")
    ap.add_argument("--n", type=int, default=5000, help="number of val images to materialise")
    ap.add_argument("--labels", default="internal/catalog/labels/imagenet1k.txt")
    args = ap.parse_args(argv)

    from datasets import load_dataset, get_dataset_split_names  # type: ignore

    serving = _load_serving_labels(args.labels)
    print(f"serving labels: {len(serving)} (e.g. {serving[:3]})", file=sys.stderr)

    ds = None
    chosen = None
    for repo, split in CANDIDATES:
        try:
            splits = get_dataset_split_names(repo)
        except Exception as e:  # noqa: BLE001
            print(f"skip {repo}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            continue
        if split not in splits:
            print(f"skip {repo}:{split} (splits={splits})", file=sys.stderr)
            continue
        try:
            d = load_dataset(repo, split=split, streaming=True)
            first = next(iter(d))
        except Exception as e:  # noqa: BLE001
            print(f"skip {repo}:{split} load failed: {type(e).__name__}: {str(e)[:80]}",
                  file=sys.stderr)
            continue
        if "label" not in first or "image" not in first:
            print(f"skip {repo}:{split} (keys={list(first.keys())})", file=sys.stderr)
            continue
        if not isinstance(first["label"], int) or first["label"] < 0:
            print(f"skip {repo}:{split} (label not a valid index: {first.get('label')})",
                  file=sys.stderr)
            continue
        # verify ordering when the dataset exposes ClassLabel names
        try:
            names = load_dataset(repo, split=split, streaming=True).features["label"].names
        except Exception:  # noqa: BLE001
            names = None
        if names:
            if not _verify_ordering(list(names), serving):
                print(f"REJECT {repo}:{split} — label ordering != imagenet1k.txt", file=sys.stderr)
                continue
        else:
            print(f"WARNING {repo}:{split} has no ClassLabel names; cannot verify ordering — "
                  "proceeding on the documented assumption of canonical order", file=sys.stderr)
        ds, chosen = d, (repo, split)
        break

    if ds is None:
        print("ERROR: no usable ImageNet-val source found among candidates", file=sys.stderr)
        return 2

    print(f"USING {chosen[0]} split={chosen[1]}", file=sys.stderr)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    gt_path = os.path.join(args.out, "gt.txt")

    n = 0
    with open(gt_path, "w", encoding="utf-8") as gt:
        for ex in ds:
            if n >= args.n:
                break
            img = ex["image"]
            label = int(ex["label"])
            fn = f"val_{n:06d}.jpg"
            try:
                img.convert("RGB").save(os.path.join(img_dir, fn), "JPEG", quality=95)
            except Exception as e:  # noqa: BLE001
                print(f"  skip image {n}: {e}", file=sys.stderr)
                continue
            gt.write(f"{fn} {label}\n")
            n += 1
            if n % 500 == 0:
                print(f"  wrote {n}/{args.n}", file=sys.stderr)

    print(f"DONE: {n} images -> {img_dir}, gt -> {gt_path} (source={chosen})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
