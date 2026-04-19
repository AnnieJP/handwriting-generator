"""
Auto-generate train/val/test splits from IAM XML annotations.

Groups by writer (first part of form ID, e.g. 'a01') so no writer
appears in more than one split — writer-independent evaluation,
matching standard IAM protocol.

Split ratio: 80% train / 10% val / 10% test
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

IAM_ROOT  = Path("data/iam")
XML_DIR   = IAM_ROOT / "xml"
WORDS_DIR = IAM_ROOT / "words"
SPLITS_DIR = IAM_ROOT / "splits"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (remainder)

SEED = 42


def parse_all_words(xml_dir: Path):
    """
    Returns a dict: writer_id -> list of (word_id, text) that have a
    corresponding image file in words/.
    """
    writer_words = defaultdict(list)

    for xml_path in sorted(xml_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        for word in root.iter("word"):
            word_id = word.get("id", "")
            text    = word.get("text", "").strip()
            if not word_id or not text:
                continue

            # Resolve image path
            parts = word_id.split("-")
            if len(parts) < 3:
                continue
            sub1   = parts[0]
            sub2   = "-".join(parts[:2])
            img    = WORDS_DIR / sub1 / sub2 / f"{word_id}.png"
            if not img.exists():
                continue

            writer_id = parts[0]          # e.g. 'a01'
            writer_words[writer_id].append(word_id)

    return writer_words


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing XML files …")
    writer_words = parse_all_words(XML_DIR)

    writers = sorted(writer_words.keys())
    random.seed(SEED)
    random.shuffle(writers)

    n        = len(writers)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    train_writers = writers[:n_train]
    val_writers   = writers[n_train : n_train + n_val]
    test_writers  = writers[n_train + n_val :]

    def collect(ws):
        ids = []
        for w in ws:
            ids.extend(writer_words[w])
        return ids

    train_ids = collect(train_writers)
    val_ids   = collect(val_writers)
    test_ids  = collect(test_writers)

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = SPLITS_DIR / f"{name}.txt"
        path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        print(f"  {name:5s}: {len(ids):6,} words  ({path})")

    total = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"\nTotal: {total:,} word samples across {n} writers.")
    print("Splits saved to data/iam/splits/")


if __name__ == "__main__":
    main()
