"""
Dataset loaders for IAM Handwriting Database, CROHME, and user handwriting samples.

IAM expected structure:
  data/iam/
    words/          # word-level images: a01/a01-000u/a01-000u-00-00.png
    lines/          # line-level images: a01/a01-000u/a01-000u-00.png
    xml/            # XML annotations: a01-000u.xml
    splits/
      train.txt     # one word/line id per line
      val.txt
      test.txt

CROHME expected structure:
  data/crohme/
    train/          # InkML files + PNG renders
    test/

User samples:
  data/samples/
    sample_01.png   # handwriting image (one line)
    sample_01.txt   # transcription
"""

import os
import glob
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    from lxml import etree
except ImportError:
    import xml.etree.ElementTree as etree


CHARSET = (
    " !\"#&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 = blank/pad
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank


def encode_text(text: str) -> List[int]:
    return [CHAR2IDX.get(c, 0) for c in text]


def decode_text(indices: List[int]) -> str:
    return "".join(IDX2CHAR.get(i, "") for i in indices if i > 0)


def collate_fn(batch):
    """Custom collate that pads images to the same width and stacks labels."""
    images, labels, lengths = zip(*batch)
    max_w = max(img.shape[-1] for img in images)
    padded = torch.zeros(len(images), images[0].shape[0], images[0].shape[1], max_w)
    for i, img in enumerate(images):
        padded[i, :, :, : img.shape[-1]] = img
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_flat = torch.tensor(
        [idx for label in labels for idx in label], dtype=torch.long
    )
    return padded, labels_flat, label_lengths


class IAMDataset(Dataset):
    """
    Loads the IAM Handwriting Database at word or line level.

    Args:
        root: Path to the iam data directory.
        split: 'train', 'val', or 'test'.
        level: 'words' or 'lines'.
        img_height: Fixed height to resize images to.
        max_width: Maximum image width (wider images are discarded).
        augment: Whether to apply data augmentation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        level: str = "words",
        img_height: int = 64,
        max_width: int = 768,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.level = level
        self.img_height = img_height
        self.max_width = max_width
        self.augment = augment and (split == "train")

        split_file = self.root / "splits" / f"{split}.txt"
        self.samples: List[Tuple[Path, str]] = []

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                "Please download the IAM dataset and place split files in data/iam/splits/"
            )

        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]

        xml_dir = self.root / "xml"
        transcriptions = self._load_transcriptions(xml_dir)

        image_dir = self.root / level
        for item_id in ids:
            text = transcriptions.get(item_id)
            if text is None:
                continue
            img_path = self._resolve_image_path(image_dir, item_id)
            if img_path and img_path.exists():
                self.samples.append((img_path, text))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found for split='{split}'. "
                "Verify IAM dataset structure."
            )

    def _load_transcriptions(self, xml_dir: Path) -> Dict[str, str]:
        transcriptions: Dict[str, str] = {}
        for xml_file in xml_dir.glob("*.xml"):
            try:
                tree = etree.parse(str(xml_file))
                root = tree.getroot()
                for line in root.iter("line"):
                    line_id = line.get("id", "")
                    text = line.get("text", "")
                    text = text.replace("|", " ")
                    transcriptions[line_id] = text
                    for word in line.iter("word"):
                        word_id = word.get("id", "")
                        word_text = word.get("text", "")
                        transcriptions[word_id] = word_text
            except Exception:
                continue
        return transcriptions

    def _resolve_image_path(self, image_dir: Path, item_id: str) -> Optional[Path]:
        parts = item_id.split("-")
        if len(parts) >= 3:
            sub1 = parts[0]
            sub2 = "-".join(parts[:2])
            p = image_dir / sub1 / sub2 / f"{item_id}.png"
            return p
        return None

    def _transform(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        aspect = img.width / img.height
        new_w = min(int(self.img_height * aspect), self.max_width)
        img = img.resize((new_w, self.img_height), Image.BICUBIC)

        if self.augment:
            transforms = T.Compose(
                [
                    T.RandomAffine(
                        degrees=2,
                        translate=(0.02, 0.02),
                        scale=(0.95, 1.05),
                        fill=255,
                    ),
                    T.ColorJitter(brightness=0.3, contrast=0.3),
                ]
            )
            img = transforms(img)

        img_tensor = T.ToTensor()(img)
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = Image.open(img_path)
        img_tensor = self._transform(img)
        label = encode_text(text)
        return img_tensor, label, len(label)

    def get_dataloader(self, batch_size: int, num_workers: int = 4) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=(self.augment),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


class UserSamplesDataset(Dataset):
    """
    Loads user-provided handwriting samples for style extraction.

    Each image in samples_dir is paired with a .txt file of the same
    stem containing the ground-truth transcription.

    Args:
        samples_dir: Directory containing image + transcript pairs.
        img_height: Fixed height to resize images to.
        max_width: Maximum image width.
    """

    def __init__(
        self,
        samples_dir: str,
        img_height: int = 64,
        max_width: int = 768,
    ):
        self.samples_dir = Path(samples_dir)
        self.img_height = img_height
        self.max_width = max_width
        self.samples: List[Tuple[Path, str]] = []

        for img_path in sorted(self.samples_dir.glob("*.png")) + sorted(
            self.samples_dir.glob("*.jpg")
        ):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8").strip()
            else:
                text = ""
            self.samples.append((img_path, text))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No handwriting samples found in '{samples_dir}'.\n"
                "Add PNG/JPG images (and optional .txt transcriptions) to that directory."
            )

    def _transform(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        aspect = img.width / img.height
        new_w = min(int(self.img_height * aspect), self.max_width)
        img = img.resize((new_w, self.img_height), Image.BICUBIC)
        img_tensor = T.ToTensor()(img)
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = Image.open(img_path)
        img_tensor = self._transform(img)
        label = encode_text(text)
        return img_tensor, label, len(label)

    def get_style_batch(self, n: int = 8, device: str = "cpu") -> torch.Tensor:
        """Return a batch of up to n style reference images."""
        indices = random.sample(range(len(self)), min(n, len(self)))
        imgs = [self[i][0] for i in indices]
        max_w = max(img.shape[-1] for img in imgs)
        batch = torch.zeros(len(imgs), imgs[0].shape[0], self.img_height, max_w)
        for i, img in enumerate(imgs):
            batch[i, :, :, : img.shape[-1]] = img
        return batch.to(device)


class CROHMEDataset(Dataset):
    """
    Minimal loader for CROHME handwritten math expression dataset.

    Expects PNG renders and label files in:
        crohme_root/train/  or  crohme_root/test/
    where each .png is paired with a .txt label file.

    If only InkML files are present, use the provided render_inkml.py
    utility (not included) to generate PNG renders first.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_height: int = 64,
        max_width: int = 512,
        augment: bool = True,
    ):
        self.root = Path(root) / split
        self.img_height = img_height
        self.max_width = max_width
        self.augment = augment and (split == "train")
        self.samples: List[Tuple[Path, str]] = []

        for img_path in sorted(self.root.glob("*.png")):
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            text = txt_path.read_text(encoding="utf-8").strip()
            self.samples.append((img_path, text))

    def _transform(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        aspect = img.width / img.height
        new_w = min(int(self.img_height * aspect), self.max_width)
        img = img.resize((new_w, self.img_height), Image.BICUBIC)

        if self.augment:
            tf = T.Compose(
                [
                    T.RandomAffine(degrees=1, translate=(0.01, 0.01), fill=255),
                ]
            )
            img = tf(img)

        t = T.ToTensor()(img)
        t = (t - 0.5) / 0.5
        return t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = Image.open(img_path)
        img_tensor = self._transform(img)
        label = encode_text(text)
        return img_tensor, label, len(label)


class CombinedDataset(Dataset):
    """
    Combines IAM and CROHME datasets into a single dataset.
    Optionally adds user samples with higher sampling weight.
    """

    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.total = sum(self.lengths)
        self.offsets = [0] + list(np.cumsum(self.lengths[:-1]))

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        for i, (offset, length) in enumerate(zip(self.offsets, self.lengths)):
            if idx < offset + length:
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range for CombinedDataset")
