"""Training dataset: full-res filament-centred crops (primary) or whole-disk resized images (fallback)."""
import json
import random

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import CACHE, TRAIN_IMG

MEAN, STD = 0.5, 0.25


def normalize(img_u8: np.ndarray) -> np.ndarray:
    return (img_u8.astype(np.float32) / 255.0 - MEAN) / STD


class LowFreqField(A.ImageOnlyTransform):
    """Smooth multiplicative gain field: clouds/haze/limb-darkening residuals (ground-based GONG data)."""

    def __init__(self, amp=0.15, p=0.3):
        super().__init__(p=p)
        self.amp = amp

    def apply(self, img, **params):
        h, w = img.shape[:2]
        g = np.random.uniform(-1, 1, (4, 4)).astype(np.float32)
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        out = img.astype(np.float32) * (1 + self.amp * g)
        return np.clip(out, 0, 255).astype(np.uint8)


def build_aug(strong=True):
    spatial = [A.RandomRotate90(p=1.0), A.HorizontalFlip(p=0.5)]
    if strong:
        spatial.append(A.Affine(scale=(0.85, 1.15), rotate=(-180, 180), border_mode=cv2.BORDER_CONSTANT,
                                fill=0, fill_mask=0, p=0.3))
    photo = [
        A.RandomBrightnessContrast(0.2, 0.2, p=0.6),
        A.RandomGamma((80, 125), p=0.3),
        LowFreqField(0.15, p=0.3),
        A.GaussianBlur(sigma_limit=(0.1, 1.5), p=0.2),
        A.GaussNoise(std_range=(0.0, 0.012), p=0.2),
        A.ImageCompression(quality_range=(70, 95), p=0.3),
    ]
    return A.Compose(spatial + photo)


class FilamentDataset(Dataset):
    def __init__(self, stems, mode="crop", crop=768, size=1024, samples_per_image=4, p_fil=0.7, aug=True):
        self.stems = list(stems)
        self.mode, self.crop, self.size = mode, crop, size
        self.n = len(self.stems) * samples_per_image
        self.p_fil = p_fil
        self.centers = json.loads((CACHE / "centers.json").read_text())
        self.aug = build_aug() if aug else None

    def __len__(self):
        return self.n

    def _load(self, stem):
        img = cv2.imread(str(TRAIN_IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        tgt = cv2.imread(str(CACHE / "targets" / f"{stem}.png"))[..., ::-1]
        return img, np.ascontiguousarray(tgt)

    def __getitem__(self, i):
        stem = self.stems[i % len(self.stems)]
        img, tgt = self._load(stem)
        if self.mode == "crop":
            c = self.crop
            cen = self.centers[stem]
            if cen and random.random() < self.p_fil:
                cx, cy = random.choice(cen)
                cx += random.uniform(-0.35, 0.35) * c
                cy += random.uniform(-0.35, 0.35) * c
            else:
                cx, cy = random.uniform(c / 2, 2048 - c / 2), random.uniform(c / 2, 2048 - c / 2)
            x0 = int(np.clip(cx - c / 2, 0, 2048 - c))
            y0 = int(np.clip(cy - c / 2, 0, 2048 - c))
            img, tgt = img[y0:y0 + c, x0:x0 + c], tgt[y0:y0 + c, x0:x0 + c]
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            tgt = cv2.resize(tgt, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.aug is not None:
            r = self.aug(image=img, mask=tgt)
            img, tgt = r["image"], r["mask"]
        x = torch.from_numpy(normalize(img))[None]
        y = torch.from_numpy(tgt.astype(np.float32) / 255.0).permute(2, 0, 1)
        return x, y
