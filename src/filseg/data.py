"""Dataset indexing: COCO annotations grouped by image file and annotator entry."""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG = DATA / "train" / "train_images"
TEST_IMG = DATA / "test" / "test_images"
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CACHE = ROOT / "cache"


@dataclass
class Entry:
    """One annotator's annotation of one image (scored as a separate GT entry)."""
    entry_id: str          # "<annotator>-<stem>", the key the official scorer groups on
    stem: str              # image name without extension
    anns: list = field(default_factory=list)


def load_index(ann_path: Path = ANN):
    """Returns {stem: [Entry, ...]} for the training set."""
    d = json.loads(Path(ann_path).read_text())
    by_id = defaultdict(list)
    for a in d["annotations"]:
        by_id[a["image_id"]].append(a)
    files = defaultdict(list)
    for im in d["images"]:
        stem = im["file_name"].rsplit(".", 1)[0]
        files[stem].append(Entry(im["id"], stem, by_id[im["id"]]))
    return dict(files)


def instrument(stem: str) -> str:
    return stem[14:16]


def date(stem: str) -> str:
    return stem[:8]
