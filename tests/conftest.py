"""測試用的迷你假 VOC 資料集（隨機影像），讓測試不需要真的資料也能在 CPU 上跑完。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.voc import VOC_CLASSES

_XML_TEMPLATE = """<annotation>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
{objects}
</annotation>"""
_OBJ_TEMPLATE = """  <object>
    <name>{name}</name><difficult>{difficult}</difficult>
    <bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>
  </object>"""


def _make_year(root: Path, year: str, num_images: int, rng: random.Random) -> None:
    base = root / f"VOC{year}"
    for sub in ("JPEGImages", "Annotations", "SegmentationClass", "ImageSets/Main", "ImageSets/Segmentation"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    ids = [f"{year}_{i:06d}" for i in range(num_images)]
    palette = np.zeros((256, 3), dtype=np.uint8).flatten().tolist()
    for image_id in ids:
        w, h = rng.randint(100, 140), rng.randint(80, 120)
        pixels = np.random.default_rng(rng.randint(0, 10**6)).integers(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(base / "JPEGImages" / f"{image_id}.jpg")

        mask = np.zeros((h, w), dtype=np.uint8)
        objects = []
        for _ in range(rng.randint(1, 3)):
            x1, y1 = rng.randint(1, w // 2), rng.randint(1, h // 2)
            x2, y2 = rng.randint(x1 + 10, w), rng.randint(y1 + 10, h)
            label = rng.randint(1, len(VOC_CLASSES))
            objects.append(
                _OBJ_TEMPLATE.format(
                    name=VOC_CLASSES[label - 1], difficult=int(rng.random() < 0.2), x1=x1, y1=y1, x2=x2, y2=y2
                )
            )
            mask[y1 - 1 : y2, x1 - 1 : x2] = label
            mask[y1 - 1, x1 - 1 : x2] = 255  # 模擬邊界
        (base / "Annotations" / f"{image_id}.xml").write_text(
            _XML_TEMPLATE.format(w=w, h=h, objects="\n".join(objects))
        )
        png = Image.fromarray(mask, mode="P")
        png.putpalette(palette)
        png.save(base / "SegmentationClass" / f"{image_id}.png")

    half = num_images // 2
    for task in ("Main", "Segmentation"):
        sets = {"train": ids[:half], "val": ids[half:], "trainval": ids, "test": ids[half:]}
        for split, split_ids in sets.items():
            (base / "ImageSets" / task / f"{split}.txt").write_text("\n".join(split_ids) + "\n")


@pytest.fixture(scope="session")
def fake_voc(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("voc") / "VOCdevkit"
    rng = random.Random(0)
    _make_year(root, "2007", 12, rng)
    _make_year(root, "2012", 12, rng)
    return root
