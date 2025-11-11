import os
import os.path as osp
import json
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils


def ensure_dir(p: str):
    if not osp.exists(p):
        os.makedirs(p, exist_ok=True)


def draw_rect(img: Image.Image, xyxy: Tuple[int, int, int, int], color=(255, 0, 0)):
    d = ImageDraw.Draw(img)
    d.rectangle(xyxy, fill=color)


def draw_circle(img: Image.Image, center: Tuple[int, int], r: int, color=(0, 255, 0)):
    x, y = center
    d = ImageDraw.Draw(img)
    d.ellipse((x - r, y - r, x + r, y + r), fill=color)


def mask_from_rect(h: int, w: int, xyxy: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = xyxy
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def mask_from_circle(h: int, w: int, center: Tuple[int, int], r: int):
    Y, X = np.ogrid[:h, :w]
    cy, cx = center[1], center[0]
    dist = (X - cx) ** 2 + (Y - cy) ** 2
    mask = (dist <= r * r).astype(np.uint8)
    return mask


def rle_encode(mask: np.ndarray):
    rle = maskUtils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    root = "data"
    img_dir = osp.join(root, "toy", "images")
    out_jsonl = osp.join(root, "refcoco", "converted", "refcoco_train.jsonl")

    ensure_dir(img_dir)
    ensure_dir(osp.dirname(out_jsonl))

    H, W = 256, 256

    # Image 1: red rectangle
    img1 = Image.new("RGB", (W, H), (0, 0, 0))
    rect1 = (50, 60, 150, 160)
    draw_rect(img1, rect1, (255, 0, 0))
    img1_path = osp.join(img_dir, "img_0001.jpg")
    img1.save(img1_path, quality=95)
    mask1 = mask_from_rect(H, W, rect1)
    rle1 = rle_encode(mask1)

    # Image 2: green circle
    img2 = Image.new("RGB", (W, H), (0, 0, 0))
    center2, r2 = (160, 120), 40
    draw_circle(img2, center2, r2, (0, 255, 0))
    img2_path = osp.join(img_dir, "img_0002.jpg")
    img2.save(img2_path, quality=95)
    mask2 = mask_from_circle(H, W, center2, r2)
    rle2 = rle_encode(mask2)

    samples = [
        dict(
            img_path=img1_path,
            height=H,
            width=W,
            text="the red rectangle",
            instances=[dict(bbox=[rect1[0], rect1[1], rect1[2] - rect1[0], rect1[3] - rect1[1]],
                            bbox_label=0,
                            ignore_flag=0,
                            mask=rle1,
                            is_thing=True)]
        ),
        dict(
            img_path=img2_path,
            height=H,
            width=W,
            text="the green circle",
            instances=[dict(bbox=[center2[0] - r2, center2[1] - r2, 2 * r2, 2 * r2],
                            bbox_label=0,
                            ignore_flag=0,
                            mask=rle2,
                            is_thing=True)]
        )
    ]

    with open(out_jsonl, "w", encoding="utf-8") as w:
        for s in samples:
            w.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Toy images written to: {img_dir}")
    print(f"Toy ref_seg jsonl written to: {out_jsonl}")
    print("You can now point RefSegDataset.ann_file to this jsonl to debug the pipeline.")


if __name__ == "__main__":
    main()


