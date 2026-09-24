"""只读公共标注。此模块不被正常推理导入。"""

import argparse
from collections import Counter
import json
import re
import shutil
import yaml
import cv2
import numpy as np
from meter_reader.config import load_config, ROOT
from meter_reader.geometry import (
    read_image,
    write_image,
    rectify,
    region_crop,
    annotate,
)


def export_pose(cfg):
    """导出两类区域+4个语义有序关键点；不提供针尖/像素掩膜或推理真值。"""
    root = cfg["paths"]["prepared"] / "pose"
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val"):
        rows = json.loads(
            (cfg["paths"]["prepared"] / f"{split}.json").read_text(encoding="utf-8")
        )
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        listing, skipped, regions, invisible = [], [], 0, 0
        for i, row in enumerate(rows):
            image = read_image(ROOT / row["image"])
            h, w = image.shape[:2]
            lines = []
            for region in row["regions"]:
                q = np.array(region["quad"], np.float32) / [w, h]
                if not np.isfinite(q).all():
                    break
                lo, hi = np.clip(q.min(0), 0, 1), np.clip(q.max(0), 0, 1)
                if (hi <= lo).any():
                    break
                values = [*((lo + hi) / 2), *(hi - lo)]
                for x, y in q:
                    # Out-of-frame corners have no visible pixel; do not invent a clipped corner.
                    visible = 0 <= x <= 1 and 0 <= y <= 1
                    values.extend([x, y, 2] if visible else [0, 0, 0])
                    invisible += int(not visible)
                kind = 0 if region["kind"] == "pointer" else 1
                lines.append(str(kind) + " " + " ".join(f"{v:.8f}" for v in values))
            if len(lines) != len(row["regions"]) or not lines:
                skipped.append(
                    dict(image=row["image"], reason="框/四角无效，整图不加入pose划分")
                )
                continue
            source = ROOT / row["image"]
            name = f"{i:05d}{source.suffix.lower()}"
            destination = images / name
            shutil.copy2(source, destination)
            (labels / f"{i:05d}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            listing.append(f"./images/{split}/{name}")
            regions += len(lines)
        # Explicit lists exclude stale files from a previous export.
        (root / f"{split}.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
        summary[split] = dict(
            images=len(listing),
            regions=regions,
            invisible_corners=invisible,
            skipped=skipped,
        )
    # Regenerate this small YAML after moving the project; source paths remain in config.yaml.
    recipe = dict(
        path=str(root.resolve()),
        train="train.txt",
        val="val.txt",
        names={0: "pointer", 1: "wheel"},
        kpt_shape=[4, 3],
    )
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    summary["note"] = (
        "四角语义顺序保留；图外角点以visibility=0屏蔽坐标监督，未夹紧成假角点。不是针尖；训练须关闭镜像。迁移后重新运行--pose-only更新dataset.yaml。尚未训练联合模型。"
    )
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                split: {**value, "skipped": len(value["skipped"])}
                for split, value in summary.items()
                if split != "note"
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def prepare(cfg):
    source, output = cfg["paths"]["public_data"], cfg["paths"]["prepared"]
    output.mkdir(parents=True, exist_ok=True)
    skip = {
        x.strip()
        for x in cfg["paths"]["skip_images"].read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    summaries = {}
    for split in ("train", "val"):
        records, anns, images, wheel_labels = [], [], [], []
        count = Counter()
        for line_no, line in enumerate(
            (source / f"label_{split}.txt").read_text(encoding="utf-8").splitlines()
        ):
            try:
                filename, raw_regions, truth = line.split("\t")
                if filename in skip:
                    count["skipped_known_images"] += 1
                    continue
                path = source / "images" / filename.replace("\\", "/").split("/")[-1]
                image = read_image(path)
                h, w = image.shape[:2]
                regions = []
                for item in raw_regions.split(";"):
                    if not item:
                        continue
                    coords, text = item.split("/")
                    quad = np.asarray(
                        [float(x) for x in coords.split(",")], np.float32
                    ).reshape(4, 2)
                    kind = (
                        "pointer"
                        if re.fullmatch(r"[0-9][+-]?", text)
                        else "wheel" if re.fullmatch(r"(?:[0-9]v?){3,}", text) else None
                    )
                    if kind is None:
                        raise ValueError(f"未知区域标签 {text!r}")
                    # Do not clamp individual corners: that changes perspective and direction.
                    crop, _ = rectify(image, quad)
                    lo = np.maximum(quad.min(0), [0, 0])
                    hi = np.minimum(quad.max(0), [w, h])
                    if np.min(hi - lo) < 4:
                        raise ValueError("框位于图外")
                    regions.append(
                        dict(
                            kind=kind,
                            quad=quad.tolist(),
                            box=[*lo.tolist(), *hi.tolist()],
                            visible_label=text,
                        )
                    )
                # The author's pointer sequence is high -> low. Derive supervision ONLY here.
                # Exclude truncated/ambiguous counts from place-value supervision.
                truth = truth.strip()
                decimal_count = (
                    len(truth.split(".")[1])
                    if re.fullmatch(r"\d+\.\d+", truth)
                    else None
                )
                pointers = [r for r in regions if r["kind"] == "pointer"]
                wheels = [r for r in regions if r["kind"] == "wheel"]
                supported = decimal_count is not None and (
                    (len(pointers) == 8 and not wheels)
                    or (
                        len(wheels) == 1
                        and len(pointers) == decimal_count
                        and decimal_count in (3, 4)
                    )
                )
                for j, r in enumerate(pointers):
                    r["exponent"] = (
                        len(pointers) - j - 1 - decimal_count if supported else None
                    )
                for r in wheels:
                    r["exponent"] = 0 if supported else None
                idx = len(records)
                images.append(dict(id=idx, file_name=path.name, width=w, height=h))
                for j, r in enumerate(regions):
                    box = r["box"]
                    x1, y1, x2, y2 = box
                    anns.append(
                        dict(
                            id=len(anns) + 1,
                            image_id=idx,
                            category_id=0 if r["kind"] == "pointer" else 1,
                            bbox=[x1, y1, x2 - x1, y2 - y1],
                            area=(x2 - x1) * (y2 - y1),
                            iscrowd=0,
                        )
                    )
                    crop, _ = rectify(image, r["quad"])
                    name = f"{split}_{idx:05d}_{j:02d}.jpg"
                    r["crop"] = f"crops/{name}"
                    write_image(output / r["crop"], crop)
                    context, offset, scale = region_crop(
                        image, box, cfg["geometry"]["padding"]
                    )
                    r["context"] = f"geometry/{name}"
                    r["local_quad"] = (
                        (np.asarray(r["quad"]) - offset) / scale
                    ).tolist()
                    write_image(output / r["context"], cv2.resize(context, (128, 128)))
                    count[r["kind"]] += 1
                    if r["kind"] == "wheel":
                        # v remains an opaque visible-state token, not an increment instruction.
                        wheel_labels.append(r["crop"] + "\t" + r["visible_label"])
                        count["wheel_with_v"] += int("v" in r["visible_label"])
                records.append(
                    dict(
                        image=str(path.relative_to(ROOT)).replace("\\", "/"),
                        regions=regions,
                        full_label=truth,
                    )
                )
                if idx < 4 or (wheels and count["preview_wheel"] < 4):
                    write_image(
                        output / "previews" / f"{split}_{idx:05d}.jpg",
                        annotate(image, regions),
                    )
                    if wheels:
                        count["preview_wheel"] += 1
            except (ValueError, OSError, cv2.error) as e:
                count["invalid_images"] += 1
                print(f"{split}:{line_no+1} 跳过: {e}", flush=True)
        (output / f"{split}.json").write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        (output / f"{split}_coco.json").write_text(
            json.dumps(
                dict(
                    images=images,
                    annotations=anns,
                    categories=[dict(id=0, name="pointer"), dict(id=1, name="wheel")],
                )
            ),
            encoding="utf-8",
        )
        (output / f"wheel_{split}.txt").write_text(
            "\n".join(wheel_labels), encoding="utf-8"
        )
        summaries[split] = dict(images=len(records), **count)
        print(split, summaries[split], flush=True)
    (output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="从已准备的公共划分导出有序四角数据，不重新生成裁剪",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.pose_only:
        export_pose(cfg)
    else:
        prepare(cfg)
