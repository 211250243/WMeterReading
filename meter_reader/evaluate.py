"""公共验证：自动整图评估与真值裁剪诊断分开保存，不可互相替代。"""

import argparse
from decimal import Decimal, InvalidOperation
import json
import random
import cv2
import numpy as np
import torch
from meter_reader.config import load_config, ROOT, resolve, stored_path
from meter_reader.geometry import read_image, write_image, region_crop, rectify


def compare_regions(cfg, rows, output):
    """两种自动裁剪先独立推理，之后才匹配标注；不计算整表值。"""
    from meter_reader.detector import Detector
    from meter_reader.region_model import GeometryModel
    from meter_reader.wheel import WheelReader
    from meter_reader.pointer import PointerReader
    from torchvision.ops import box_iou

    detector, geometry = Detector(cfg), GeometryModel(cfg)
    wheel, pointer = WheelReader(cfg), PointerReader(cfg)
    counts = {
        kind: dict(
            truth=0,
            detected=0,
            matched_iou50=0,
            quad_valid=0,
            direct_correct=0,
            rectified_correct=0,
            quad_iou80=0,
            direction_over15=0,
        )
        for kind in ("wheel", "pointer")
    }
    details, previews = [], {"wheel": 0, "pointer": 0}
    for row in rows:
        image = read_image(ROOT / row["image"])
        predicted = detector.predict(image)
        outputs = []
        for region in predicted:
            kind = region["kind"]
            counts[kind]["detected"] += 1
            direct, _, _ = region_crop(image, region["box"], 0)
            item = dict(kind=kind, box=region["box"], direct=direct)
            recognize = (
                wheel.predict
                if kind == "wheel"
                else lambda crop: pointer.individuals([crop])[0]
            )
            item["direct_prediction"] = recognize(direct)
            try:
                crop = geometry.predict(image, region)
                item.update(
                    rectified=crop,
                    quad=region["quad"],
                    rectified_prediction=recognize(crop),
                )
            except ValueError as error:
                item["geometry_error"] = str(error)
            outputs.append(item)
        # Ground truth is only used below this line, for correspondence/diagnosis.
        for kind in counts:
            truths = [r for r in row["regions"] if r["kind"] == kind]
            predictions = [r for r in outputs if r["kind"] == kind]
            counts[kind]["truth"] += len(truths)
            pairs = {}
            if truths and predictions:
                overlaps = box_iou(
                    torch.tensor([r["box"] for r in predictions]),
                    torch.tensor([r["box"] for r in truths]),
                )
                for _ in range(min(overlaps.shape)):
                    if overlaps.max() < 0.5:
                        break
                    a, b = divmod(int(overlaps.argmax()), len(truths))
                    pairs[b] = a
                    overlaps[a, :] = 0
                    overlaps[:, b] = 0
            for j, truth in enumerate(truths):
                item = dict(
                    image=row["image"],
                    kind=kind,
                    label=truth["visible_label"],
                    matched=j in pairs,
                )
                if j not in pairs:
                    details.append(item)
                    continue
                pred = predictions[pairs[j]]
                counts[kind]["matched_iou50"] += 1
                key = "visible_text" if kind == "wheel" else "raw_class"
                for variant in ("direct", "rectified"):
                    result = pred.get(variant + "_prediction", {})
                    item[variant] = result.get(key)
                    counts[kind][variant + "_correct"] += (
                        result.get(key) == truth["visible_label"]
                    )
                if "quad" in pred:
                    counts[kind]["quad_valid"] += 1
                    q, t = np.float32(pred["quad"]), np.float32(truth["quad"])
                    intersection, _ = cv2.intersectConvexConvex(q, t)
                    item["quad_iou"] = float(
                        intersection
                        / max(
                            1.0, cv2.contourArea(q) + cv2.contourArea(t) - intersection
                        )
                    )
                    angles = [np.arctan2(*(p[1] - p[0])[::-1]) for p in (q, t)]
                    item["direction_error"] = float(
                        abs((np.degrees(angles[0] - angles[1]) + 180) % 360 - 180)
                    )
                    counts[kind]["quad_iou80"] += item["quad_iou"] >= 0.8
                    counts[kind]["direction_over15"] += item["direction_error"] > 15
                else:
                    item["geometry_error"] = pred.get("geometry_error")
                if previews[kind] < 8:
                    reference, _ = rectify(image, truth["quad"])
                    cells = []
                    for title, crop in [
                        (f'direct: {item["direct"]}', pred["direct"]),
                        (f'quad: {item["rectified"]}', pred.get("rectified")),
                        (f'GT diagnostic: {truth["visible_label"]}', reference),
                    ]:
                        cell = np.full((185, 300, 3), 245, np.uint8)
                        cv2.putText(
                            cell,
                            title,
                            (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                        )
                        if crop is not None:
                            h, w = crop.shape[:2]
                            scale = min(290 / w, 145 / h)
                            resized = cv2.resize(
                                crop, (max(1, int(w * scale)), max(1, int(h * scale)))
                            )
                            cell[
                                30 : 30 + resized.shape[0], 5 : 5 + resized.shape[1]
                            ] = resized
                        cells.append(cell)
                    preview = output / f"{kind}_{previews[kind]:02d}.jpg"
                    write_image(preview, np.hstack(cells))
                    item["preview"] = stored_path(preview)
                    previews[kind] += 1
                details.append(item)
        print(f"裁剪对照 {row['image']}", flush=True)
    summary = dict(
        mode="crop_comparison_diagnostic",
        images=len(rows),
        counts=counts,
        note="correct为单区域可见标签完全匹配，分母用truth包含漏检/校正失败；指针仅CNN原始类别。直接框不保证零位方向；GT只用于事后匹配/对照，非整表准确率。",
    )
    return details, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=["automatic", "crops", "regions"], default="automatic"
    )
    p.add_argument("--config")
    p.add_argument("--device")
    p.add_argument(
        "--wheels-only", action="store_true", help="仅选择含字轮的公共验证图片做诊断"
    )
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--output")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    torch.set_num_threads(cfg["threads"])
    rows = json.loads(
        (cfg["paths"]["prepared"] / "val.json").read_text(encoding="utf-8")
    )
    if args.wheels_only:
        rows = [
            row for row in rows if any(r["kind"] == "wheel" for r in row["regions"])
        ]
    if args.limit:
        rows = random.Random(42).sample(rows, min(args.limit, len(rows)))
    output = (
        resolve(args.output)
        if args.output
        else ROOT / "outputs" / f"evaluation_{args.mode}"
    )
    output.mkdir(parents=True, exist_ok=True)
    details = []
    if args.mode == "automatic":
        from meter_reader.pipeline import Pipeline
        from meter_reader.predict import batch

        pipeline = Pipeline(cfg)
        # Only these image paths cross the inference boundary.
        results = batch([ROOT / row["image"] for row in rows], pipeline, output)
        exact = accepted = candidates = 0
        for row, result in zip(rows, results):
            correct = False
            candidate_correct = False
            try:
                correct = result["reading"] is not None and Decimal(
                    result["reading"]
                ) == Decimal(row["full_label"])
                candidate_correct = result.get(
                    "candidate_reading"
                ) is not None and Decimal(result["candidate_reading"]) == Decimal(
                    row["full_label"]
                )
            except InvalidOperation:
                pass
            exact += correct
            accepted += result["reading"] is not None
            candidates += candidate_correct
            details.append(
                dict(
                    image=row["image"],
                    truth=row["full_label"],
                    reading=result["reading"],
                    candidate=result.get("candidate_reading"),
                    correct=correct,
                    reasons=result["reasons"],
                )
            )
        summary = dict(
            mode="automatic_image_only",
            total=len(rows),
            complete_correct=exact,
            complete_wrong=accepted - exact,
            accepted=accepted,
            needs_review=len(rows) - accepted,
            complete_accuracy=exact / max(1, len(rows)),
            candidate_correct=candidates,
            note="分母包含所有图片；作者划分有相似固定机位，仅作公开基线，非独立业务准确率",
        )
    elif args.mode == "regions":
        details, summary = compare_regions(cfg, rows, output)
    else:
        from meter_reader.wheel import WheelReader
        from meter_reader.pointer import PointerReader

        wheel = WheelReader(cfg)
        pointer = PointerReader(cfg)
        wc = wn = pc = pn = 0
        for i, row in enumerate(rows):
            pointers = [r for r in row["regions"] if r["kind"] == "pointer"]
            wheels = [r for r in row["regions"] if r["kind"] == "wheel"]
            detail = dict(
                image=row["image"], mode="ground_truth_crops_diagnostic", wheels=[]
            )
            for r in wheels:
                result = wheel.predict(read_image(cfg["paths"]["prepared"] / r["crop"]))
                wc += result["visible_text"] == r["visible_label"]
                wn += 1
                detail["wheels"].append(
                    dict(truth_visible=r["visible_label"], prediction=result)
                )
                if wn <= 8:
                    write_image(
                        output / f"wheel_{wn:02d}.jpg",
                        read_image(cfg["paths"]["prepared"] / r["crop"]),
                    )
            if pointers:
                result = pointer.predict(
                    [read_image(cfg["paths"]["prepared"] / r["crop"]) for r in pointers]
                )
                truth_digits = (
                    row["full_label"]
                    .replace(".", "")[-len(pointers) :]
                    .zfill(len(pointers))
                )
                pc += result["pointer_digits"] == truth_digits
                pn += 1
                detail.update(pointer_result=result, pointer_truth=truth_digits)
            details.append(detail)
            print(f"真值裁剪诊断 {i+1}/{len(rows)}", flush=True)
        summary = dict(
            mode="ground_truth_crops_diagnostic",
            wheel_windows=wn,
            wheel_visible_correct=wc,
            pointer_sequences=pn,
            pointer_sequences_correct=pc,
            note="使用真值区域和作者顺序，只检查识别模型，不是自动整表结果",
        )
    (output / "comparison.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
