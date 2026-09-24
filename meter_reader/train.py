"""三个直接的训练任务：detector / geometry / wheel。只读作者训练划分。"""

import argparse
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from meter_reader.config import load_config, ROOT, resolve, stored_path
from meter_reader.geometry import read_image, region_crop, augment_regions


def records(cfg, split, limit=0):
    rows = json.loads(
        (cfg["paths"]["prepared"] / f"{split}.json").read_text(encoding="utf-8")
    )
    if limit:
        rows = random.Random(42).sample(rows, min(limit, len(rows)))
    return rows


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def save_epoch(args, state, score):
    """同一训练目录保存last/best，不覆盖当前推理权重。分数统一越大越好。"""
    if not np.isfinite(score):
        raise ValueError("验证指标非有限，不保存权重")
    improved = score > args.best_score
    if improved:
        args.best_score = score
    state.update(best_score=args.best_score, selection=args.selection)
    save(args.output / "last.pt", state)
    if improved:
        save(args.output / "best.pt", state)
    with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(state["training"], ensure_ascii=False) + "\n")


def restore_best(args, previous):
    # Changed validation scope or selection rule cannot inherit an old best score.
    if (
        previous
        and previous.get("selection") == args.selection
        and (args.output / "best.pt").is_file()
    ):
        args.best_score = previous.get("best_score", -float("inf"))


class DetectionData(Dataset):
    def __init__(self, rows, size, augment=False, degrees=10):
        self.rows, self.size, self.augment = rows, size, augment
        self.degrees = degrees

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from meter_reader.detector import image_tensor

        r = self.rows[i]
        image = read_image(ROOT / r["image"])
        h, w = image.shape[:2]
        quads = np.array([x["quad"] for x in r["regions"]], np.float32)
        if self.augment:
            image, quads = augment_regions(image, quads, self.degrees)
            h, w = image.shape[:2]
        lo = np.clip(quads.min(1) / [w, h], 0, 1)
        hi = np.clip(quads.max(1) / [w, h], 0, 1)
        boxes = np.concatenate([(lo + hi) / 2, hi - lo], 1)
        target = {
            "orig_size": torch.tensor([w, h]),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(
                [0 if x["kind"] == "pointer" else 1 for x in r["regions"]],
                dtype=torch.long,
            ),
        }
        return image_tensor(image, self.size), target


def detection_collate(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


def train_detector(cfg, args):
    from meter_reader.detector import build_detector, postprocess

    size = args.size or cfg["detector"]["size"]
    model, criterion, post = build_detector(cfg, size)
    previous = None
    if args.resume or args.init:
        loaded = torch.load(
            args.resume or args.init, map_location="cpu", weights_only=True
        )
        previous = loaded if args.resume else None
        state = loaded["model"]
        if loaded["size"] != size:
            state = {
                k: v
                for k, v in state.items()
                if k not in ("decoder.anchors", "decoder.valid_mask")
            }
            missing = model.load_state_dict(state, strict=False).missing_keys
            if set(missing) - {"decoder.anchors", "decoder.valid_mask"}:
                raise ValueError(missing)
            print(
                f"调整训练分辨率 {loaded['size']} → {size}，重新生成 anchors",
                flush=True,
            )
        else:
            model.load_state_dict(state)
    else:
        state = torch.load(
            cfg["paths"]["detector_pretrained"], map_location="cpu", weights_only=False
        )
        state = state["ema"]["module"] if "ema" in state else state["model"]
        own = model.state_dict()
        state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing = model.load_state_dict(state, strict=False).missing_keys
        print("COCO 初始化；新类别头:", missing, flush=True)
    device = cfg["device"]
    model.to(device)
    criterion.to(device)
    post.to(device).eval()
    args.selection.update(metric="f1_iou80", size=size, postprocess=cfg["detector"])
    restore_best(args, previous)
    # Freeze backbone only on request; full GPU training uses all parameters.
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr or 1e-4,
        weight_decay=1e-4,
    )
    if (
        previous
        and "optimizer" in previous
        and previous["training"].get("freeze_backbone", False) == args.freeze_backbone
    ):
        optimizer.load_state_dict(previous["optimizer"])
        if args.lr:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
    rows = records(cfg, "train", args.limit)
    loader = DataLoader(
        DetectionData(rows, size, True, args.rotation),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=detection_collate,
    )
    val = DataLoader(
        DetectionData(records(cfg, "val", args.val_limit), size),
        batch_size=args.batch,
        num_workers=args.workers,
        collate_fn=detection_collate,
    )
    if not len(loader.dataset) or not len(val.dataset):
        raise ValueError("检测训练/验证没有图片")
    start_epoch = previous.get("epoch", 0) if previous else 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        loss_sum = 0
        start = time.time()
        if args.freeze_backbone:
            model.backbone.eval()
        for step, (x, targets) in enumerate(loader):
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            loss = sum(criterion(model(x.to(device), targets), targets).values())
            if not torch.isfinite(loss):
                raise ValueError(f"检测损失非有限: {loss}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            loss_sum += float(loss.detach())
            if step % 10 == 0:
                print(
                    f"detector epoch={epoch+1} batch={step+1}/{len(loader)} loss={float(loss):.3f} elapsed={time.time()-start:.1f}s",
                    flush=True,
                )
        # Same original-image clipping, threshold and NMS as Detector.predict.
        from torchvision.ops import box_convert, box_iou

        model.eval()
        matched = {0.5: 0, 0.8: 0}
        total = predicted = 0
        with torch.inference_mode():
            for x, targets in val:
                out = model(x.to(device))
                sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
                predictions = postprocess(out, post, sizes, cfg["detector"])
                for result, target in zip(predictions, targets):
                    boxes, labels = result["boxes"].cpu(), result["labels"].cpu()
                    total += len(target["labels"])
                    predicted += len(boxes)
                    if not len(boxes):
                        continue
                    iou = box_iou(
                        boxes,
                        box_convert(target["boxes"], "cxcywh", "xyxy")
                        * target["orig_size"].repeat(2),
                    )
                    iou[labels[:, None] != target["labels"][None, :]] = 0
                    for threshold in matched:
                        available = iou.clone()
                        for _ in range(min(available.shape)):
                            if available.max() < threshold:
                                break
                            a, b = divmod(int(available.argmax()), available.shape[1])
                            matched[threshold] += 1
                            available[a, :] = 0
                            available[:, b] = 0
        info = dict(
            images=len(rows),
            val_images=len(val.dataset),
            epochs=epoch + 1,
            size=size,
            loss=loss_sum / len(loader),
            val_recall=matched[0.5] / max(1, total),
            val_precision=matched[0.5] / max(1, predicted),
            val_recall_iou80=matched[0.8] / max(1, total),
            val_precision_iou80=matched[0.8] / max(1, predicted),
            val_f1_iou80=2 * matched[0.8] / max(1, total + predicted),
            rotation=args.rotation,
            device=device,
            limited_baseline=bool(args.limit),
            freeze_backbone=args.freeze_backbone,
        )
        save_epoch(
            args,
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch + 1,
                size=size,
                queries=100,
                training=info,
            ),
            info["val_f1_iou80"],
        )
        print(info, flush=True)


class GeometryData(Dataset):
    def __init__(self, cfg, rows, augment=False, degrees=10, jitter=0):
        from meter_reader.region_model import LAYOUTS, EXPONENTS

        self.root = cfg["paths"]["prepared"]
        self.augment = augment
        self.degrees, self.jitter = degrees, jitter
        self.padding = cfg["geometry"]["padding"]
        self.rows = []
        for row in rows:
            pointers = [r for r in row["regions"] if r["kind"] == "pointer"]
            wheels = [r for r in row["regions"] if r["kind"] == "wheel"]
            if any(r["exponent"] is None for r in row["regions"]):
                continue
            d = -min(r["exponent"] for r in pointers)
            name = f"wheel{d}" if wheels else f"pointer8d{d}"
            if name not in LAYOUTS:
                continue
            for r in row["regions"]:
                self.rows.append(
                    (
                        {**r, "image": row["image"]},
                        EXPONENTS.index(r["exponent"]),
                        LAYOUTS.index(name),
                    )
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from meter_reader.region_model import tensor

        r, e, l = self.rows[i]
        image = read_image(ROOT / r["image"])
        q = np.array(r["quad"], np.float32)
        if self.augment:
            image, q = augment_regions(image, q, self.degrees)
        lo, hi = q.min(0), q.max(0)
        rng = random if self.augment else random.Random(42 + i)
        size = hi - lo
        shift = (
            np.array([rng.uniform(-self.jitter, self.jitter) for _ in range(2)]) * size
        )
        scale = np.array([1 + rng.uniform(-self.jitter, self.jitter) for _ in range(2)])
        center = (lo + hi) / 2 + shift
        box = np.r_[center - size * scale / 2, center + size * scale / 2]
        image, offset, size = region_crop(image, box, self.padding)
        q = ((q - offset) / size).astype(np.float32)
        return tensor(image), torch.from_numpy(q), e, l


def train_geometry(cfg, args):
    from meter_reader.region_model import RegionNet

    device = cfg["device"]
    model = RegionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or 0.001)
    previous = (
        torch.load(args.resume, map_location=device, weights_only=True)
        if args.resume
        else None
    )
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        if args.lr:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
    elif args.init:
        model.load_state_dict(
            torch.load(args.init, map_location=device, weights_only=True)["model"]
        )
    args.selection.update(
        metric="negative_perturbed_corner_error", jitter=args.box_jitter
    )
    restore_best(args, previous)
    train = GeometryData(
        cfg, records(cfg, "train", args.limit), True, args.rotation, args.box_jitter
    )
    val = GeometryData(cfg, records(cfg, "val", args.val_limit), jitter=args.box_jitter)
    if not len(train) or not len(val):
        raise ValueError("四角训练/验证没有可用区域，请检查数据划分")
    # Windows are only ~3% of regions. Balance them so pointer crops do not
    # dominate the shared direction/corner regression head.
    weights = [
        args.wheel_sampling if r["kind"] == "wheel" else 1.0 for r, _, _ in train.rows
    ]
    loader = DataLoader(
        train,
        batch_size=args.batch,
        sampler=WeightedRandomSampler(weights, len(train), replacement=True),
        num_workers=args.workers,
    )
    valid = DataLoader(val, batch_size=args.batch, num_workers=args.workers)
    first = previous.get("epoch", 0) if previous else 0
    for epoch in range(first, first + args.epochs):
        model.train()
        losses = 0
        for step, (x, q, e, l) in enumerate(loader):
            x, q, e, l = [v.to(device) for v in (x, q, e, l)]
            pq, pe, pl = model(x)
            loss = (
                20 * nn.functional.mse_loss(pq, q)
                + nn.functional.cross_entropy(pe, e)
                + nn.functional.cross_entropy(pl, l)
            )
            if not torch.isfinite(loss):
                raise ValueError("几何损失非有限")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses += float(loss.detach())
            if step % 40 == 0:
                print(
                    f"geometry epoch={epoch+1} batch={step+1}/{len(loader)} loss={float(loss):.3f}",
                    flush=True,
                )
        model.eval()
        error = correct = layout_correct = n = 0
        corner_errors = []
        with torch.inference_mode():
            for x, q, e, l in valid:
                pq, pe, pl = model(x.to(device))
                pq, pe, pl = pq.cpu(), pe.cpu(), pl.cpu()
                error += float(torch.linalg.vector_norm(pq - q, dim=-1).mean(-1).sum())
                corner_errors.extend(
                    torch.linalg.vector_norm(pq - q, dim=-1).mean(-1).tolist()
                )
                correct += int((pe.argmax(-1) == e).sum())
                layout_correct += int((pl.argmax(-1) == l).sum())
                n += len(x)
        info = dict(
            regions=len(train),
            epochs=epoch + 1,
            val_regions=n,
            corner_error=error / max(n, 1),
            weight_accuracy=correct / max(n, 1),
            layout_accuracy=layout_correct / max(n, 1),
            limited_baseline=bool(args.limit),
            wheel_sampling=args.wheel_sampling,
            rotation=args.rotation,
            box_jitter=args.box_jitter,
            validation="固定扰动的标注框诊断；不是自动检测框准确率",
            wheel_corner_error=(
                float(
                    np.mean(
                        [
                            err
                            for err, (r, _, _) in zip(corner_errors, val.rows)
                            if r["kind"] == "wheel"
                        ]
                    )
                )
                if any(r["kind"] == "wheel" for r, _, _ in val.rows)
                else None
            ),
        )
        save_epoch(
            args,
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch + 1,
                training=info,
            ),
            -info["corner_error"],
        )
        print(info, flush=True)


class WheelData(Dataset):
    def __init__(self, cfg, rows, chars, augment=False):
        self.root = cfg["paths"]["prepared"]
        self.chars = chars
        self.augment = augment
        self.rows = [r for row in rows for r in row["regions"] if r["kind"] == "wheel"]
        for r in self.rows:
            if any(c not in chars for c in r["visible_label"]):
                raise ValueError("字典缺少可见标签字符")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from meter_reader.wheel import wheel_tensor

        r = self.rows[i]
        image = read_image(self.root / r["crop"])
        if self.augment:
            # Keep every visible character; add border/scale variation, not random truncation.
            h, w = image.shape[:2]
            image = cv2.copyMakeBorder(
                image,
                random.randint(0, max(1, h // 12)),
                random.randint(0, max(1, h // 12)),
                random.randint(0, max(1, w // 30)),
                random.randint(0, max(1, w // 30)),
                cv2.BORDER_REPLICATE,
            )
            image = np.clip(
                image.astype(np.float32) * random.uniform(0.8, 1.2), 0, 255
            ).astype(np.uint8)
        target = torch.tensor(
            [self.chars.index(c) for c in r["visible_label"]], dtype=torch.long
        )
        return wheel_tensor(image), target, r["visible_label"]


def wheel_collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.cat([b[1] for b in batch]),
        torch.tensor([len(b[1]) for b in batch]),
        [b[2] for b in batch],
    )


def train_wheel(cfg, args):
    from meter_reader.wheel import (
        build_wheel,
        load_pretrained,
        decode,
        wheel_characters,
        load_wheel_state,
    )

    device = cfg["device"]
    previous = (
        torch.load(args.resume, map_location="cpu", weights_only=True)
        if args.resume
        else None
    )
    initial = previous or (
        torch.load(args.init, map_location="cpu", weights_only=True)
        if args.init
        else None
    )
    source_chars = (
        initial.get("characters", wheel_characters(cfg))
        if initial
        else wheel_characters(cfg)
    )
    chars = ["blank", *"0123456789v"] if args.wheel_digits else source_chars
    if previous and chars != source_chars:
        raise ValueError("更换字典请用--init开始新实验，不能resume旧优化器")
    model, chars = build_wheel(cfg, chars)
    if previous:
        model.load_state_dict(previous["model"])
    elif args.init:
        load_wheel_state(model, initial["model"], source_chars, chars)
    else:
        load_pretrained(model, cfg["paths"]["wheel_pretrained"], source_chars, chars)
    if args.freeze_backbone:
        model.encoder.requires_grad_(False)
    model.to(device)
    args.selection.update(metric="visible_exact_accuracy", characters=chars)
    restore_best(args, previous)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr or 1e-5
    )
    if (
        previous
        and previous["training"].get("freeze_backbone", False) == args.freeze_backbone
    ):
        optimizer.load_state_dict(previous["optimizer"])
        if args.lr:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
    train = WheelData(cfg, records(cfg, "train", args.limit), chars, True)
    valid = WheelData(cfg, records(cfg, "val", args.val_limit), chars)
    if not len(train) or not len(valid):
        raise ValueError("划分内没有字轮")
    loader = DataLoader(
        train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=wheel_collate,
    )
    val = DataLoader(
        valid, batch_size=args.batch, num_workers=args.workers, collate_fn=wheel_collate
    )
    first = previous.get("epoch", 0) if previous else 0
    for epoch in range(first, first + args.epochs):
        model.train()
        if args.freeze_backbone:
            model.encoder.eval()
        for step, (x, target, lengths, texts) in enumerate(loader):
            pred = model(x.to(device))
            t = pred.shape[1]
            minimum = [
                len(text) + sum(a == b for a, b in zip(text, text[1:]))
                for text in texts
            ]
            if max(minimum) > t:
                raise ValueError(f"CTC时间步{t}不足以对齐重复字符: {texts}")
            loss = nn.functional.ctc_loss(
                pred.log_softmax(-1).transpose(0, 1),
                target.to(device),
                torch.full((len(x),), t, dtype=torch.long),
                lengths,
                zero_infinity=False,
            )
            if not torch.isfinite(loss):
                raise ValueError("CTC 损失非有限，检查序列宽度与标签")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step % 10 == 0:
                print(
                    f"wheel epoch={epoch+1} batch={step+1}/{len(loader)} loss={float(loss):.3f}",
                    flush=True,
                )
        model.eval()
        correct = n = 0
        with torch.inference_mode():
            for x, _, _, truths in val:
                results = decode(model(x.to(device)), chars)
                correct += sum(
                    r["visible_text"] == truth for r, truth in zip(results, truths)
                )
                n += len(truths)
        info = dict(
            windows=len(train),
            epochs=epoch + 1,
            val_windows=n,
            visible_exact_accuracy=correct / n,
            limited_baseline=bool(args.limit),
            freeze_backbone=args.freeze_backbone,
            character_count=len(chars),
        )
        save_epoch(
            args,
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch + 1,
                training=info,
                characters=chars,
            ),
            info["visible_exact_accuracy"],
        )
        print(info, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["detector", "geometry", "wheel"])
    parser.add_argument("--config")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--limit", type=int, default=0, help="只作有限 CPU 基线；0 使用全部训练集"
    )
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--wheel-sampling", type=float, default=6, help="geometry 中字轮区域的采样权重"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--output", help="独立训练目录；默认outputs/training/<task>")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        help="恢复完整训练状态；不填路径时用输出目录last.pt",
    )
    parser.add_argument("--init", help="仅加载本项目权重开始新实验，不继承优化器/轮次")
    parser.add_argument(
        "--rotation",
        type=float,
        default=10,
        help="检测/几何的小角度增强上限，保持原图基础方向",
    )
    parser.add_argument(
        "--box-jitter", type=float, default=0.08, help="几何框中心/尺度扰动比例"
    )
    parser.add_argument(
        "--wheel-digits",
        action="store_true",
        help="字轮使用数字+v字典，按字符迁移预训练CTC头",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无可用 CUDA")
    if args.epochs < 1 or args.batch < 1 or args.wheel_sampling <= 0:
        raise ValueError("epochs/batch/wheel-sampling 必须大于0")
    if args.resume and args.init:
        raise ValueError("resume恢复训练与init开始新实验不能同时使用")
    if not 0 <= args.rotation <= 180 or not 0 <= args.box_jitter < 0.3:
        raise ValueError("rotation需在0到180度，box-jitter需在0到0.3之间")
    if min(args.limit, args.val_limit, args.workers) < 0 or args.threads < 1:
        raise ValueError("limit/val-limit/workers不能为负，threads必须为正")
    args.output = (
        resolve(args.output) if args.output else cfg["paths"]["training"] / args.task
    )
    if not args.resume and any(
        (args.output / name).exists()
        for name in ("last.pt", "best.pt", "metrics.jsonl")
    ):
        raise ValueError("训练目录已有结果，请指定新--output或--resume")
    if args.resume:
        args.resume = (
            args.output / "last.pt" if args.resume == "last" else resolve(args.resume)
        )
    if args.init:
        args.init = resolve(args.init)
    args.output.mkdir(parents=True, exist_ok=True)
    args.best_score = -float("inf")
    args.selection = dict(
        task=args.task,
        prepared=stored_path(cfg["paths"]["prepared"]),
        val_limit=args.val_limit,
    )
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    print(f"训练 {args.task}，device={cfg['device']}，仅使用公共训练标签", flush=True)
    {"detector": train_detector, "geometry": train_geometry, "wheel": train_wheel}[
        args.task
    ](cfg, args)


if __name__ == "__main__":
    main()
