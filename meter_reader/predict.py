import argparse
import csv
import json
from pathlib import Path
from meter_reader.config import load_config, resolve, stored_path
from meter_reader.pipeline import Pipeline

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def batch(paths, pipeline, output, base=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for i, path in enumerate(paths):
        relative = path.relative_to(base) if base else Path(path.name)
        directory = output / relative
        try:
            result = pipeline.predict(path, directory)
        except (ValueError, OSError) as e:
            directory.mkdir(parents=True, exist_ok=True)
            result = dict(
                image=stored_path(path),
                mode="automatic_image_only",
                status="error",
                reading=None,
                candidate_reading=None,
                unit="m³",
                reasons=[str(e)],
                wheel_results=[],
                pointer_results=[],
            )
        (directory / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(result)
        print(
            f'{i+1}/{len(paths)} {path.name}: {result["reading"] or result["status"]}',
            flush=True,
        )
        # Save after each image, so an interrupted batch retains already computed results.
        (output / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    save_table(results, output)
    return results


def save_table(results, output):
    output = Path(output)
    (output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "total": len(results),
        "complete_predictions": sum(r["status"] == "ok" for r in results),
        "needs_review": sum(r["status"] == "needs_review" for r in results),
        "errors": sum(r["status"] == "error" for r in results),
        "note": "仅统计模型输出状态，不代表准确率",
    }
    (output / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "reading",
                "candidate_reading",
                "unit",
                "status",
                "wheel_visible",
                "pointer_digits",
                "reasons",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    **{
                        k: r.get(k)
                        for k in [
                            "image",
                            "reading",
                            "candidate_reading",
                            "unit",
                            "status",
                        ]
                    },
                    "wheel_visible": " | ".join(
                        w.get("visible_text", "") for w in r["wheel_results"]
                    ),
                    "pointer_digits": "".join(
                        p.get("digit") or "?" for p in r["pointer_results"]
                    ),
                    "reasons": "；".join(r["reasons"]),
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="原始图片 → 自动定位/校正 → 字轮与指针 → 整表结果"
    )
    parser.add_argument("input")
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    source = resolve(args.input)
    if not source.exists():
        raise FileNotFoundError(source)
    paths = (
        sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if source.is_dir()
        else [source]
    )
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("输入目录没有图片")
    batch(
        paths,
        Pipeline(cfg),
        resolve(args.output) if args.output else cfg["paths"]["predictions"],
        source if source.is_dir() else None,
    )


if __name__ == "__main__":
    main()
