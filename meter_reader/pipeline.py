from pathlib import Path
import torch
from meter_reader.geometry import read_image, write_image, annotate, region_crop
from meter_reader.detector import Detector
from meter_reader.region_model import GeometryModel
from meter_reader.wheel import WheelReader
from meter_reader.pointer import PointerReader
from meter_reader.reading import calculate, pointer_sequence_issues
from meter_reader.config import stored_path


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.set_num_threads(cfg["threads"])
        self.detector = Detector(cfg)
        self.geometry = GeometryModel(cfg)
        self.wheel = WheelReader(cfg)
        self.pointer = PointerReader(cfg)

    def predict(self, path, output):
        """Normal inference has no label/ROI/profile input; only an image path."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        image = read_image(path)
        regions = self.detector.predict(image)
        wheels = []
        pointers = []
        crops = {}
        reasons = []
        if not regions:
            reasons.append("未检测到读数区域")
        for i, r in enumerate(regions):
            r["index"] = i
            r["detection_score"] = r["score"]
            raw_crop, _, _ = region_crop(image, r["box"], 0)
            raw_path = output / f'{i:02d}_{r["kind"]}_detected.jpg'
            write_image(raw_path, raw_crop)
            r["detected_crop"] = stored_path(raw_path)
            try:
                crop = self.geometry.predict(image, r)
                crop_path = output / f'{i:02d}_{r["kind"]}.jpg'
                write_image(crop_path, crop)
                r["crop"] = stored_path(crop_path)
                crops[i] = crop
                if r["kind"] == "wheel":
                    r.update(self.wheel.predict(crop))
                    wheels.append(r)
                    if r["score"] < self.cfg["wheel"]["threshold"] or any(
                        s < 0.5 for s in r["character_scores"]
                    ):
                        reasons.append("字轮识别置信度不足")
                else:
                    pointers.append(r)
            except ValueError as e:
                r["reason"] = str(e)
                reasons.append(f"区域 {i}: {e}")
        geometry_threshold = self.cfg["geometry"]["weight_threshold"]
        valid_regions = [r for r in regions if "exponent" in r]
        layouts = {r["layout"] for r in valid_regions}
        layout = (
            next(iter(layouts))
            if len(layouts) == 1 and len(valid_regions) == len(regions)
            else None
        )
        if len(layouts) > 1:
            reasons.append("各区域预测的表型不一致")
        if any(
            r["weight_score"] < geometry_threshold
            or r["layout_score"] < geometry_threshold
            for r in valid_regions
        ):
            reasons.append("倍率或表型图像预测置信度不足")
        sequence_issues = pointer_sequence_issues(regions, layout, geometry_threshold)
        reasons.extend(sequence_issues)
        context = None
        if pointers and not sequence_issues:
            pointers.sort(key=lambda r: r["exponent"], reverse=True)
            context = self.pointer.predict([crops[r["index"]] for r in pointers])
            for i, r in enumerate(pointers):
                r.update(
                    raw_class=context["individual_classes"][i],
                    raw_score=context["individual_scores"][i],
                    digit=context["decoder_tokens"][i] if context["valid"] else None,
                    context_score=context["token_scores"][i],
                )
            if not context["valid"]:
                reasons.append("指针上下文模型输出非数字")
            if min(context["token_scores"]) < self.cfg["pointer"]["threshold"]:
                reasons.append("指针上下文置信度不足")
        elif pointers:
            predictions = self.pointer.individuals(
                [crops[r["index"]] for r in pointers]
            )
            for r, pred in zip(pointers, predictions):
                r.update(pred)
        final = calculate(wheels, pointers, layout, reasons)
        overlay = output / "regions.jpg"
        write_image(overlay, annotate(image, regions))
        return dict(
            image=stored_path(path),
            mode="automatic_image_only",
            **final,
            layout=layout,
            wheel_results=wheels,
            pointer_results=pointers,
            pointer_context=context,
            pointer_context_status=dict(
                decoded=context is not None,
                reasons=sequence_issues,
            ),
            regions=regions,
            preview=stored_path(overlay),
            model_status=dict(
                detector=self.detector.training,
                geometry=self.geometry.training,
                wheel=self.wheel.training,
            ),
        )
