import sys
import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms


def build_detector(cfg, size, queries=100):
    source = cfg["paths"]["rtdetr_source"]
    sys.path.insert(0, str(source))
    from src.core import YAMLConfig

    recipe = YAMLConfig(
        str(source / "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"),
        num_classes=2,
        remap_mscoco_category=False,
        eval_spatial_size=[size, size],
        PResNet={"pretrained": False},
        RTDETRTransformerv2={"num_queries": queries},
        RTDETRPostProcessor={"num_top_queries": queries},
    )
    return recipe.model, recipe.criterion, recipe.postprocessor


def image_tensor(image, size):
    rgb = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255


def postprocess(prediction, postprocessor, sizes, settings):
    """训练验证和图片推理共用；sizes按原图(width, height)排列。"""
    results = []
    for result, (w, h) in zip(postprocessor(prediction, sizes), sizes):
        boxes, scores, labels = (result[k] for k in ("boxes", "scores", "labels"))
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        valid = (
            (scores >= settings["threshold"])
            & torch.isfinite(boxes).all(-1)
            & ((boxes[:, 2:] - boxes[:, :2]).min(-1).values >= 5)
        )
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        keep = batched_nms(boxes, scores, labels, settings["nms_iou"])
        results.append(
            dict(boxes=boxes[keep], scores=scores[keep], labels=labels[keep])
        )
    return results


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]
        path = cfg["paths"]["detector_weights"]
        if not path.is_file():
            raise FileNotFoundError(f"尚无水表检测权重，请先训练 detector: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.size = state["size"]
        self.model, _, self.post = build_detector(
            cfg, self.size, state.get("queries", 100)
        )
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.post.to(self.device).eval()
        self.training = state.get("training", {})

    @torch.inference_mode()
    def predict(self, image):
        h, w = image.shape[:2]
        pred = self.model(image_tensor(image, self.size)[None].to(self.device))
        result = postprocess(
            pred,
            self.post,
            torch.tensor([[w, h]], device=self.device),
            self.cfg["detector"],
        )[0]
        scores, boxes, labels = result["scores"], result["boxes"], result["labels"]
        regions = []
        for i in range(len(boxes)):
            b = boxes[i].cpu().tolist()
            if min(b[2] - b[0], b[3] - b[1]) >= 5:
                regions.append(
                    dict(
                        kind=("pointer", "wheel")[int(labels[i])],
                        box=b,
                        score=float(scores[i]),
                    )
                )
        return regions
