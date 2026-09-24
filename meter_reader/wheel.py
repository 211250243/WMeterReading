"""SVTRv2 整行字轮识别。保留原始字符，不从文件名或整表标签补字。"""

import copy
import sys
import cv2
import numpy as np
import torch
import yaml


def wheel_characters(cfg):
    return (
        ["blank"]
        + (cfg["paths"]["openocr_source"] / "tools/utils/ppocr_keys_v1.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        + [" "]
    )


def build_wheel(cfg, chars=None):
    source = cfg["paths"]["openocr_source"]
    sys.path.insert(0, str(source))
    from openrec.modeling import build_model

    recipe = yaml.safe_load(
        (source / "configs/rec/svtrv2/svtrv2_ch.yml").read_text(encoding="utf-8")
    )
    chars = chars if chars is not None else wheel_characters(cfg)
    if not chars or chars[0] != "blank" or len(set(chars)) != len(chars):
        raise ValueError("CTC字典必须以blank开始且字符唯一")
    architecture = copy.deepcopy(recipe["Architecture"])
    architecture["Decoder"]["out_channels"] = len(chars)
    return build_model(architecture), chars


def load_pretrained(model, path, source_chars=None, target_chars=None):
    state = torch.load(
        path, map_location="cpu", weights_only=False
    )  # official OpenOCR checkpoint includes metadata
    state = state.get("state_dict", state.get("model", state))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # Official checkpoint retains the GTC training-only guidance head.
    state = {
        k: v
        for k, v in state.items()
        if not k.startswith(("decoder.before_gtc.", "decoder.gtc_head."))
    }
    load_wheel_state(model, state, source_chars, target_chars)


def load_wheel_state(model, state, source_chars=None, target_chars=None):
    """仅迁移CTC输出头对应字符行；其余参数仍严格加载。"""
    if (
        source_chars is not None
        and target_chars is not None
        and source_chars != target_chars
    ):
        indices = [source_chars.index(c) for c in target_chars]
        state = dict(state)
        for key in ("decoder.fc.weight", "decoder.fc.bias"):
            state[key] = state[key][indices]
    model.load_state_dict(state, strict=True)


def wheel_tensor(crop, width=320):
    # Same BGR / [-1,1] convention as OpenOCR's RecDynamicResize inference.
    h, w = crop.shape[:2]
    resized = cv2.resize(crop, (min(width, max(8, int(np.ceil(48 * w / h)))), 48))
    x = np.zeros((3, 48, width), np.float32)
    x[:, :, : resized.shape[1]] = (
        resized.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
    )
    return torch.from_numpy(x)


def decode(pred, chars):
    if isinstance(pred, dict):
        pred = pred["ctc"]
    # CTCDecoder returns probabilities in eval mode, logits in train mode.
    scores, indices = pred.max(-1)
    results = []
    for ids, probs in zip(indices.tolist(), scores.tolist()):
        text = []
        confidences = []
        last = None
        for i, p in zip(ids, probs):
            if i and i != last:
                text.append(chars[i])
                confidences.append(p)
            last = i
        results.append(
            dict(
                visible_text="".join(text),
                score=float(np.mean(confidences)) if confidences else 0,
                character_scores=confidences,
            )
        )
    return results


class WheelReader:
    def __init__(self, cfg):
        self.device = cfg["device"]
        if cfg["paths"]["wheel_weights"].is_file():
            state = torch.load(
                cfg["paths"]["wheel_weights"], map_location="cpu", weights_only=True
            )
            self.model, self.chars = build_wheel(cfg, state.get("characters"))
            self.model.load_state_dict(state["model"])
            self.training = state.get("training", {})
            self.finetuned = True
        else:
            self.model, self.chars = build_wheel(cfg)
            load_pretrained(self.model, cfg["paths"]["wheel_pretrained"])
            self.training = {"status": "通用预训练，未做字轮微调"}
            self.finetuned = False
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, crop):
        result = decode(
            self.model(wheel_tensor(crop)[None].to(self.device)), self.chars
        )[0]
        result["finetuned"] = self.finetuned
        result["transition_marked"] = "v" in result["visible_text"]
        result["candidates"] = [result["visible_text"]]
        if result["transition_marked"]:
            result["reason"] = (
                "v 保留为公共标签状态符号，语义未确认，不自动改成相邻数字"
            )
        return result
