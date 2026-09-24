"""WMeter 的上游导入只出现在本适配文件。"""

import sys
import cv2
import numpy as np
import torch


class PointerReader:
    def __init__(self, cfg):
        sys.path.insert(0, str(cfg["paths"]["wmeter_source"]))
        from local_inference import Reader, CLASSES

        self.reader = Reader(cfg["paths"]["pointer_weights"], cfg["device"])
        self.classes = CLASSES
        self.threshold = cfg["pointer"]["threshold"]

    def predict(self, crops):
        result = self.reader.predict(crops)
        result["score_note"] = "分数为模型输出，未经正确率校准"
        return result

    @torch.inference_mode()
    def individuals(self, crops):
        # Use only the CNN when order/completeness is unknown. Running the
        # carry decoder on a one-dial sequence produces misleading digits.
        results = []
        for crop in crops:
            if crop is None or crop.size == 0:
                raise ValueError("需要有效指针裁剪")
            gray = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_RGB2GRAY)
            x = (
                torch.from_numpy(np.ascontiguousarray(gray)).to(
                    device=self.reader.device, dtype=torch.float32
                )[None, None, None]
                / 127.5
                - 1
            )
            _, distribution = self.reader.encoder(x)
            scores = distribution[0, 0]
            results.append(
                dict(
                    raw_class=self.classes[int(scores.argmax())],
                    raw_score=float(scores.max()),
                )
            )
        return results
