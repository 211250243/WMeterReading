"""由图像学习有向四角、位权与表型；不在推理时读取标签或人工坐标。"""

import cv2
import numpy as np
import torch
from torch import nn
from meter_reader.geometry import region_crop, rectify

EXPONENTS = list(range(-4, 6))
LAYOUTS = ["wheel3", "wheel4", "pointer8d2", "pointer8d3", "pointer8d4"]


class RegionNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        previous = 3
        for channels in (24, 48, 96, 128):
            layers += [
                nn.Conv2d(previous, channels, 3, 2, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ]
            previous = channels
        self.features = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU()
        )
        self.corners = nn.Linear(256, 8)
        self.exponent = nn.Linear(256, len(EXPONENTS))
        self.layout = nn.Linear(256, len(LAYOUTS))

    def forward(self, x):
        f = self.features(x)
        return self.corners(f).reshape(-1, 4, 2), self.exponent(f), self.layout(f)


def tensor(image):
    rgb = cv2.cvtColor(cv2.resize(image, (128, 128)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 127.5 - 1


class GeometryModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]
        state = torch.load(
            cfg["paths"]["geometry_weights"], map_location="cpu", weights_only=True
        )
        self.model = RegionNet().to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.training = state.get("training", {})

    @torch.inference_mode()
    def predict(self, image, region):
        context, offset, scale = region_crop(
            image, region["box"], self.cfg["geometry"]["padding"]
        )
        q, e, l = self.model(tensor(context)[None].to(self.device))
        local = q[0].cpu().numpy()
        if local.min() < -0.15 or local.max() > 1.15:
            raise ValueError("预测角点越出区域")
        quad = local * scale + offset
        crop, matrix = rectify(image, quad)
        area = cv2.contourArea(quad.astype(np.float32))
        b = region["box"]
        if area < 0.25 * (b[2] - b[0]) * (b[3] - b[1]):
            raise ValueError("预测角点收缩，方向不可靠")
        ep, lp = e.softmax(-1)[0], l.softmax(-1)[0]
        region.update(
            quad=quad.tolist(),
            transform=matrix.tolist(),
            exponent=EXPONENTS[int(ep.argmax())],
            weight_score=float(ep.max()),
            layout=LAYOUTS[int(lp.argmax())],
            layout_score=float(lp.max()),
            weight_source="learned_public_labels",
            direction_source="ordered_corner_model",
        )
        return crop
