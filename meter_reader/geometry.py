"""有序四角的透视校正。角点顺序必须保留，不能按图像左上角重排。"""

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片无法读取: {path}")
    return image


def write_image(path, image):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"图片无法保存: {path}")
    encoded.tofile(str(path))


def rectify(image, quad):
    q = np.asarray(quad, np.float32).reshape(4, 2)
    if (
        not np.isfinite(q).all()
        or not cv2.isContourConvex(q)
        or cv2.contourArea(q) < 16
    ):
        raise ValueError("四角退化或自交")
    width = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    height = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    if min(width, height) < 4 or max(width, height) > 2 * max(image.shape[:2]):
        raise ValueError("裁剪尺寸无效")
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(q, dst)
    crop = cv2.warpPerspective(
        image, matrix, (width, height), borderValue=(255, 255, 255)
    )
    return crop, matrix


def region_crop(image, box, padding=0.35):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(w, int(np.ceil(x2 + dx))), min(h, int(np.ceil(y2 + dy)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise ValueError("区域尺寸不足")
    return image[y1:y2, x1:x2], np.float32([x1, y1]), np.float32([x2 - x1, y2 - y1])


def augment_regions(image, quads, degrees=10):
    """小角度旋转并扩展画布；有序角点随图变换，绝不重排或镜像。"""
    import random

    h, w = image.shape[:2]
    angle = random.uniform(-degrees, degrees)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(np.ceil(w * c + h * s)), int(np.ceil(h * c + w * s))
    matrix[:, 2] += [(nw - w) / 2, (nh - h) / 2]
    image = cv2.warpAffine(image, matrix, (nw, nh), borderValue=(120, 120, 120))
    points = cv2.transform(np.asarray(quads, np.float32).reshape(-1, 1, 2), matrix)
    image = np.clip(
        image.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255
    ).astype(np.uint8)
    if random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
    return image, points.reshape(np.asarray(quads).shape)


def annotate(image, regions):
    out = image.copy()
    for i, region in enumerate(regions):
        color = (30, 220, 30) if region["kind"] == "wheel" else (0, 170, 255)
        if region.get("quad") is not None:
            q = np.int32(region["quad"])
            cv2.polylines(out, [q], True, color, 2)
            cv2.circle(out, tuple(q[0]), 4, (255, 0, 255), -1)
            xy = tuple(q[0])
        else:
            x1, y1, x2, y2 = map(int, region["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            xy = (x1, y1)
        text = f"{i}:{region['kind']}"
        if region.get("exponent") is not None:
            text += f" 10^{region['exponent']}"
        cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out
