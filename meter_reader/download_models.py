"""按需下载官方初始化权重；不会覆盖本地微调权重。"""

import urllib.request
from meter_reader.config import load_config


def main():
    cfg = load_config()
    urls = {
        "detector_pretrained": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "wheel_pretrained": "https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth",
    }
    for name, url in urls.items():
        path = cfg["paths"][name]
        if path.is_file():
            print("已有:", path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        print("下载:", url, flush=True)
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
        print("已保存:", path)


if __name__ == "__main__":
    main()
