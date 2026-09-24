from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    with open(
        resolve(path) if path else ROOT / "meter_reader/config.yaml", encoding="utf-8"
    ) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: resolve(v) for k, v in cfg["paths"].items()}
    return cfg


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def stored_path(path):
    """项目内产物使用根目录相对路径，复制整个项目后仍能打开。"""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
