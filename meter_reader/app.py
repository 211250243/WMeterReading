"""批量读数与分项结果展示；可选人工记录独立保存。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import streamlit as st
from meter_reader.config import load_config, resolve, ROOT
from meter_reader.pipeline import Pipeline
from meter_reader.predict import batch, save_table, IMAGE_SUFFIXES

st.set_page_config(page_title="水表批量读数", layout="wide")
st.title("水表批量读数")
cfg = load_config()
folder = resolve(
    st.sidebar.text_input(
        "结果目录", str(cfg["paths"]["predictions"].relative_to(ROOT))
    )
)


@st.cache_resource
def get_pipeline():
    return Pipeline(load_config())


with st.sidebar.expander("识别新图片"):
    incoming = st.file_uploader(
        "上传原始水表照片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    source_text = st.text_input("或输入图片/文件夹路径", "data/field_images")
    if st.button("运行自动识别"):
        paths = []
        base = None
        if incoming:
            directory = ROOT / "data" / "incoming"
            directory.mkdir(parents=True, exist_ok=True)
            for upload in incoming:
                path = directory / Path(upload.name).name
                index = 1
                while path.exists():
                    path = (
                        directory
                        / f"{Path(upload.name).stem}_{index}{Path(upload.name).suffix}"
                    )
                    index += 1
                path.write_bytes(upload.getvalue())
                paths.append(path)
        else:
            source = resolve(source_text)
            if source.is_dir():
                base = source
                paths = sorted(
                    p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
                )
            elif source.is_file():
                paths = [source]
        if not paths:
            st.error("没有找到输入图片")
        else:
            with st.spinner("自动识别中，进度输出到终端"):
                get_pipeline.clear()
                batch(paths, get_pipeline(), folder, base)
            st.rerun()
result_path = folder / "results.json"
if not result_path.is_file():
    st.info("先运行批量识别命令生成 results.json，再选择结果目录。")
    st.stop()
results = json.loads(result_path.read_text(encoding="utf-8"))
stats = st.columns(3)
stats[0].metric("图片总数", len(results))
stats[1].metric("已输出完整读数", sum(r["status"] == "ok" for r in results))
stats[2].metric("未确定或处理失败", sum(r["status"] != "ok" for r in results))
st.caption("完整读数为模型通过当前一致性检查的结果；输出数量不代表准确率。")
with st.expander("批量读数表", expanded=True):
    st.dataframe(
        [
            dict(
                图片=Path(r["image"]).name,
                完整读数=r.get("reading") or "未确定",
                字轮可见字符=" | ".join(
                    w.get("visible_text", "") for w in r["wheel_results"]
                ),
                指针上下文数字="".join(
                    p.get("digit") or "?" for p in r["pointer_results"]
                ),
                原因="；".join(r["reasons"]),
            )
            for r in results
        ],
        hide_index=True,
    )
reviews_path = folder / "manual_reviews.json"
reviews = (
    json.loads(reviews_path.read_text(encoding="utf-8"))
    if reviews_path.exists()
    else {}
)
pending = st.sidebar.checkbox("只看未确定结果", value=False)
choices = [i for i, r in enumerate(results) if not pending or r["status"] != "ok"]
if not choices:
    st.info("没有符合条件的结果")
    st.stop()
index = st.sidebar.selectbox(
    "图片", choices, format_func=lambda i: f'{i+1}. {Path(results[i]["image"]).name}'
)
r = results[index]
key = r["image"]
status = {
    "ok": "模型给出完整值",
    "needs_review": "完整读数未确定",
    "error": "图片处理失败",
}.get(r["status"], r["status"])
st.write(f"状态：{status}　完整读数：**{r.get('reading') or '未确定'}** m³")
if r.get("candidate_reading") and not r.get("reading"):
    st.write(
        "未确认候选：",
        " / ".join(r.get("reading_candidates") or [r["candidate_reading"]]),
    )
if r["reasons"]:
    with st.expander(f'未确定原因（{len(r["reasons"])}项）'):
        for reason in r["reasons"]:
            st.write("• " + reason)
left, right = st.columns(2)
with left:
    preview = resolve(r.get("preview", r["image"]))
    if Path(preview).is_file():
        st.image(preview, caption="自动检测与方向四角；紫点为校正后的左上角")
    with st.expander("原图"):
        if resolve(r["image"]).is_file():
            st.image(str(resolve(r["image"])))
with right:
    st.subheader("字轮")
    if not r["wheel_results"]:
        st.write("没有可用字轮结果（全指针表可能没有字轮）")
    for w in r["wheel_results"]:
        if resolve(w["crop"]).is_file():
            st.image(str(resolve(w["crop"])))
        st.write("模型可见字符：", w.get("visible_text", ""))
        st.write("字符分数：", w.get("score"))
        st.write(w.get("reason", ""))
    st.subheader("指针")
    for p in r["pointer_results"]:
        cols = st.columns([1, 3])
        if p.get("crop") and resolve(p["crop"]).is_file():
            cols[0].image(str(resolve(p["crop"])), width=90)
        cols[1].write(
            f"倍率候选 10^{p.get('exponent')}；单盘 {p.get('raw_class','?')}；上下文 {p.get('digit') or '未确定'}"
        )
        context_score = p.get("context_score")
        cols[1].caption(
            f"倍率分数 {p.get('weight_score',0):.3f}，上下文分数 "
            + (f"{context_score:.3f}" if context_score is not None else "未解码")
        )
    failed = [
        region
        for region in r.get("regions", [])
        if region.get("reason") and not region.get("crop")
    ]
    if failed:
        st.subheader("已检测、未完成校正的区域")
        for region in failed:
            if (
                region.get("detected_crop")
                and resolve(region["detected_crop"]).is_file()
            ):
                st.image(str(resolve(region["detected_crop"])), width=160)
            st.caption(f"{region['kind']}：{region['reason']}")
with st.expander("可选：独立人工记录"), st.form(f"review_{index}"):
    manual = st.text_input(
        "人工完整读数（字符串，保留前导零）", reviews.get(key, {}).get("reading", "")
    )
    note = st.text_area("核对备注", reviews.get(key, {}).get("note", ""))
    if st.form_submit_button("保存人工核对"):
        reviews[key] = {"reading": manual, "note": note}
        temporary = reviews_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(reviews_path)
        st.success("已保存到独立人工记录；不会进入训练数据")
if st.button("用当前权重重新识别本图"):
    destination = resolve(
        r.get("preview", str(folder / "rerun" / "regions.jpg"))
    ).parent
    with st.spinner("正在识别"):
        get_pipeline.clear()
        updated = get_pipeline().predict(resolve(r["image"]), destination)
    results[index] = updated
    save_table(results, folder)
    (destination / "result.json").write_text(
        json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    st.rerun()
with st.expander("模型状态和完整结果"):
    st.json(r)
