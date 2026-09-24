# 水表识别

真实应用场景为**多块水表，各自正上方固定正向摄像头，批量识别**。按用户最新要求，当前不考虑强光，已删除局部强光/可读性判断工作项。主流程为原图自动定位、准确裁剪、字轮/指针识别及完整计算，不以参考图或时间序列为前提。详见 [急需修复问题、模型数量与5090训练准备](docs/PLAN.md)。目前只有detector、geometry、wheel三个自研训练入口；指针复用作者两份权重，尚无适配后的训练入口。下面命令运行的是现有基线。

当前代码：原始照片 → RT-DETRv2 区域检测 → 有向四角校正 → OpenOCR SVTRv2 字轮 + WMeter 指针 → 位权/进位核对 → 批量完整及分项结果展示。

2026-09-23迁移准备已实施：统一检测后处理与验证，修正小角度/框扰动训练输入，独立保存best/last，增加数字+v字轮训练选项，导出联合四角数据，改进批量展示和相对路径。联合模型训练/推理仍未接入。详情见[PLAN第7节](docs/PLAN.md)，实际运行见[RESULTS](docs/RESULTS.md)。这些是代码和数据准备成果，不代表精度问题已解决。

已经接通真实模型与训练，当前是**需继续提高准确率的 CPU 基线**，不是高准确率生产模型。最终抽取96张公共原图，完整正确9张、87张待核对；53张业务图均待核对。完整正确结果目前都是全指针表，混合字轮表尚未可靠完成。正常推理只读图片，不使用人工 ROI、文件名数字、公共标签或人工核对答案。实际运行结果见 [RESULTS](docs/RESULTS.md)。

最新修正：已禁止缺盘、错位或倍率不可信的序列进入指针进位解码；用4张真实图片核对，结果在 `outputs/correction_sequence/`。上述96/53张统计是修改前的历史基线，尚未重跑整批。**裁剪与识别精度尚未修好**；640全参数训练本机估算40轮约37小时，仅包含检测器训练。按用户要求已在训练资源瓶颈处暂停，等待GPU环境，详见 [修正与阻塞说明](docs/DIAGNOSIS.md)。

## 安装

当前已新建 `.venv`，未复制旧环境。已验证 Python 3.11、torch 2.6.0+cpu、torchvision 0.21，无可用 CUDA，约15 GB内存。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r meter_reader/requirements.txt
.venv/Scripts/python -m pip install -e meter_reader
```

迁移至Windows RTX 5090时重新建环境，勿复制当前CPU版 `.venv`。官方已有支持Windows Blackwell的CUDA 12.8构建；下面是一组待在目标机验证的兼容起点，不代表本项目已完成GPU验证，也不代表最新版本（[官方安装组合](https://pytorch.org/get-started/previous-versions/)、[Blackwell支持说明](https://discuss.pytorch.org/t/cuda-support-for-rtx-50x-on-windows/221309/2)）：

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r meter_reader/requirements.txt
.venv/Scripts/python -m pip install -e meter_reader
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

先修复PLAN中的训练输入、验证和算法问题，再启动长时间训练。不要只在5090执行下方旧基线命令就认为修正完成。仅CUDA可见也不等于项目所有算子已验证，仍需实际前后向与真实图片推理。

Ubuntu 重新建环境，勿迁移 Windows `.venv`。下例仅适用于支持CUDA 12.4构建的旧GPU环境，**不作为RTX 5090安装方案**：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r meter_reader/requirements.txt
pip install -e meter_reader
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

复制代码、`references/`、`data/`、`weights/` 并保留目录关系。路径和阈值集中在 `meter_reader/config.yaml`。可编辑安装后模块入口可从其它目录启动，相对输入路径仍以项目根目录为基准。

源码和权重当前已就位。新部署需要重新下载时：

```bash
git clone --depth 1 https://github.com/lyuwenyu/RT-DETR.git references/RT-DETR
git clone --depth 1 https://github.com/Topdu/OpenOCR.git references/OpenOCR
python -m meter_reader.download_models
```

WMeter 应复制本项目 `references/WMeter-Reader` 的 CPU 修复版及 `weights/wmeter/{im,tr}.pkl`。下载命令仅获取 COCO、通用 OCR 初始化，不能代替水表微调权重。来源和许可见 [REFERENCES](docs/REFERENCES.md)。

## 数据与训练

```powershell
.venv/Scripts/python -m meter_reader.prepare_data
```

读取 `data/WMeter5K/` 的作者划分并跳过 `docs/skip_images.txt`，生成 `data/prepared/` 下 COCO 框、有向四角/位权监督、整行字轮标签、裁剪和预览。转换后3975/991张，字轮920/235个。保留前导零、`v`、`+/-`；不使用 `outputs/previous_review/` 人工记录训练。

当前训练入口默认全参数训练，输出到独立实验目录，不改当前推理权重。以下小批次命令已在CPU实际运行，用于检查训练链路，不是有效精度训练；输出目录已有结果时需换目录或恢复训练：

```powershell
.venv/Scripts/python -m meter_reader.train detector --init weights/detector/meter.pt --output outputs/training/detector_trial --size 640 --batch 2 --epochs 1 --limit 4 --val-limit 4
.venv/Scripts/python -m meter_reader.train geometry --init weights/geometry.pt --output outputs/training/geometry_trial --batch 8 --epochs 1 --limit 4 --val-limit 4
.venv/Scripts/python -m meter_reader.train wheel --init weights/wheel/meter.pt --wheel-digits --output outputs/training/wheel_trial --batch 2 --epochs 1 --limit 16 --val-limit 16
```

每个训练目录保存`last.pt`、`best.pt`及简短`metrics.jsonl`。检测best按IoU 0.8匹配F1选择（不是mAP）；几何按固定扰动框上的四角误差；字轮按整行可见标签完全匹配。几何分数仍是诊断分数，选定权重后还需跑自动原图验证。

`--init 路径`只加载权重开启新实验；`--resume`恢复同一输出目录的last，或`--resume 路径`指定完整checkpoint。`--epochs`为追加轮数。更改字典用init而非resume；改变验证范围会重置best比较。未指定output时使用`outputs/training/<task>`；已有实验不会被新训练静默覆盖。

5090上先完成环境与小批次验证，再用下面命令启动现有基线的全量适配。轮数仅为命令示例，batch按实测调整；是否长期训练独立geometry要先与联合四角方案比较。GPU及AMP尚未实测，当前训练使用FP32：

```bash
python -m meter_reader.train detector --device cuda --init weights/detector/meter.pt --output outputs/training/detector_640 --size 640 --batch 4 --epochs 20 --workers 0
python -m meter_reader.train geometry --device cuda --init weights/geometry.pt --output outputs/training/geometry_jitter --batch 32 --epochs 20 --workers 0 --lr 0.0003
python -m meter_reader.train wheel --device cuda --init weights/wheel/meter.pt --wheel-digits --output outputs/training/wheel_digits --batch 4 --epochs 20 --workers 0 --lr 0.00001
# 后续接着训练同一实验
python -m meter_reader.train wheel --device cuda --output outputs/training/wheel_digits --resume --batch 4 --epochs 5 --workers 0
```

`--limit 0 --val-limit 0`是默认完整划分。冻结主干切换到全参数训练时重建优化器；检测分辨率可调整。训练与推理均不静默下载模型。确认新权重效果后，在`meter_reader/config.yaml`中把相应`*_weights`指向实验的best.pt，再运行识别；训练不自动替换线上/当前模型。

检测/几何增强默认`--rotation 10`，保留原图基础方向，禁止镜像。几何从原图生成`--box-jitter 0.08`扰动框，训练与验证都保留角点语义顺序；验证扰动固定。字轮区域采样权重默认6倍，可用`--wheel-sampling`调整；验证不重复采样。`--wheel-digits`使用blank、0–9和v共12类，按字符迁移CTC头；checkpoint自带字典，旧通用字典权重也可加载。

准备联合检测四角数据（无需安装新模型）：

```powershell
.venv/Scripts/python -m meter_reader.prepare_data --pose-only
```

生成`data/prepared/pose/`的images、labels、划分列表和dataset.yaml，保持3975/991划分。图外角点用visibility=0屏蔽，不伪造坐标或针尖标签。迁移后重跑此命令更新生成YAML里的绝对根路径。这里只完成数据转换；联合模型训练适配、关闭镜像等配置及GPU实际运行仍待完成。

## 识别与查看

```powershell
# 目录（递归）或单图共用入口
.venv/Scripts/python -m meter_reader.predict data/field_images --output outputs/predictions
.venv/Scripts/python -m meter_reader.predict path/to/photo.jpg --output outputs/single

# 本地批量结果页面
.venv/Scripts/python -m streamlit run meter_reader/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

打开 [本地页面](http://127.0.0.1:8501)。默认展示批量完整读数表、字轮可见字符和逐盘结果，人工记录放在可选折叠区。可把结果目录设为`outputs/pre5090/field_predictions`查看本轮4张业务原图运行结果。

每个输出目录有 `results.json` 和 `results.csv`；逐图子目录有 `result.json`、标框图、检测裁剪和成功校正的裁剪。JSON 数字保持字符串；Excel 导入 CSV 时请把读数列设成文本。人工值单独存为 `manual_reviews.json`，重识别不覆盖、训练不读取。新批次更新结果列表，已有逐图产物与人工记录保留。

新结果的项目内图片路径相对于项目根目录，迁移整个项目后仍可查看；项目外图片保持绝对路径。历史JSON仍保留原样，含旧绝对路径的结果迁移后需重新生成或单独迁移路径。

`reading` 是满足当前一致性检查的模型完整值；`candidate_reading` 仍可能有疑问。`wheel_results` 是可见字轮，`pointer_results` 包含单盘类别与上下文数字。模型分数未经正确率校准。

## 真实验证

```powershell
# 原始图片自动整表评估；limit 0 运行完整991张验证集
.venv/Scripts/python -m meter_reader.evaluate --mode automatic --limit 32
# 使用真值裁剪和顺序，仅诊断识别模块
.venv/Scripts/python -m meter_reader.evaluate --mode crops --limit 32
# 同一自动检测框：直接裁剪/四角校正/真值裁剪可视化对照
.venv/Scripts/python -m meter_reader.evaluate --mode regions --wheels-only --limit 12 --output outputs/crop_comparison
# 用几个直观例子核对位权、重复位、缺位和临界进位
.venv/Scripts/python -m meter_reader.reading
```

结果保存到 `outputs/evaluation_automatic/` 或 `outputs/evaluation_crops/`，含 `summary.json` 和逐图对照。自动整表准确率的分母包含所有待核对/失败图。公共集有相似固定机位，只作基线，不是独立业务成绩；53张业务图无正式真值，不编造准确率。

## 当前边界

- 位权/表型由小型图像网络预测。监督仅在训练数据准备时由作者区域顺序与整表小数位生成；推理不读标签，不按左右位置猜倍率。目前没有独立倍率文字 OCR，域外图像需要核对。
- 支持的表型是字轮加3/4个小数盘，或8盘纯指针、2/3/4位小数。要求盘数、表型、位权一致；其它情况保留部分结果。
- 有向四角保留原方向、做完整透视变换；角点退化或收缩会标出。该网络对业务图、模糊和大透视仍会失败。
- 作者指针上下文模型处理各盘过渡进位；展示 `4-` 与上下文 `3` 等差异，不把 `+/-` 当简单加减一。Decimal 组合时重复位仅核对，冲突/缺位不补数字。
- `v` 保留为公共字轮状态符号并参与训练，其精确定义尚未可靠确认。无 `v` 的字轮也可能提前翻位：低位≥0.5或接近0时保留相邻整数候选和疑问，不强行加减一；遇到 `v` 保留未解析状态。尚未实现可靠的字轮翻字消歧。
- 下一步重点是方向/位权和字轮微调，再按真实失败补业务标注；不把模型输出或人工核对记录自动当训练真值。

代码按职责平铺在 `meter_reader/`。修改一个模型只需调整相应文件，不需要复制上游页面、服务或建立插件框架。
