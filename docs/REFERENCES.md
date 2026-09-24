# 参考项目及采用方式

2026-09-23 最新范围：多水表固定正向拍摄，批量读数；用户已要求暂不考虑强光。以下配准/模板资料仅作可选参考，不作为当前必需实现；未引入新仓库或模型。
- [AI-on-the-edge-device配准](https://jomjol.github.io/AI-on-the-edge-device-docs/Alignment/)及[ROI配置](https://jomjol.github.io/AI-on-the-edge-device-docs/ROI-Configuration/)说明参考位置对齐后复用区域的流程。上游要求人工配置ROI；本项目每图自动定位，不要求建档。
- [OpenCV ECC配准](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)及[特征匹配与RANSAC](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)：未来确有固定区域复用需求时的可选资料，当前不优先接入。
- [Ultralytics Pose](https://docs.ultralytics.com/tasks/pose/)：参考检测与关键点联合输出能力，考虑将已有有序四角作为自定义关键点；尚未接入，不能据通用基准宣称水表更准确。

此前查阅的采集侧资料仅保留链接，已移出当前工作范围：[Basler照明](https://www.baslerweb.com/en-us/learning/illumination/)、[Edmund Optics偏振](https://www.edmundoptics.com/knowledge-center/application-notes/illumination/successful-light-polarization-techniques)。本轮不接入相关模块。

## 2026-09-23：按用户要求复查四个功能块

以下仅记录直接影响本项目选择的作者资料和厂商资料。未找到这些候选在本项目混合字轮/小盘场景下的统一比较，也没有足以判断商业市场算法占比的公开证据。因而区分“有公开实现”“有通用实验结果”和“本项目已验证”，不称任何一项工业最优。实施决策见[PLAN第6节](PLAN.md#6-四个功能块的调研结论与选择2026-09-23补充)。

| 来源 | 能支持的结论 | 本项目采用边界 |
|---|---|---|
| [Anyline商业抄表产品](https://anyline.com/products/ocr-meter-reading) | 商业SDK公开支持字轮/数字/表盘读数；说明图像抄表有实际产品 | 页面未公开足以复现的内部网络，不能据此宣称工业普遍用YOLO或某OCR，也不能假定覆盖我们的混合表型 |
| [OpenVINO Industrial Meter Reader](https://docs.openvino.ai/2024/notebooks/meter-reader-with-output.html) | 工业厂商的公开示例采用检测→针/刻度分割→圆环展开→刻度位置计算 | 是指针仪表示例，不是多盘水表联合进位方案；只能证明这条技术路线存在，不能证明行业占比 |
| [ETH论文](https://arxiv.org/abs/2404.08785)及[代码](https://github.com/ethz-asl/analog_gauge_reader) | 学习型模拟仪表识别的研究实现，包含检测、分割、关键点等环节 | 压力表读数任务与多位水表不同；新增针线/刻度等监督成本高，暂不整合 |
| [AI-on-the-edge ROI配置](https://jomjol.github.io/AI-on-the-edge-device-docs/ROI-Configuration/) | 固定机位开源系统可以逐字符/逐盘设ROI并使用训练模型 | 文档明确每台设备人工设ROI；不符合本项目正常推理只输入图片的要求，不把它当即插即用替代 |
| [Ultralytics自定义关键点格式](https://docs.ultralytics.com/datasets/pose/)及[Pose任务](https://docs.ultralytics.com/tasks/pose/) | 支持自定义关键点数、检测框和关键点联合训练 | 用已有有序四角做4关键点是本项目拟议适配，并非现成水表模型；优先验证能否取代RT-DETR+RegionNet。角点标签/增强须保留读数语义方向 |
| [D-FINE作者代码](https://github.com/Peterande/D-FINE) | 提供更细的框回归方法、COCO预训练和自定义训练配置 | 保留D-FINE-M作为条件性框检测对照；通用AP不能证明水表更准，也不输出倍率或有向四角 |
| [SVTRv2作者实现](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)及[论文](https://arxiv.org/abs/2411.15858) | CTC整行识别，有公开训练配置及通用场景文字实验 | 继续用已有OpenOCR微调；其通用成绩不能套用到翻字水表，当前冻结编码器的结果不是能力上限 |
| [SCUT-WMN作者说明](https://github.com/HCIILAB/Water-Meter-Number-DataSet) | 整行字符监督可表达字轮中间状态，不必先标每位框 | 只借鉴过渡状态编码思路；不能将其标签定义等同于WMeter的v，不能照搬末位加0.5为本项目进位规则。数据声明仅限非商业研究，本次不下载或加入训练 |
| [PARSeq作者实现](https://github.com/baudm/parseq) | 通用序列识别候选，有训练与预训练模型 | 字轮专门适配仍失败时再比较。现有OpenOCR有configs/rec/parseq，优先复用，但本项目尚未适配或训练 |
| [WMeter-Reader作者实现](https://github.com/ZZZHANG-jx/WMeter-Reader)及[论文](https://ieeexplore.ieee.org/document/10909286) | 专门研究水表过渡读数，WMeter5K来自附加摄像头；发布指针权重及上下文训练代码 | 继续复用本地CPU修复版本。作者推理说明先用标注裁剪指针，因此不能将上游结果当成本项目自动定位或字轮识别已完成 |

公开方法共同支持按区域、读数类型和表盘语义分阶段处理，但**不要求固定拆成四个神经网络**。精确窗口可以用框/有序四角表达；指针像素分割是另一种任务，不能混为一谈。固定正向场景先保证裁剪准确，不必把整表大角度旋正当必经步骤；小盘零刻度和读数方向仍需处理。以上是结合来源与本项目约束作出的工程选择，不是商业市场统计。

许可简记补充：Ultralytics提供AGPL-3.0与企业许可选项（[官方许可说明](https://www.ultralytics.com/license)）；采用其代码时需按实际方式确认条款。SCUT-WMN数据限制见上表。此次没有引入这些新代码/数据，不改变已有依赖。

## 2026-09-22：最初选型记录

下文保留已接入组件和早期候选。模型效果最终要以相同图片上的整表结果比较，不能据仓库宣传直接认定谁最准。

## 第一版实际考虑引入

| 项目 | 放置位置 | 用法 |
|---|---|---|
| [WMeter-Reader](https://github.com/ZZZHANG-jx/WMeter-Reader) | `references/WMeter-Reader/` | 从旧目录复制已修复版本，复用指针模型；我们的适配逻辑写在 pointer.py |
| [RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) | `references/RT-DETR/` | 复用v2预训练和训练入口，微调区域检测；四角/方向功能需另接 |
| [OpenOCR](https://github.com/Topdu/OpenOCR) | `references/OpenOCR/` | 复用SVTRv2识别和微调入口；我们的窗口读取写在 wheel.py |

首次选型时仅WMeter指针权重已在旧环境运行；随后三项训练已在本机执行，见下文“本次实际接入”及RESULTS。替代候选仍只保留来源，不把“存在预训练链接”写成“本机已有可用水表模型”。

## 后续按问题选择，不全部下载

| 项目/方法 | 借鉴内容 | 为什么不直接整合全项目 |
|---|---|---|
| [D-FINE](https://github.com/Peterande/D-FINE)、[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | 更高精度检测候选 | 先有基线再比较；v4所查配置有DINOv3教师训练依赖 |
| [PARSeq](https://github.com/baudm/parseq)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 字轮序列/倍率文字识别对照 | 不先安装多个完整OCR运行栈 |
| [Mbarira](https://github.com/Ntarekp/Mbarira-Watermeter-model) | 自动窗口定位、旋转框、逐数字检测 | 主要字轮，缺多指针进位，透视与倒置有局限 |
| [ETH analog_gauge_reader](https://github.com/ethz-asl/analog_gauge_reader) | 刻度、针线、几何校正 | 需要额外几何监督，旧依赖较多；其仓库部分权重文件只是LFS指针 |
| [AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) | 字轮过渡、指针衔接与核对思路 | 固定机位需人工ROI；商业用途条款需另看 |
| [Automatic-Water-Meter-Reader](https://github.com/revelrush/Automatic-Water-Meter-Reader) | 窗口提取和数字组合 | 主要字轮、旧TensorFlow栈，不适合整个合并 |
| [VectorDetectionNetwork](https://github.com/DrawZeroPoint/VectorDetectionNetwork) | 指针向量方法 | 首版公共数据没有对应关键点监督 |

独立完成我们自己的pipeline、配置、数据转换、整表计算和页面；无需复用上游页面/服务。只借鉴方法时不必复制上游代码。直接复用代码或模型时保留来源与必要声明。

许可简记：RT-DETR、D-FINE、RT-DETRv4、OpenOCR、PARSeq根仓库声明Apache-2.0；ETH项目根仓库MIT。WMeter、部分小仓库的数据/权重授权尚需明确，公开可下载不等于已确认生产用途。用几行说明记录即可，不建设许可管理平台。

## 本次实际接入（2026-09-22）

- `references/WMeter-Reader/` 从旧目录复制，保留 `local_inference.py`、CPU 模态嵌入和权重加载修复。`im.pkl/tr.pkl` 已在新环境实际推理。未改写其 `+/-` 类别和上下文进位逻辑。
- `references/RT-DETR/` 与 `references/OpenOCR/` 从上表官方仓库下载，保留仓库 LICENSE。只导入模型，不搬上游页面/应用。
- 检测采用官方 [RT-DETRv2-R18 COCO 权重](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth)，在两类水表区域上微调；CPU 基线先用320分辨率和100 queries，服务器可继续640训练。
- 字轮采用官方 [OpenOCR SVTRv2 中文权重](https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth)，使用整窗可见字符微调。适配器只去掉 checkpoint 中推理不使用的 GTC 辅助头，主模型严格检查参数。
- 我们实现小型有向四角、倍率及表型网络，使用公共四角与仅训练阶段派生的位权监督。当前以该模型接通方向和倍率，不另引入倍率 OCR 或多个备选检测器。真实局限见 README 和 RESULTS。
