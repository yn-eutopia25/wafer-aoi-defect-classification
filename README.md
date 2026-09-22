# AOI Defect Classification and Localization

面向半导体封装 AOI 局部图像的小样本缺陷分类、实例定位与结果分析项目。项目覆盖数据清单、Web 标注、标注校验、训练前审计、目标感知 Patch 分类、YOLO 目标检测、错误归因和小目标切片推理。

真实 AOI 图像、人工标注、模型权重和实验输出包含非公开数据，不写入 Git 历史。经授权的项目成员可从私有 Release `private-data-v1.0.0` 下载去标识化数据与实验归档；普通代码使用者仍需准备自己的数据。

## 主要能力

- 扫描原始图像并生成带哈希的 manifest。
- 使用 FastAPI + Canvas 标注矩形或多边形缺陷实例。
- 导出实例表和图片级多标签汇总，并执行几何、越界和一致性校验。
- 按 `image_id` 进行多标签分层划分，避免同图 Patch 泄漏。
- 构建 context-aware、focused 和 A 类 clean-tile Patch。
- 比较手工纹理特征、传统分类器和冻结的 ResNet18 特征。
- 导出 YOLO 检测数据集，训练并评估轻量检测器。
- 分析小目标、嵌套框、定位偏差、类别混淆和计数误差。
- 使用重叠切片改善颗粒类小目标的召回率。

## 标签体系

| 代码 | 类别 | 含义 |
|---|---|---|
| A | `residue_cleaning` | 胶残留、清洗不净、大片发黑 |
| B | `edge_glue` | 边缘挂胶、胶丝、溢胶 |
| C | `particle` | 颗粒、异物、小目标 |
| D | `pad_abnormal` | Pad 或植球开口异常 |
| E | `surface_damage` | 刻蚀或表面损伤 |
| F | `pi_broken` | PI 胶破损、露金属 |
| G | `uncertain_other` | 不确定、其他或疑似误报 |
| N1–N3 | `pseudo_normal_*` | 局部正常表面、Pad 和边缘 |

完整定义见 [`configs/label_schema.yaml`](configs/label_schema.yaml)。N1–N3 只是 NG 图像中的局部正常区域，不等同于完整 GOOD 样本。

## 仓库结构

```text
configs/                 标签定义
data/README.md           私有数据目录约定
docs/REPRODUCIBILITY.md  数据划分、评估与复现口径
docs/PRIVATE_RELEASE.md  私有 Release 内容、下载和完整性校验
models/README.md         本地模型目录约定
reports/README.md        本地报告目录约定
scripts/                 编号式数据和训练流水线
src/aoi_defect/          标注、特征、模型和指标模块
web/                     Web 标注工具前端
requirements.txt         Python 依赖
```

## 环境

推荐 Python 3.10–3.12。GPU 训练需要与本机 CUDA 驱动兼容的 PyTorch；仅使用标注工具和传统 Patch 分类时可以在 CPU 上运行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 数据准备

按照 [`data/README.md`](data/README.md) 创建本地目录，将原始图像放入 `data/raw_original/`。该目录应保持只读。

```powershell
python scripts/01_build_manifest.py
python scripts/02_rename_images.py
python scripts/03_make_contact_sheets.py
```

## Web 标注与校验

```powershell
python scripts/05_run_annotation_server.py --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。人工标注写入本地 `data/annotations/annotations.json`，该文件是唯一人工标注源，不应提交到 Git。

```powershell
python scripts/06_export_annotations.py
python scripts/07_validate_annotations.py
python scripts/08_make_annotation_review_sheets.py --only_done
python scripts/09_audit_training_data.py
python scripts/10_make_image_splits.py
```

数据划分以图片为单位。首次生成后应保留本地 `test_split_lock.json`，模型选择和阈值调整只使用训练集与验证集。

## Patch 分类

```powershell
python scripts/11_build_target_aware_patches.py --run_full
python scripts/12_train_patch_baselines.py --run-full
python scripts/13_evaluate_patch_classifier.py --split val
python scripts/18_optimize_patch_classifier.py --run-full
```

目标感知 Patch 会根据类别保留不同范围的上下文，并对大片残胶采用 clean tile 与 focused bbox 两种策略。所有派生 Patch 继承源图片的数据划分。

## 全图目标检测

```powershell
python scripts/14_export_detection_dataset.py --link-mode hardlink
python scripts/15_train_detector.py --smoke-test --model yolo11n.pt
python scripts/15_train_detector.py --run-full --model yolo11n.pt --device auto
python scripts/16_evaluate_detector.py --split val
python scripts/17_analyze_detection_errors.py --split val
```

正式测试只应在模型和工作点冻结后执行一次：

```powershell
python scripts/16_evaluate_detector.py --split test --confirm-test-evaluation
python scripts/17_analyze_detection_errors.py --split test --confirm-test-evaluation
```

小目标切片与最终测试脚本：

```powershell
python scripts/23_run_tiled_inference_ablation.py
python scripts/24_validate_tiled_inference_robustness.py
python scripts/26_evaluate_final_tiled_test.py
python scripts/27_generate_final_optimization_report.py
```

## 本地实验结果摘要

以下结果来自单批次、同产品的内部数据，仅用于验证技术路线，不代表跨晶圆或量产泛化。

| 模块 | 指标 | 结果 |
|---|---|---:|
| 数据集 | 图像 / 标注实例 | 200 / 4005 |
| Patch 分类二期 | Accuracy | 0.9374 |
| Patch 分类二期 | Balanced Accuracy | 0.8509 |
| Patch 分类二期 | Macro-F1 | 0.8547 |
| YOLO 冻结测试 | mAP50 | 0.3879 |
| YOLO 冻结测试 | mAP50-95 | 0.1964 |
| C 类切片测试 | C Recall | 0.4572 → 0.5613 |
| C 类切片测试 | C F1 | 0.4134 → 0.4719 |

结果表明：给定正确 ROI 后，多数缺陷类别具有较好的可分性；全图检测的主要瓶颈是极小目标、模糊边界和大框内部的嵌套缺陷。简单延长训练或放大全图输入没有稳定收益，局部切片更适合改善颗粒类召回。

## 数据与隐私

Git 历史不包含以下内容：

- 原始或重命名后的 AOI 图像；
- 人工标注 JSON、CSV、图像清单和数据哈希；
- 训练权重、推理缓存和实验运行目录；
- 标注复核图、预测报告、PPT、Word 报告和历史备份；
- 用户姓名、学号、联系方式、绝对路径或访问密钥。

私有 Release `private-data-v1.0.0` 将已获授权的材料分为图像、标注、模型和实验输出四个附件。发布副本采用统一的去标识化图像名，排除原始文件名、全部历史备份、个人答辩材料和访问凭据，并附 `SHA256SUMS.txt`。具体范围见 [`docs/PRIVATE_RELEASE.md`](docs/PRIVATE_RELEASE.md)。

未获得该私有仓库访问权限时，请使用自有或获授权的数据，并在本地生成派生文件。

## 项目边界

现有数据只包含 AOI 筛出的 NG 局部图，缺少完整 GOOD 图、总 die 数、wafer 坐标和工艺参数，因此不能据此计算真实良率。真实良率建模还需要完整的 GOOD/NG 分母、wafer map、批次、recipe、光照和时间信息。
