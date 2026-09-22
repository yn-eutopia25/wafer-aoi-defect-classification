#!/usr/bin/env python3
"""
17_analyze_detection_errors.py — 检测错误分析

消费 scripts/16_evaluate_detector.py 产生的 prediction_instances.csv 与 missed_instances.csv，
将错误细分为 9 类，生成错误图集与专项分析。

错误类型:
  1. missed_small_object       漏检小目标 (尤其 C 类)
  2. localization_error        定位偏差 (同类 IoU 在 [0.1, 0.5))
  3. wrong_class               分类错误 (异类空间匹配)
  4. duplicate_prediction      重复预测 (同 GT 被多条预测占用)
  5. background_false_positive 背景误检 (无任何 GT 空间重叠)
  6. nested_box_conflict       嵌套冲突 (异类框完全包含)
  7. low_confidence_correct_region  低置信但命中正确区域
  8. oversized_prediction      预测框过大 (area 比 > 2)
  9. undersized_prediction     预测框过小 (area 比 < 0.5)

专项分析:
  - D 被预测为 F / F 被预测为 D
  - A 被预测为 E / E 被预测为 A
  - A 大框漏检
  - C 极小目标漏检

输出 (reports/detection_errors/):
  ├── error_summary_<split>.json
  ├── error_summary_<split>.md
  ├── missed_by_class/
  ├── false_positive_by_class/
  ├── wrong_class/
  ├── nested_conflicts/
  ├── D_vs_F/
  ├── A_vs_E/
  ├── C_small_objects/
  └── worst_images/

用法:
    python scripts/17_analyze_detection_errors.py --split val
    python scripts/17_analyze_detection_errors.py --split test --confirm-test-evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("det_errors")

CLASS_CODES = ["A", "B", "C", "D", "E", "F"]
CLASS_NAMES = {
    "A": "residue_cleaning", "B": "edge_glue", "C": "particle",
    "D": "pad_abnormal", "E": "surface_damage", "F": "pi_broken",
}
CLASS_COLORS = {
    "A": "#e41a1c", "B": "#ff7f00", "C": "#377eb8", "D": "#984ea3",
    "E": "#a65628", "F": "#4daf4a",
}

# 阈值
LOOSE_IOU = 0.3
NESTED_IOU = 0.7        # 嵌套: 一框包含另一框 ≥70% 面积
LOW_CONF = 0.25         # 低置信阈值
SIZE_RATIO_HI = 2.0     # 预测面积 / GT 面积 > 2 → oversized
SIZE_RATIO_LO = 0.5     # < 0.5 → undersized
SMALL_AREA_C = 100      # C 类小目标面积阈值
LARGE_AREA_A = 5000     # A 类大框面积阈值


# =============================================================================
# 工具
# =============================================================================
def iou_box(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def containment_ratio(inner, outer) -> float:
    """inner 被 outer 包含的面积比例。"""
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_inner = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return float(inter / a_inner) if a_inner > 0 else 0.0


def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def imread_rgb(path: Path) -> Optional[np.ndarray]:
    """读取图片 (支持中文路径)。"""
    try:
        import cv2
        raw = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        import cv2
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning("读取图片失败 %s: %s", path.name, e)
        return None


def find_image_path(detection_dir: Path, split: str, image_id: str) -> Optional[Path]:
    p = detection_dir / "images" / split / f"{image_id}.jpg"
    return p if p.exists() else None


# =============================================================================
# 错误分类
# =============================================================================
def classify_errors(preds: List[Dict], gts_by_img: Dict[str, List[Dict]],
                    missed_by_img: Dict[str, List[Dict]]) -> Tuple[List[Dict], List[Dict]]:
    """
    为每条预测标注 detailed_error_type；为每条漏检 GT 标注 detailed_error_type。
    返回 (preds_with_error, missed_with_error)。
    """
    # 统计每个 GT 被多少同类预测占用 (用于 duplicate)
    gt_match_count: Dict[str, int] = defaultdict(int)
    for p in preds:
        if p["error_type"] == "correct" and p["matched_ann_id"]:
            gt_match_count[p["matched_ann_id"]] += 1

    # 标记 GT 是否在 0.5 下被匹配 (matched_ann_id 出现在 correct 预测中)
    matched_gt_ann_ids = {p["matched_ann_id"] for p in preds if p["error_type"] == "correct"}

    out_preds: List[Dict] = []
    for p in preds:
        box_p = (float(p["x_min"]), float(p["y_min"]), float(p["x_max"]), float(p["y_max"]))
        area_p = box_area(box_p)
        gts = gts_by_img.get(p["image_id"], [])
        detailed = _classify_one_pred(p, box_p, area_p, gts, gt_match_count)
        p2 = dict(p)
        p2["detailed_error_type"] = detailed
        out_preds.append(p2)

    out_missed: List[Dict] = []
    for img, mlist in missed_by_img.items():
        gts = gts_by_img.get(img, [])
        preds_img = [pp for pp in preds if pp["image_id"] == img]
        for m in mlist:
            detailed = _classify_one_missed(m, gts, preds_img, matched_gt_ann_ids)
            m2 = dict(m)
            m2["detailed_error_type"] = detailed
            out_missed.append(m2)
    return out_preds, out_missed


def _classify_one_pred(p: Dict, box_p, area_p: float,
                       gts: List[Dict], gt_match_count: Dict[str, int]) -> str:
    label = p["predicted_label"]
    err = p["error_type"]
    matched_ann = p.get("matched_ann_id", "")
    iou_val = float(p.get("iou", 0.0))

    # 1. correct → 但检查是否 duplicate (该 GT 被多条 correct 预测占用)
    if err == "correct":
        if matched_ann and gt_match_count.get(matched_ann, 0) > 1:
            return "duplicate_prediction"
        # 低置信但命中正确区域
        if float(p["confidence"]) < LOW_CONF:
            return "low_confidence_correct_region"
        return "correct"

    # 2. classification_error → wrong_class / nested_box_conflict
    if err == "classification_error":
        # 检查嵌套
        box_g = _find_gt_box(gts, matched_ann)
        if box_g is not None:
            cr1 = containment_ratio(box_p, box_g)
            cr2 = containment_ratio(box_g, box_p)
            if cr1 >= NESTED_IOU or cr2 >= NESTED_IOU:
                return "nested_box_conflict"
        return "wrong_class"

    # 3. localization_error (同类 IoU 在 [0.1, 0.5))
    if err == "localization_error":
        if iou_val < 0.1:
            # 几乎不重叠，按背景误检处理
            return "background_false_positive"
        box_g = _find_gt_box(gts, matched_ann)
        if box_g is not None:
            area_g = box_area(box_g)
            if area_g > 0:
                ratio = area_p / area_g
                if ratio > SIZE_RATIO_HI:
                    return "oversized_prediction"
                if ratio < SIZE_RATIO_LO:
                    return "undersized_prediction"
        return "localization_error"

    # 4. false_positive → background / duplicate (同 GT 被占用)
    if err == "false_positive":
        # 是否存在同类 GT 且该 GT 已被另一预测占用
        if matched_ann and gt_match_count.get(matched_ann, 0) >= 1:
            # 该预测指向一个已被占用的 GT
            return "duplicate_prediction"
        # 检查是否有异类 GT 嵌套
        for g in gts:
            box_g = g["box"] if isinstance(g["box"], tuple) else (g["x_min"], g["y_min"], g["x_max"], g["y_max"])
            cr1 = containment_ratio(box_p, box_g)
            cr2 = containment_ratio(box_g, box_p)
            if cr1 >= NESTED_IOU or cr2 >= NESTED_IOU:
                return "nested_box_conflict"
        return "background_false_positive"
    return "unknown"


def _classify_one_missed(m: Dict, gts: List[Dict], preds_img: List[Dict],
                         matched_gt_ann_ids: set) -> str:
    label = m["gt_label"]
    area = float(m.get("bbox_area", 0))
    best_iou = float(m.get("best_pred_iou", 0.0))
    # 小目标漏检
    if label == "C" and area < SMALL_AREA_C:
        return "missed_small_object"
    if area < SMALL_AREA_C:
        return "missed_small_object"
    # 有同类预测但 IoU 不足 → localization
    if best_iou >= 0.1 and best_iou < 0.5:
        return "localization_error"
    # 大框漏检 (A 类)
    if label == "A" and area >= LARGE_AREA_A:
        return "oversized_prediction_missed"
    return "missed_object"


def _find_gt_box(gts: List[Dict], ann_id: str) -> Optional[Tuple[float, float, float, float]]:
    for g in gts:
        if g.get("ann_id") == ann_id:
            if isinstance(g.get("box"), tuple):
                return g["box"]
            return (float(g["x_min"]), float(g["y_min"]), float(g["x_max"]), float(g["y_max"]))
    return None


# =============================================================================
# 绘图
# =============================================================================
def draw_gallery(image_path: Path, out_path: Path,
                 gts: List[Dict], preds: List[Dict],
                 title: str, highlight_pred_ids: Optional[set] = None,
                 highlight_ann_ids: Optional[set] = None) -> bool:
    """绘制单图 gallery: GT (绿) + 预测 (按类别色)，高亮指定框。"""
    img = imread_rgb(image_path)
    if img is None:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import io
    except Exception as e:
        logger.warning("matplotlib 不可用: %s", e)
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    # GT 框 (绿色实线)
    for g in gts:
        box = g["box"] if isinstance(g.get("box"), tuple) else (
            float(g["x_min"]), float(g["y_min"]), float(g["x_max"]), float(g["y_max"]))
        x1, y1, x2, y2 = box
        lw = 2.5 if (highlight_ann_ids and g.get("ann_id") in highlight_ann_ids) else 1.2
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               edgecolor="#2ca02c", facecolor="none", linewidth=lw, linestyle="-"))
        ax.text(x1, max(0, y1 - 3), f"GT:{g.get('label', g.get('gt_label', ''))}",
                color="#2ca02c", fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

    # 预测框 (按类别色虚线)
    for p in preds:
        box = (float(p["x_min"]), float(p["y_min"]), float(p["x_max"]), float(p["y_max"]))
        x1, y1, x2, y2 = box
        c = CLASS_COLORS.get(p.get("predicted_label", p.get("label", "")), "#ffffff")
        hl = highlight_pred_ids and p.get("prediction_id") in highlight_pred_ids
        lw = 3.0 if hl else 1.0
        ls = "-" if hl else "--"
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               edgecolor=c, facecolor="none", linewidth=lw, linestyle=ls))
        conf = float(p.get("confidence", 0))
        iou_v = p.get("iou", "")
        iou_str = f" iou={float(iou_v):.2f}" if iou_v not in ("", None) else ""
        et = p.get("detailed_error_type", p.get("error_type", ""))
        ax.text(x1, y2 + 3,
                f"P:{p.get('predicted_label','')} {conf:.2f}{iou_str} [{et}]",
                color=c, fontsize=6,
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 注意: matplotlib savefig 在 Windows 上对中文路径有 bug (路径被插入多余空格)。
    # 解决方案: 先保存到内存缓冲区，再用 Path.write_bytes 写盘。
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    out_path.write_bytes(buf.read())
    return True


# =============================================================================
# 专项分析
# =============================================================================
def analyze_confusion_pair(preds: List[Dict], from_label: str, to_label: str) -> List[Dict]:
    """from_label 被预测为 to_label 的实例。"""
    return [p for p in preds
            if p.get("matched_gt_label") == from_label and p.get("predicted_label") == to_label
            and p.get("error_type") == "classification_error"]


def build_galleries(preds: List[Dict], missed: List[Dict],
                    gts_by_img: Dict[str, List[Dict]],
                    preds_by_img: Dict[str, List[Dict]],
                    detection_dir: Path, split: str, out_root: Path) -> Dict[str, int]:
    """生成所有 gallery 子目录，返回各类计数。"""
    counts: Dict[str, int] = defaultdict(int)

    def _gallery(sub: str, items: List[Dict], kind: str, limit: int = 30) -> None:
        """kind: 'pred' / 'missed' / 'auto' (auto 按每条记录字段判断)。"""
        sub_dir = out_root / sub
        seen_img: set = set()
        n = 0
        for it in items:
            if n >= limit:
                break
            img_id = it.get("image_id", "")
            if img_id in seen_img:
                # 同图多错误也只画一次，但合并高亮
                pass
            seen_img.add(img_id)
            img_path = find_image_path(detection_dir, split, img_id)
            if img_path is None:
                continue
            gts = gts_by_img.get(img_id, [])
            ps = preds_by_img.get(img_id, [])
            # 自动判断: 有 prediction_id 视为 pred，否则 missed
            it_kind = kind if kind != "auto" else ("pred" if it.get("prediction_id") else "missed")
            if it_kind == "pred":
                hl_pred = {it["prediction_id"]}
                title = f"{img_id} | {it.get('detailed_error_type','')} | P:{it.get('predicted_label')} conf={float(it.get('confidence',0)):.2f}"
            else:
                hl_pred = set()
                title = f"{img_id} | MISSED {it.get('gt_label')} area={float(it.get('bbox_area',0)):.0f} | {it.get('detailed_error_type','')}"
            ok = draw_gallery(img_path, sub_dir / f"{img_id}.png", gts, ps, title,
                              highlight_pred_ids=hl_pred)
            if ok:
                n += 1
        counts[sub] = n

    # 1. missed_by_class (每类漏检)
    for c in CLASS_CODES:
        items = [m for m in missed if m.get("gt_label") == c]
        _gallery(f"missed_by_class/class_{c}", items, "missed")

    # 2. false_positive_by_class (每类背景误检)
    for c in CLASS_CODES:
        items = [p for p in preds if p.get("predicted_label") == c
                 and p.get("detailed_error_type") == "background_false_positive"]
        _gallery(f"false_positive_by_class/class_{c}", items, "pred")

    # 3. wrong_class
    _gallery("wrong_class",
             [p for p in preds if p.get("detailed_error_type") == "wrong_class"], "pred")

    # 4. nested_conflicts
    _gallery("nested_conflicts",
             [p for p in preds if p.get("detailed_error_type") == "nested_box_conflict"], "pred")

    # 5. D_vs_F
    df_items = analyze_confusion_pair(preds, "D", "F") + analyze_confusion_pair(preds, "F", "D")
    _gallery("D_vs_F", df_items, "pred")

    # 6. A_vs_E
    ae_items = analyze_confusion_pair(preds, "A", "E") + analyze_confusion_pair(preds, "E", "A")
    _gallery("A_vs_E", ae_items, "pred")

    # 7. C_small_objects (C 类小目标漏检 + C 误检) — 混合 missed/pred，用 auto
    c_items = [m for m in missed if m.get("gt_label") == "C"
               and float(m.get("bbox_area", 0)) < SMALL_AREA_C]
    c_items += [p for p in preds if p.get("predicted_label") == "C"
                and p.get("detailed_error_type") in ("background_false_positive", "wrong_class")]
    _gallery("C_small_objects", c_items, "auto")

    # 8. worst_images (错误最多的图)
    err_by_img: Dict[str, int] = defaultdict(int)
    for p in preds:
        if p.get("detailed_error_type") not in ("correct", "low_confidence_correct_region"):
            err_by_img[p["image_id"]] += 1
    for m in missed:
        err_by_img[m["image_id"]] += 1
    worst = sorted(err_by_img.items(), key=lambda x: -x[1])[:10]
    sub_dir = out_root / "worst_images"
    for img_id, cnt in worst:
        img_path = find_image_path(detection_dir, split, img_id)
        if img_path is None:
            continue
        gts = gts_by_img.get(img_id, [])
        ps = preds_by_img.get(img_id, [])
        title = f"WORST {img_id} | {cnt} errors"
        draw_gallery(img_path, sub_dir / f"{img_id}_{cnt}errors.png", gts, ps, title)
        counts["worst_images"] += 1

    return dict(counts)


# =============================================================================
# 主流程
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="检测错误分析")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--confirm-test-evaluation", action="store_true")
    p.add_argument("--eval-dir", type=str, default="reports/detection_evaluation")
    p.add_argument("--detection-dir", type=str, default="data/derived/detection")
    p.add_argument("--output-dir", type=str, default="reports/detection_errors")
    p.add_argument("--instances", type=str, default="data/annotations/instances.csv")
    p.add_argument("--splits", type=str, default="data/derived/splits.csv")
    p.add_argument("--gallery-limit", type=int, default=30, help="每个 gallery 子目录最多图片数")
    args = p.parse_args()

    if args.split == "test" and not args.confirm_test_evaluation:
        logger.error("test 集错误分析需要 --confirm-test-evaluation 确认。")
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    eval_dir = (root / args.eval_dir).resolve()
    detection_dir = (root / args.detection_dir).resolve()
    instances_csv = (root / args.instances).resolve()
    splits_csv = (root / args.splits).resolve()
    out_root = (root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pred_csv = eval_dir / "prediction_instances.csv"
    missed_csv = eval_dir / "missed_instances.csv"
    if not pred_csv.exists() or not missed_csv.exists():
        logger.error("找不到 %s / %s。请先运行 scripts/16_evaluate_detector.py --split %s",
                     pred_csv, missed_csv, args.split)
        sys.exit(1)

    logger.info("加载预测与漏检记录...")
    pred_rows = read_csv(pred_csv)
    missed_rows = read_csv(missed_csv)
    # 过滤到本 split (通过 splits.csv)
    split_map = {}
    with open(splits_csv, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            split_map[r["image_id"]] = r["split"]
    # prediction_instances.csv 是上次评估的 split；若与当前不一致则警告
    eval_split_images = {r["image_id"] for r in pred_rows}
    sample_img = next(iter(eval_split_images), None)
    if sample_img and split_map.get(sample_img) != args.split:
        logger.error("prediction_instances.csv 的 split 与 --split=%s 不一致。请先重跑 16 脚本。", args.split)
        sys.exit(1)
    logger.info("预测: %d 条, 漏检: %d 条 (split=%s)", len(pred_rows), len(missed_rows), args.split)

    # 加载 GT (含 box tuple) 用于嵌套/分类分析
    gts_by_img: Dict[str, List[Dict]] = defaultdict(list)
    with open(instances_csv, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("is_pseudo_normal") == "True":
                continue
            if r["label_code"] not in CLASS_CODES:
                continue
            if split_map.get(r["image_id"]) != args.split:
                continue
            gts_by_img[r["image_id"]].append({
                "ann_id": r["ann_id"],
                "label": r["label_code"],
                "box": (float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])),
                "area": float(r["bbox_area_px"]),
            })
    missed_by_img: Dict[str, List[Dict]] = defaultdict(list)
    for m in missed_rows:
        m["image_id"] = m.get("image_id", "")
        missed_by_img[m["image_id"]].append(m)

    preds_by_img: Dict[str, List[Dict]] = defaultdict(list)
    for pr in pred_rows:
        pr["image_id"] = pr.get("image_id", "")
        preds_by_img[pr["image_id"]].append(pr)

    # ---- 错误分类 ----
    logger.info("执行 9 类错误分类...")
    preds_classified, missed_classified = classify_errors(pred_rows, gts_by_img, missed_by_img)

    # 统计
    pred_err_counts: Dict[str, int] = defaultdict(int)
    for p in preds_classified:
        pred_err_counts[p["detailed_error_type"]] += 1
    missed_err_counts: Dict[str, int] = defaultdict(int)
    for m in missed_classified:
        missed_err_counts[m["detailed_error_type"]] += 1

    # 专项分析
    d_to_f = analyze_confusion_pair(preds_classified, "D", "F")
    f_to_d = analyze_confusion_pair(preds_classified, "F", "D")
    a_to_e = analyze_confusion_pair(preds_classified, "A", "E")
    e_to_a = analyze_confusion_pair(preds_classified, "E", "A")
    a_large_missed = [m for m in missed_classified if m.get("gt_label") == "A"
                      and float(m.get("bbox_area", 0)) >= LARGE_AREA_A]
    c_tiny_missed = [m for m in missed_classified if m.get("gt_label") == "C"
                     and float(m.get("bbox_area", 0)) < SMALL_AREA_C]

    # ---- 生成 gallery ----
    logger.info("生成错误图集...")
    gallery_counts = build_galleries(preds_classified, missed_classified,
                                     gts_by_img, preds_by_img,
                                     detection_dir, args.split, out_root)

    # ---- 汇总 ----
    summary = {
        "split": args.split,
        "n_predictions": len(preds_classified),
        "n_missed_gt": len(missed_classified),
        "prediction_error_counts": dict(pred_err_counts),
        "missed_error_counts": dict(missed_err_counts),
        "special_analysis": {
            "D_predicted_as_F": len(d_to_f),
            "F_predicted_as_D": len(f_to_d),
            "A_predicted_as_E": len(a_to_e),
            "E_predicted_as_A": len(e_to_a),
            "A_large_box_missed": len(a_large_missed),
            "C_tiny_object_missed": len(c_tiny_missed),
        },
        "gallery_counts": gallery_counts,
        "thresholds": {
            "LOOSE_IOU": LOOSE_IOU, "NESTED_IOU": NESTED_IOU,
            "LOW_CONF": LOW_CONF, "SIZE_RATIO_HI": SIZE_RATIO_HI,
            "SIZE_RATIO_LO": SIZE_RATIO_LO, "SMALL_AREA_C": SMALL_AREA_C,
            "LARGE_AREA_A": LARGE_AREA_A,
        },
    }
    (out_root / f"error_summary_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(out_root / f"error_summary_{args.split}.md", summary, args.split)
    logger.info("错误分析完成: %s", out_root)


def _write_md(path: Path, summary: Dict, split: str) -> None:
    pc = summary["prediction_error_counts"]
    mc = summary["missed_error_counts"]
    sa = summary["special_analysis"]
    lines = [
        f"# 检测错误分析 — {split.upper()}",
        "",
        f"- 预测总数: {summary['n_predictions']}",
        f"- 漏检 GT: {summary['n_missed_gt']}",
        "",
        "## 预测端错误分布 (9 类)",
        "| error_type | 数量 |",
        "|------------|------|",
    ]
    err_order = ["correct", "low_confidence_correct_region", "localization_error",
                 "wrong_class", "duplicate_prediction", "background_false_positive",
                 "nested_box_conflict", "oversized_prediction", "undersized_prediction"]
    for k in err_order:
        lines.append(f"| {k} | {pc.get(k, 0)} |")
    for k, v in pc.items():
        if k not in err_order:
            lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## 漏检端错误分布",
        "| error_type | 数量 |",
        "|------------|------|",
    ]
    for k in ["missed_small_object", "localization_error", "oversized_prediction_missed", "missed_object"]:
        lines.append(f"| {k} | {mc.get(k, 0)} |")
    for k, v in mc.items():
        if k not in ["missed_small_object", "localization_error", "oversized_prediction_missed", "missed_object"]:
            lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## 专项混淆分析",
        "| 项目 | 数量 |",
        "|------|------|",
        f"| D 被预测为 F | {sa['D_predicted_as_F']} |",
        f"| F 被预测为 D | {sa['F_predicted_as_D']} |",
        f"| A 被预测为 E | {sa['A_predicted_as_E']} |",
        f"| E 被预测为 A | {sa['E_predicted_as_A']} |",
        f"| A 大框漏检 (area>={LARGE_AREA_A}px²) | {sa['A_large_box_missed']} |",
        f"| C 极小目标漏检 (area<{SMALL_AREA_C}px²) | {sa['C_tiny_object_missed']} |",
        "",
        "## 错误图集",
        "| 目录 | 内容 | 图片数 |",
        "|------|------|--------|",
    ]
    gallery_desc = {
        "missed_by_class": "每类漏检 gallery",
        "false_positive_by_class": "每类背景误检 gallery",
        "wrong_class": "分类错误 gallery",
        "nested_conflicts": "嵌套冲突 gallery",
        "D_vs_F": "D↔F 混淆 gallery",
        "A_vs_E": "A↔E 混淆 gallery",
        "C_small_objects": "C 小目标 gallery",
        "worst_images": "错误最多图片 gallery",
    }
    gc = summary["gallery_counts"]
    for k, desc in gallery_desc.items():
        # 汇总子目录计数
        total = sum(v for key, v in gc.items() if key == k or key.startswith(k + "/"))
        lines.append(f"| {k}/ | {desc} | {total} |")
    lines += [
        "",
        "## 阈值参数",
        f"- LOOSE_IOU (宽松匹配): {LOOSE_IOU}",
        f"- NESTED_IOU (嵌套判定): {NESTED_IOU}",
        f"- LOW_CONF (低置信): {LOW_CONF}",
        f"- SIZE_RATIO_HI (过大): {SIZE_RATIO_HI}",
        f"- SIZE_RATIO_LO (过小): {SIZE_RATIO_LO}",
        f"- SMALL_AREA_C (C 小目标): {SMALL_AREA_C} px²",
        f"- LARGE_AREA_A (A 大框): {LARGE_AREA_A} px²",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
