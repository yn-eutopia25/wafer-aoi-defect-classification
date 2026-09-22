#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_optimize_detection_postprocess.py — 检测后处理离线优化 (不重训模型)

基于 16 号脚本导出的 conf=0.001 全量预测 (prediction_instances.csv)，
在 val 上离线搜索部署侧后处理策略：

  P1  每类置信度阈值优化 (目标1: F1@IoU0.5 最大; 目标2: 计数 MAE 最小)
      —— 同时报告 val-oracle 与 留一图交叉验证(LOO) 两种口径, 防止在 30 张图上过拟合
  P2  嵌套框抑制规则: 非 A 预测框被高置信 A 框包含 ≥70% 时抑制
  P3  与 baseline (全局 conf=0.25) 的计数 MAE / F1 对比

匹配协议与 16 号脚本一致的简化版: 同图同类, 按 conf 降序贪心匹配最高 IoU 的未匹配 GT,
IoU≥0.5 记 TP。仅使用 val; test 不参与。

用法:
    python scripts/19_optimize_detection_postprocess.py
"""
from __future__ import annotations

import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED_CSV = ROOT / "reports/detection_evaluation/epoch150_20260720/prediction_instances.csv"
INST_CSV = ROOT / "data/annotations/instances.csv"
SPLIT_CSV = ROOT / "data/derived/splits.csv"
OUT = ROOT / "reports/optimization_phase2/detection_postprocess"

CLASSES = ["A", "B", "C", "D", "E", "F"]
THR_GRID = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def containment(inner, outer):
    """inner 被 outer 包含的比例 (交集/inner面积)."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def load_data():
    with open(SPLIT_CSV, encoding="utf-8-sig", newline="") as f:
        val_imgs = [r["image_id"] for r in csv.DictReader(f) if r["split"] == "val"]
    val_set = set(val_imgs)
    gt = defaultdict(list)          # (img, cls) -> [box]
    with open(INST_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["image_id"] in val_set and r["label_code"] in CLASSES and r["is_pseudo_normal"] == "False":
                gt[(r["image_id"], r["label_code"])].append(
                    (float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])))
    preds = defaultdict(list)       # img -> [ (cls, conf, box) ]
    with open(PRED_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["image_id"] in val_set:
                preds[r["image_id"]].append((r["predicted_label"], float(r["confidence"]),
                    (float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"]))))
    n_gt = sum(len(v) for v in gt.values())
    print(f"val images={len(val_imgs)}  GT={n_gt}  preds(conf>=0.001)={sum(len(v) for v in preds.values())}")
    return val_imgs, gt, preds


def evaluate(val_imgs, gt, preds, thr: dict, nested_rule=False, nested_tau=0.5, nested_cont=0.7):
    """返回每类 tp/fp/fn 与每图每类计数."""
    stats = {c: [0, 0, 0] for c in CLASSES}          # tp, fp, fn
    counts = {c: {} for c in CLASSES}                # img -> pred count
    for img in val_imgs:
        plist = [p for p in preds.get(img, []) if p[1] >= thr[p[0]]]
        if nested_rule:
            a_boxes = [p for p in plist if p[0] == "A" and p[1] >= nested_tau]
            keep = []
            for p in plist:
                if p[0] != "A" and any(containment(p[2], ab[2]) >= nested_cont for ab in a_boxes):
                    continue
                keep.append(p)
            plist = keep
        for c in CLASSES:
            cp = sorted([p for p in plist if p[0] == c], key=lambda t: -t[1])
            counts[c][img] = len(cp)
            gts = list(gt.get((img, c), []))
            used = [False] * len(gts)
            tp = 0
            for _, _, box in cp:
                best, bi = 0.0, -1
                for gi, gbox in enumerate(gts):
                    if used[gi]:
                        continue
                    v = iou(box, gbox)
                    if v > best:
                        best, bi = v, gi
                if best >= 0.5 and bi >= 0:
                    used[bi] = True
                    tp += 1
            stats[c][0] += tp
            stats[c][1] += len(cp) - tp
            stats[c][2] += len(gts) - tp
    return stats, counts


def prf(st):
    tp, fp, fn = st
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def count_mae(val_imgs, gt, counts):
    """返回 (总数口径MAE=|Σpred-Σgt| 与 16 号脚本一致, 逐类绝对误差和口径, 每类MAE)."""
    per_cls, l1_err, tot_err = {}, [], []
    for img in val_imgs:
        e1 = 0; pred_tot = 0; gt_tot = 0
        for c in CLASSES:
            pc = counts[c].get(img, 0); gc = len(gt.get((img, c), []))
            e1 += abs(pc - gc); pred_tot += pc; gt_tot += gc
        l1_err.append(e1); tot_err.append(abs(pred_tot - gt_tot))
    for c in CLASSES:
        errs = [abs(counts[c].get(img, 0) - len(gt.get((img, c), []))) for img in val_imgs]
        per_cls[c] = float(np.mean(errs))
    return float(np.mean(tot_err)), float(np.mean(l1_err)), per_cls


def sweep_class(val_imgs, gt, preds, c, imgs_subset=None):
    """对单类扫阈值, 返回 {thr: (tp,fp,fn, mae)} (在 imgs_subset 上)."""
    imgs = imgs_subset if imgs_subset is not None else val_imgs
    out = {}
    for t in THR_GRID:
        tp = fp = fn = 0
        errs = []
        for img in imgs:
            cp = sorted([p for p in preds.get(img, []) if p[0] == c and p[1] >= t], key=lambda x: -x[1])
            gts = list(gt.get((img, c), []))
            used = [False] * len(gts)
            tpi = 0
            for _, _, box in cp:
                best, bi = 0.0, -1
                for gi, gbox in enumerate(gts):
                    if used[gi]:
                        continue
                    v = iou(box, gbox)
                    if v > best:
                        best, bi = v, gi
                if best >= 0.5 and bi >= 0:
                    used[bi] = True; tpi += 1
            tp += tpi; fp += len(cp) - tpi; fn += len(gts) - tpi
            errs.append(abs(len(cp) - len(gts)))
        out[t] = (tp, fp, fn, float(np.mean(errs)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val_imgs, gt, preds = load_data()

    # ---------- baseline: 全局 0.25 ----------
    thr_base = {c: 0.25 for c in CLASSES}
    st, cnt = evaluate(val_imgs, gt, preds, thr_base)
    mae_base, l1_base, mae_cls_base = count_mae(val_imgs, gt, cnt)
    macro_f1_base = float(np.mean([prf(st[c])[2] for c in CLASSES]))
    print(f"\n[baseline conf=0.25] macroF1={macro_f1_base:.4f} totalMAE={mae_base:.3f} L1MAE={l1_base:.3f}")

    # ---------- P1a: val-oracle 每类最优阈值 ----------
    sweeps = {c: sweep_class(val_imgs, gt, preds, c) for c in CLASSES}
    thr_f1, thr_cnt = {}, {}
    rows_thr = []
    for c in CLASSES:
        best_f, tf = -1, 0.25
        best_m, tm = 1e9, 0.25
        for t, (tp, fp, fn, mae) in sweeps[c].items():
            p = tp / (tp + fp) if tp + fp else 0
            r = tp / (tp + fn) if tp + fn else 0
            f = 2 * p * r / (p + r) if p + r else 0
            if f > best_f:
                best_f, tf = f, t
            if mae < best_m:
                best_m, tm = mae, t
        thr_f1[c], thr_cnt[c] = tf, tm
        rows_thr.append({"class": c, "thr_bestF1": tf, "bestF1": round(best_f, 4),
                         "thr_bestMAE": tm, "bestMAE": round(best_m, 4)})

    st_f1, cnt_f1 = evaluate(val_imgs, gt, preds, thr_f1)
    macro_f1_opt = float(np.mean([prf(st_f1[c])[2] for c in CLASSES]))
    st_c, cnt_c = evaluate(val_imgs, gt, preds, thr_cnt)
    mae_opt, l1_opt, mae_cls_opt = count_mae(val_imgs, gt, cnt_c)
    print(f"[P1a oracle] per-class thr(F1): macroF1={macro_f1_opt:.4f}")
    print(f"[P1a oracle] per-class thr(MAE): totalMAE={mae_opt:.3f}  (baseline {mae_base:.3f})")

    # ---------- P1b: LOO 交叉验证口径 ----------
    loo_f1_stats = {c: [0, 0, 0] for c in CLASSES}
    loo_counts = {c: {} for c in CLASSES}
    for hold in val_imgs:
        others = [i for i in val_imgs if i != hold]
        thr_hm, thr_hf = {}, {}
        for c in CLASSES:
            sw = sweep_class(val_imgs, gt, preds, c, imgs_subset=others)
            best_m, tm = 1e9, 0.25
            best_f, tf = -1, 0.25
            for t, (tp_, fp_, fn_, mae) in sw.items():
                if mae < best_m:
                    best_m, tm = mae, t
                pp = tp_ / (tp_ + fp_) if tp_ + fp_ else 0
                rr = tp_ / (tp_ + fn_) if tp_ + fn_ else 0
                ff = 2 * pp * rr / (pp + rr) if pp + rr else 0
                if ff > best_f:
                    best_f, tf = ff, t
            thr_hm[c] = tm; thr_hf[c] = tf
        _, cnt_h = evaluate([hold], gt, preds, thr_hm)
        st_hf, _ = evaluate([hold], gt, preds, thr_hf)
        for c in CLASSES:
            for k in range(3):
                loo_f1_stats[c][k] += st_hf[c][k]
            loo_counts[c][hold] = cnt_h[c][hold]
    mae_loo, l1_loo, mae_cls_loo = count_mae(val_imgs, gt, loo_counts)
    macro_f1_loo = float(np.mean([prf(loo_f1_stats[c])[2] for c in CLASSES]))
    print(f"[P1b LOO ] per-class thr(MAE): totalMAE={mae_loo:.3f}  thr(F1): macroF1={macro_f1_loo:.4f}")

    # ---------- P2: 嵌套抑制 (在 F1 最优阈值之上) ----------
    p2_rows = []
    for tau in (0.3, 0.5, 0.7):
        for cont in (0.7, 0.9):
            st2, _ = evaluate(val_imgs, gt, preds, thr_f1, nested_rule=True,
                              nested_tau=tau, nested_cont=cont)
            mf = float(np.mean([prf(st2[c])[2] for c in CLASSES]))
            fp_tot = sum(st2[c][1] for c in CLASSES)
            tp_tot = sum(st2[c][0] for c in CLASSES)
            p2_rows.append({"A_conf_tau": tau, "containment": cont,
                            "macroF1": round(mf, 4), "total_TP": tp_tot, "total_FP": fp_tot,
                            "delta_macroF1_vs_P1a": round(mf - macro_f1_opt, 4)})
            print(f"[P2 nested tau={tau} cont={cont}] macroF1={mf:.4f} (Δ{mf-macro_f1_opt:+.4f}) FP={fp_tot}")

    # ---------- outputs ----------
    per_cls_rows = []
    for c in CLASSES:
        p0, r0, f0 = prf(st[c])
        p1, r1, f1_ = prf(st_f1[c])
        pl, rl, fl = prf(loo_f1_stats[c])
        per_cls_rows.append({
            "class": c,
            "baseline025_P": round(p0, 4), "baseline025_R": round(r0, 4), "baseline025_F1": round(f0, 4),
            "oracle_thr": thr_f1[c], "oracle_P": round(p1, 4), "oracle_R": round(r1, 4), "oracle_F1": round(f1_, 4),
            "MAE_base": round(mae_cls_base[c], 4), "MAE_oracle": round(mae_cls_opt[c], 4),
            "MAE_loo": round(mae_cls_loo[c], 4),
        })
    with open(OUT / "per_class_threshold_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cls_rows[0].keys())); w.writeheader(); w.writerows(per_cls_rows)
    with open(OUT / "threshold_sweep_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_thr[0].keys())); w.writeheader(); w.writerows(rows_thr)
    with open(OUT / "nested_suppression_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(p2_rows[0].keys())); w.writeheader(); w.writerows(p2_rows)

    summary = {
        "protocol": "greedy same-class match by conf desc, TP iff IoU>=0.5; val only",
        "baseline_global_conf025": {"macroF1": round(macro_f1_base, 4), "total_count_MAE": round(mae_base, 4), "perclass_L1_MAE": round(l1_base, 4),
                                    "per_class_MAE": {k: round(v, 4) for k, v in mae_cls_base.items()}},
        "per_class_thr_bestF1_oracle": {"thresholds": thr_f1, "macroF1": round(macro_f1_opt, 4)},
        "per_class_thr_bestMAE_oracle": {"thresholds": thr_cnt, "total_count_MAE": round(mae_opt, 4), "perclass_L1_MAE": round(l1_opt, 4),
                                         "per_class_MAE": {k: round(v, 4) for k, v in mae_cls_opt.items()}},
        "per_class_thr_bestF1_LOO": {"macroF1": round(macro_f1_loo, 4)},
        "per_class_thr_bestMAE_LOO": {"total_count_MAE": round(mae_loo, 4), "perclass_L1_MAE": round(l1_loo, 4),
                                      "per_class_MAE": {k: round(v, 4) for k, v in mae_cls_loo.items()}},
    }
    (OUT / "postprocess_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 检测后处理离线优化报告 (val · 150-epoch 基线)", "",
             "不重训模型, 仅在 conf=0.001 全量预测上离线搜索部署侧后处理。",
             "匹配协议: 同图同类按 conf 降序贪心匹配, IoU≥0.5 计 TP; test 不参与。", "",
             "## 总览", "",
             f"- baseline (全局 conf=0.25): macro-F1 **{macro_f1_base:.4f}**, 总数口径 MAE **{mae_base:.3f}** (=16号脚本口径), 逐类误差和 **{l1_base:.3f}**",
             f"- 每类阈值 (F1 最优, val-oracle): macro-F1 **{macro_f1_opt:.4f}**; LOO 无泄漏口径 **{macro_f1_loo:.4f}**",
             f"- 每类阈值 (MAE 最优, val-oracle): 总数口径 **{mae_opt:.3f}** / 逐类误差和 **{l1_opt:.3f}**",
             f"- 每类阈值 (MAE 最优, 留一图 LOO 无泄漏口径): 总数口径 **{mae_loo:.3f}** / 逐类误差和 **{l1_loo:.3f}**", "",
             "## 每类结果", "",
             "| 类 | conf=0.25 F1 | 最优阈值 | 阈值后 F1 | MAE(0.25) | MAE(oracle) | MAE(LOO) |",
             "|---|---|---|---|---|---|---|"]
    for r_ in per_cls_rows:
        lines.append(f"| {r_['class']} | {r_['baseline025_F1']:.4f} | {r_['oracle_thr']} "
                     f"| {r_['oracle_F1']:.4f} | {r_['MAE_base']:.3f} | {r_['MAE_oracle']:.3f} | {r_['MAE_loo']:.3f} |")
    lines += ["", "## 嵌套抑制规则 (P2)", "",
              "非 A 类预测框被 conf≥τ 的 A 框包含 ≥cont 时抑制, 在每类 F1 最优阈值之上叠加:", "",
              "| τ_A | 包含率 | macro-F1 | Δ vs 仅阈值 | 总 FP |", "|---|---|---|---|---|"]
    for r_ in p2_rows:
        lines.append(f"| {r_['A_conf_tau']} | {r_['containment']} | {r_['macroF1']:.4f} "
                     f"| {r_['delta_macroF1_vs_P1a']:+.4f} | {r_['total_FP']} |")
    (OUT / "postprocess_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 {OUT}")


if __name__ == "__main__":
    main()
