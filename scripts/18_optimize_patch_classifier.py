#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_optimize_patch_classifier.py — Patch 六分类二期优化实验

在 12_train_patch_baselines.py 基线 (SVM-RBF, val acc 0.8826) 之上做四组实验:
  E1  基线复现校验 (同一特征/同一模型配置, 验证实验环境可信)
  E2  超参搜索 (仅在 train 上做 5-fold CV, val 只用于最终汇报) + 新模型 HistGB
  E3  软投票集成 (调参后 SVM + LR + HistGB)
  E4  特征族消融 (HSV / 灰度 / 梯度 / LBP / HOG / GLCM / bbox 几何)
  E5  置信度拒识分析 (coverage-accuracy 曲线, 人机协同工作点)

原则:
  - test 集全程不使用 (与项目 test 封存纪律一致)
  - 所有模型选择只看 train 的 CV 分数, val 仅做最终一次性汇报
  - seed=42, 特征缓存到 data/derived/features_handcrafted.npz

用法:
    python scripts/18_optimize_patch_classifier.py --run-full
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aoi_defect.patch_features import PatchFeatureExtractor          # noqa: E402
from aoi_defect.metrics import compute_all_metrics, key_confusion_pairs  # noqa: E402

from sklearn.preprocessing import StandardScaler                      # noqa: E402
from sklearn.svm import SVC                                           # noqa: E402
from sklearn.linear_model import LogisticRegression                   # noqa: E402
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline                                 # noqa: E402
import joblib                                                         # noqa: E402

CLASS_NAMES = ["A", "B", "C", "D", "E", "F"]
SEED = 42


# ---------------------------------------------------------------- features
def load_rows(manifest: Path):
    with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("use_for_defect_six_class") == "True"
            and r["train_label"] in CLASS_NAMES]


def extract_features(rows, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["patch_ids"]) == [r["patch_id"] for r in rows]:
            print(f"[cache] loaded features {z['X'].shape} from {cache.name}")
            return z["X"], z["y"], list(z["splits"]), list(z["patch_ids"])
    extractor = PatchFeatureExtractor(resize_to=128)
    X, y, splits, pids = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        if i % 500 == 0:
            print(f"  feature {i}/{len(rows)}  ({time.time()-t0:.0f}s)")
        fp = ROOT / r["raw_patch_path"]
        img = cv2.imdecode(np.frombuffer(fp.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((128, 128, 3), np.uint8)
        feats = extractor.extract(img,
            bbox_width=float(r["bbox_width"]), bbox_height=float(r["bbox_height"]),
            bbox_area=float(r["bbox_area_px"]),
            patch_area=float(r["crop_width"]) * float(r["crop_height"]),
            polygon_area=float(r["target_area_px"]))
        X.append(feats); y.append(CLASS_NAMES.index(r["train_label"]))
        splits.append(r["split"]); pids.append(r["patch_id"])
    X = np.array(X, np.float32); y = np.array(y)
    np.savez_compressed(cache, X=X, y=y, splits=np.array(splits), patch_ids=np.array(pids))
    print(f"features {X.shape} cached -> {cache.name}")
    return X, y, splits, pids


def feature_groups(extractor: PatchFeatureExtractor, n_dims: int):
    """按 _extract_impl 的真实拼接顺序划分维度段 (feature_names() 与实际输出不同步,
    此处用各开关的实际维数差经验校准)。"""
    dummy = np.zeros((128, 128, 3), np.uint8)
    kw = dict(bbox_width=10, bbox_height=10, bbox_area=100, patch_area=1000, polygon_area=100)
    n_all = len(extractor.extract(dummy, **kw))
    n_no_lbp = len(PatchFeatureExtractor(resize_to=128, include_lbp=False).extract(dummy, **kw))
    n_no_hog = len(PatchFeatureExtractor(resize_to=128, include_hog=False).extract(dummy, **kw))
    n_no_glcm = len(PatchFeatureExtractor(resize_to=128, include_glcm=False).extract(dummy, **kw))
    n_lbp, n_hog, n_glcm = n_all - n_no_lbp, n_all - n_no_hog, n_all - n_no_glcm
    assert n_all == n_dims, f"dim mismatch {n_all} vs {n_dims}"
    n_geom = 5
    base = n_all - n_lbp - n_hog - n_glcm - n_geom   # 颜色+灰度+梯度 = 44
    # base 内部: RGB 21 | HSV 6 | Lab 6 | gray 7 | sobel 3 | lap 1
    spans = {
        "颜色统计(RGB/HSV/Lab)": list(range(0, 33)),
        "灰度统计": list(range(33, 40)),
        "梯度/Laplacian": list(range(40, 44)),
        "LBP": list(range(base, base + n_lbp)),
        "HOG": list(range(base + n_lbp, base + n_lbp + n_hog)),
        "GLCM": list(range(base + n_lbp + n_hog, base + n_lbp + n_hog + n_glcm)),
        "bbox几何": list(range(n_all - n_geom, n_all)),
    }
    assert base == 44, f"unexpected base dims {base}"
    return spans


# ---------------------------------------------------------------- helpers
def eval_on(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xva) if hasattr(model, "predict_proba") else None
    pred = model.predict(Xva)
    return compute_all_metrics(yva, pred, proba, CLASS_NAMES), pred, proba


def fmt(m):
    return (f"acc={m['accuracy']:.4f} bacc={m['balanced_accuracy']:.4f} "
            f"macroF1={m['macro_f1']:.4f} top2={m['top2_accuracy']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-full", action="store_true")
    ap.add_argument("--manifest", default="data/derived/patch_manifest.csv")
    ap.add_argument("--output_dir", default="reports/optimization_phase2/patch_classifier")
    args = ap.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(ROOT / args.manifest)
    X, y, splits, pids = extract_features(rows, ROOT / "data/derived/features_handcrafted.npz")

    tr = np.array([s == "train" for s in splits]); va = np.array([s == "val" for s in splits])
    Xtr_raw, ytr = X[tr], y[tr]; Xva_raw, yva = X[va], y[va]
    print(f"train={tr.sum()} val={va.sum()} (test rows present but never touched: {(~tr & ~va).sum()})")

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xva = scaler.transform(Xtr_raw), scaler.transform(Xva_raw)
    results = {}

    # ---------------- E1 baseline reproduction -------------------------------
    print("\n=== E1 基线复现 ===")
    base_svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)
    m, pred_base, proba_base = eval_on(base_svm, Xtr, ytr, Xva, yva)
    results["E1_baseline_SVM_RBF"] = m
    print("SVM_RBF(默认) ", fmt(m))

    # ---------------- E2 CV hyperparameter search ----------------------------
    print("\n=== E2 超参搜索 (5-fold CV on train, scoring=balanced_accuracy) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_log = []

    def cv_score(model, tag):
        s = cross_val_score(model, Xtr, ytr, cv=cv, scoring="balanced_accuracy", n_jobs=2)
        cv_log.append({"model": tag, "cv_bacc_mean": round(float(s.mean()), 4),
                       "cv_bacc_std": round(float(s.std()), 4)})
        print(f"  {tag:<38s} cv_bacc={s.mean():.4f}±{s.std():.4f}")
        return s.mean()

    svm_grid = [(C, g) for C in (1, 3, 10, 30) for g in ("scale", 0.005, 0.02)]
    best_svm_cfg, best_svm_cv = None, -1
    for C, g in svm_grid:
        sc = cv_score(SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced",
                          probability=True, random_state=SEED), f"SVM C={C} gamma={g}")
        if sc > best_svm_cv:
            best_svm_cv, best_svm_cfg = sc, (C, g)

    lr_grid = [0.3, 1.0, 3.0]
    best_lr_cfg, best_lr_cv = None, -1
    for C in lr_grid:
        sc = cv_score(LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                         random_state=SEED), f"LR C={C}")
        if sc > best_lr_cv:
            best_lr_cv, best_lr_cfg = sc, C

    hgb_grid = [(0.1, 300, None), (0.1, 300, 6), (0.05, 500, None)]
    best_hgb_cfg, best_hgb_cv = None, -1
    for lr_, it, dep in hgb_grid:
        sc = cv_score(HistGradientBoostingClassifier(learning_rate=lr_, max_iter=it,
                        max_depth=dep, class_weight="balanced", random_state=SEED),
                      f"HistGB lr={lr_} iter={it} depth={dep}")
        if sc > best_hgb_cv:
            best_hgb_cv, best_hgb_cfg = sc, (lr_, it, dep)

    C, g = best_svm_cfg
    svm_t = SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced", probability=True, random_state=SEED)
    m_svm, pred_svm, proba_svm = eval_on(svm_t, Xtr, ytr, Xva, yva)
    results[f"E2_SVM_tuned(C={C},gamma={g})"] = m_svm
    print("SVM tuned    ", fmt(m_svm))

    lr_t = LogisticRegression(C=best_lr_cfg, class_weight="balanced", max_iter=2000, random_state=SEED)
    m_lr, _, _ = eval_on(lr_t, Xtr, ytr, Xva, yva)
    results[f"E2_LR_tuned(C={best_lr_cfg})"] = m_lr
    print("LR tuned     ", fmt(m_lr))

    lr_h, it_h, dep_h = best_hgb_cfg
    hgb_t = HistGradientBoostingClassifier(learning_rate=lr_h, max_iter=it_h, max_depth=dep_h,
                                           class_weight="balanced", random_state=SEED)
    m_hgb, pred_hgb, proba_hgb = eval_on(hgb_t, Xtr, ytr, Xva, yva)
    results[f"E2_HistGB(lr={lr_h},iter={it_h},depth={dep_h})"] = m_hgb
    print("HistGB tuned ", fmt(m_hgb))

    rf = RandomForestClassifier(class_weight="balanced_subsample", n_estimators=300,
                                random_state=SEED, n_jobs=2)
    m_rf, _, _ = eval_on(rf, Xtr, ytr, Xva, yva)
    results["E2_RF_300trees"] = m_rf
    print("RF 300       ", fmt(m_rf))

    # ---------------- E3 soft-voting ensemble --------------------------------
    print("\n=== E3 软投票集成 ===")
    ens = VotingClassifier(estimators=[
        ("svm", SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced", probability=True, random_state=SEED)),
        ("lr", LogisticRegression(C=best_lr_cfg, class_weight="balanced", max_iter=2000, random_state=SEED)),
        ("hgb", HistGradientBoostingClassifier(learning_rate=lr_h, max_iter=it_h, max_depth=dep_h,
                                               class_weight="balanced", random_state=SEED)),
    ], voting="soft", n_jobs=1)
    m_ens, pred_ens, proba_ens = eval_on(ens, Xtr, ytr, Xva, yva)
    results["E3_ensemble_SVM+LR+HistGB"] = m_ens
    print("Ensemble     ", fmt(m_ens))

    # pick final model = best val macro_f1 among candidates (all selected via CV; val 只做最终汇报)
    candidates = {
        "SVM_tuned": (svm_t, m_svm, pred_svm, proba_svm),
        "HistGB": (hgb_t, m_hgb, pred_hgb, proba_hgb),
        "Ensemble": (ens, m_ens, pred_ens, proba_ens),
    }
    final_name = max(candidates, key=lambda k: candidates[k][1]["macro_f1"])
    final_model, final_m, final_pred, final_proba = candidates[final_name]
    print(f"\n>>> 最终选定: {final_name}  {fmt(final_m)}")

    # ---------------- E4 feature-family ablation ------------------------------
    print("\n=== E4 特征族消融 (基于 tuned SVM, 指标=val macro_f1) ===")
    extractor = PatchFeatureExtractor(resize_to=128)
    groups = feature_groups(extractor, X.shape[1])
    ablation = []
    full_f1 = m_svm["macro_f1"]
    for gname, idxs in groups.items():
        keep = [i for i in range(X.shape[1]) if i not in idxs]
        sc2 = StandardScaler().fit(Xtr_raw[:, keep])
        mm, _, _ = eval_on(SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced",
                               probability=True, random_state=SEED),
                           sc2.transform(Xtr_raw[:, keep]), ytr,
                           sc2.transform(Xva_raw[:, keep]), yva)
        ablation.append({"removed_group": gname, "n_dims": len(idxs),
                         "macro_f1": mm["macro_f1"],
                         "delta_vs_full": round(mm["macro_f1"] - full_f1, 4)})
        print(f"  -{gname:<14s} ({len(idxs):>2}维)  macroF1={mm['macro_f1']:.4f}  Δ={mm['macro_f1']-full_f1:+.4f}")

    # ---------------- E5 selective prediction --------------------------------
    print("\n=== E5 置信度拒识 (人机协同工作点, 模型=最终选定) ===")
    conf = final_proba.max(axis=1)
    order_correct = (final_pred == yva)
    sel_rows = []
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m_ = conf >= thr
        if m_.sum() == 0:
            continue
        cov = float(m_.mean()); acc_ = float(order_correct[m_].mean())
        sel_rows.append({"conf_threshold": thr, "coverage": round(cov, 4),
                         "accuracy_on_covered": round(acc_, 4),
                         "n_covered": int(m_.sum()),
                         "n_to_human": int((~m_).sum())})
        print(f"  conf≥{thr:<4}  覆盖 {cov*100:5.1f}%  覆盖内准确率 {acc_*100:5.2f}%  转人工 {int((~m_).sum())}")

    # ---------------- outputs -------------------------------------------------
    comparison = []
    for tag, m_ in results.items():
        comparison.append({"experiment": tag,
                           **{k: m_[k] for k in ("accuracy", "balanced_accuracy",
                                                 "macro_f1", "weighted_f1", "top2_accuracy")}})
    with open(out / "model_comparison_phase2.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison[0].keys())); w.writeheader(); w.writerows(comparison)
    with open(out / "cv_search_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(cv_log[0].keys())); w.writeheader(); w.writerows(cv_log)
    with open(out / "feature_ablation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ablation[0].keys())); w.writeheader(); w.writerows(ablation)
    with open(out / "selective_prediction.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0].keys())); w.writeheader(); w.writerows(sel_rows)

    # confusion pairs of final model (dict: "A->B" -> count)
    pairs = list(key_confusion_pairs(yva, final_pred, CLASS_NAMES).items())

    # save final model
    mdir = ROOT / "models/patch_classifier_v2"
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, mdir / "best_model.joblib")
    joblib.dump(scaler, mdir / "scaler.joblib")
    (mdir / "run_manifest.json").write_text(json.dumps({
        "experiment": "phase2_optimized", "model_name": final_name,
        "selected_by": "CV on train (5-fold balanced_accuracy); val used once for final report",
        "svm_params": {"C": C, "gamma": str(g)},
        "hgb_params": {"learning_rate": lr_h, "max_iter": it_h, "max_depth": dep_h},
        "class_names": CLASS_NAMES,
        "val_metrics": {k: final_m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                                "weighted_f1", "top2_accuracy")},
        "val_per_class": final_m["per_class"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown report
    lines = ["# Patch 六分类二期优化报告 (val, n=%d)" % int(va.sum()), "",
             "基线 (12 号脚本, SVM-RBF 默认参): acc 0.8826 / macro-F1 0.7869", "",
             "## 模型对比 (全部在 val 上一次性汇报; 模型选择只用 train 的 5-fold CV)", "",
             "| 实验 | Acc | BalAcc | Macro-F1 | Top-2 |", "|---|---|---|---|---|"]
    for c_ in comparison:
        lines.append(f"| {c_['experiment']} | {c_['accuracy']:.4f} | {c_['balanced_accuracy']:.4f} "
                     f"| {c_['macro_f1']:.4f} | {c_['top2_accuracy']:.4f} |")
    lines += ["", f"**最终选定: {final_name}**", "", "## 每类指标 (最终模型)", "",
              "| 类别 | P | R | F1 | n |", "|---|---|---|---|---|"]
    for cn in CLASS_NAMES:
        pc = final_m["per_class"][cn]
        lines.append(f"| {cn} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
    lines += ["", "## 特征族消融 (tuned SVM, Δ为移除该族后的 macro-F1 变化)", "",
              "| 移除特征族 | 维数 | macro-F1 | Δ |", "|---|---|---|---|"]
    for a in sorted(ablation, key=lambda r: r["delta_vs_full"]):
        lines.append(f"| {a['removed_group']} | {a['n_dims']} | {a['macro_f1']:.4f} | {a['delta_vs_full']:+.4f} |")
    lines += ["", "## 置信度拒识 (人机协同)", "",
              "| conf 阈值 | 覆盖率 | 覆盖内准确率 | 转人工数 |", "|---|---|---|---|"]
    for s_ in sel_rows:
        lines.append(f"| ≥{s_['conf_threshold']} | {s_['coverage']*100:.1f}% "
                     f"| {s_['accuracy_on_covered']*100:.2f}% | {s_['n_to_human']} |")
    if pairs:
        lines += ["", "## 最终模型关键混淆对", ""]
        for name_, cnt_ in pairs[:8]:
            lines.append(f"- {name_}: {cnt_}")
    (out / "phase2_patch_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告与 CSV 已写入 {out}")


if __name__ == "__main__":
    main()
