#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_cascade_detector_classifier.py — 检测 × Patch 分类级联融合实验

动机: patch 分类器"框内是什么"很强 (A/C/D F1≥0.93), 而检测器的
classification_error / A↔E 混淆 / 背景误检很多。将两者级联:

  阶段1  YOLO 检测框 (150ep, conf=0.001 全量导出) + 每类 F1 最优阈值 (来自 19 号脚本)
  阶段2  对每个检测框按类别自适应 context 裁剪 patch → 手工特征 → 分类器

融合策略 (在 val 上全部网格汇报, 不做单点挑选):
  P1  重判类 (relabel): 分类器置信度 ≥ τ_r 且与检测类别不一致时改判
  P2  背景否决 (veto): 额外用 N1-N3 patch 训练 7 类"守门员"模型,
      若判为 normal 且置信度 ≥ τ_v 则删除该检测框 (针对 2815 条背景误检)
  P1+P2 组合

评估: 与 19 号脚本相同的贪心匹配协议 (IoU≥0.5), 汇报 macro-F1 / 每类 P/R/F1 /
改判修正数 / A↔E 混淆变化。仅 val; test 不参与。

依赖: 先运行 18 号脚本 (提供 tuned SVM 超参与特征缓存)。

用法:
    python scripts/20_cascade_detector_classifier.py
"""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aoi_defect.patch_features import PatchFeatureExtractor  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

CLASSES = ["A", "B", "C", "D", "E", "F"]
CLASSES7 = CLASSES + ["N"]
SEED = 42
OUT = ROOT / "reports/optimization_phase2/cascade_fusion"

CTX = {"A": (1.2, 96), "B": (1.8, 96), "C": (3.0, 96), "D": (1.5, 128), "E": (1.8, 96), "F": (1.8, 96)}


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_eval(val_imgs, gt, plist_by_img):
    stats = {c: [0, 0, 0] for c in CLASSES}
    for img in val_imgs:
        for c in CLASSES:
            cp = sorted([p for p in plist_by_img.get(img, []) if p[0] == c], key=lambda t: -t[1])
            gts = list(gt.get((img, c), []))
            used = [False]*len(gts); tp = 0
            for _, _, box in cp:
                best, bi = 0.0, -1
                for gi, g in enumerate(gts):
                    if used[gi]:
                        continue
                    v = iou(box, g)
                    if v > best:
                        best, bi = v, gi
                if best >= 0.5 and bi >= 0:
                    used[bi] = True; tp += 1
            stats[c][0] += tp; stats[c][1] += len(cp)-tp; stats[c][2] += len(gts)-tp
    return stats


def prf(st):
    tp, fp, fn = st
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return p, r, (2*p*r/(p+r) if p+r else 0.0)


def macro_f1(stats):
    return float(np.mean([prf(stats[c])[2] for c in CLASSES]))


def spatial_gt_label(box, img, gt):
    """该预测框空间上最匹配的 GT 类别 (跨类, IoU≥0.3), 无则 None."""
    best, lab = 0.3, None
    for c in CLASSES:
        for g in gt.get((img, c), []):
            v = iou(box, g)
            if v >= best:
                best, lab = v, c
    return lab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- inputs
    thr = load_json(ROOT / "reports/optimization_phase2/detection_postprocess/postprocess_summary.json")[
        "per_class_thr_bestF1_oracle"]["thresholds"]
    v2 = load_json(ROOT / "models/patch_classifier_v2/run_manifest.json")

    def make_model():
        """按 18 号脚本最终选定的模型类型构建二阶段分类器."""
        name = v2.get("model_name", "SVM_tuned")
        if name == "HistGB":
            from sklearn.ensemble import HistGradientBoostingClassifier
            hp = v2["hgb_params"]
            return HistGradientBoostingClassifier(
                learning_rate=hp["learning_rate"], max_iter=hp["max_iter"],
                max_depth=hp["max_depth"], class_weight="balanced", random_state=SEED)
        C_ = v2["svm_params"]["C"]; g_ = v2["svm_params"]["gamma"]
        gamma = g_ if g_ == "scale" else float(g_)
        return SVC(kernel="rbf", C=C_, gamma=gamma, class_weight="balanced",
                   probability=True, random_state=SEED)

    print(f"per-class thresholds: {thr}\nstage-2 model: {v2.get('model_name')}")

    with open(ROOT / "data/derived/splits.csv", encoding="utf-8-sig", newline="") as f:
        split_map = {r["image_id"]: r["split"] for r in csv.DictReader(f)}
    val_imgs = [i for i, s in split_map.items() if s == "val"]

    gt = defaultdict(list)
    with open(ROOT / "data/annotations/instances.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if split_map.get(r["image_id"]) == "val" and r["label_code"] in CLASSES and r["is_pseudo_normal"] == "False":
                gt[(r["image_id"], r["label_code"])].append(
                    (float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])))

    preds = defaultdict(list)
    with open(ROOT / "reports/detection_evaluation/epoch150_20260720/prediction_instances.csv",
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if split_map.get(r["image_id"]) == "val":
                c = r["predicted_label"]; conf = float(r["confidence"])
                if conf >= thr[c]:
                    preds[r["image_id"]].append((c, conf,
                        (float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"]))))
    n_pred = sum(len(v) for v in preds.values())
    print(f"val preds after per-class thr: {n_pred}")

    # ---- train 7-class guard (A-F + N) on train patches, reuse cached 6-class features
    z = np.load(ROOT / "data/derived/features_handcrafted.npz", allow_pickle=True)
    X6, y6, sp6 = z["X"], z["y"], list(z["splits"])
    with open(ROOT / "data/derived/patch_manifest.csv", encoding="utf-8-sig", newline="") as f:
        man = list(csv.DictReader(f))
    n_rows = [r for r in man if r["train_label"] == "normal"]
    extractor = PatchFeatureExtractor(resize_to=128)
    Xn, spn = [], []
    for r in n_rows:
        fp = ROOT / r["raw_patch_path"]
        img = cv2.imdecode(np.frombuffer(fp.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((128, 128, 3), np.uint8)
        Xn.append(extractor.extract(img,
            bbox_width=float(r["bbox_width"]), bbox_height=float(r["bbox_height"]),
            bbox_area=float(r["bbox_area_px"]),
            patch_area=float(r["crop_width"])*float(r["crop_height"]),
            polygon_area=float(r["target_area_px"])))
        spn.append(r["split"])
    Xn = np.array(Xn, np.float32)
    print(f"N patches: {len(Xn)} (train {sum(1 for s in spn if s=='train')})")

    tr6 = np.array([s == "train" for s in sp6]); trn = np.array([s == "train" for s in spn])
    Xtr7 = np.vstack([X6[tr6], Xn[trn]])
    ytr7 = np.concatenate([y6[tr6], np.full(trn.sum(), 6)])
    sc7 = StandardScaler().fit(Xtr7)
    guard = make_model().fit(sc7.transform(Xtr7), ytr7)

    # 6-class relabel model (tuned SVM, train only)
    sc6 = StandardScaler().fit(X6[tr6])
    clf6 = make_model().fit(sc6.transform(X6[tr6]), y6[tr6])

    # ---- crop & classify every kept prediction
    print("cropping & classifying detector boxes ...")
    feats, meta = [], []
    for img_id in val_imgs:
        plist = preds.get(img_id, [])
        if not plist:
            continue
        im = cv2.imread(str(ROOT / f"data/images/{img_id}.jpg"))
        H, W = im.shape[:2]
        for (c, conf, box) in plist:
            x1, y1, x2, y2 = box
            bw, bh = x2-x1, y2-y1
            if bw < 2 or bh < 2:
                feats.append(None); meta.append((img_id, c, conf, box)); continue
            fac, mn = CTX[c]
            half = max(mn/2, fac*max(bw, bh)/2)
            cx, cy = (x1+x2)/2, (y1+y2)/2
            a1, b1 = int(max(0, cx-half)), int(max(0, cy-half))
            a2, b2 = int(min(W, cx+half)), int(min(H, cy+half))
            crop = im[b1:b2, a1:a2]
            if crop.size == 0:
                feats.append(None); meta.append((img_id, c, conf, box)); continue
            f_ = extractor.extract(crop, bbox_width=bw, bbox_height=bh, bbox_area=bw*bh,
                                   patch_area=(a2-a1)*(b2-b1), polygon_area=bw*bh)
            feats.append(f_); meta.append((img_id, c, conf, box))
    ok = [i for i, f_ in enumerate(feats) if f_ is not None]
    Xp = np.array([feats[i] for i in ok], np.float32)
    proba6 = np.zeros((len(feats), 6)); proba7 = np.zeros((len(feats), 7))
    proba6[ok] = clf6.predict_proba(sc6.transform(Xp))
    proba7[ok] = guard.predict_proba(sc7.transform(Xp))

    # ---- policies
    def build(policy, tau_r=0.6, tau_v=0.6):
        out = defaultdict(list)
        n_relabel = n_veto = 0
        for i, (img_id, c, conf, box) in enumerate(meta):
            cls6 = proba6[i]; cls7 = proba7[i]
            lab = c
            if policy in ("P2", "P12") and cls7[6] >= tau_v:
                n_veto += 1
                continue
            if policy in ("P1", "P12"):
                j = int(np.argmax(cls6))
                if CLASSES[j] != c and cls6[j] >= tau_r:
                    lab = CLASSES[j]; n_relabel += 1
            out[img_id].append((lab, conf, box))
        return out, n_relabel, n_veto

    # ---- P3: 检测框分布对齐的二阶段模型 (抖动 GT + 随机背景负样本, 仅 train 图) ----
    rng = np.random.RandomState(SEED)
    gt_train = defaultdict(list)
    with open(ROOT / "data/annotations/instances.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if split_map.get(r["image_id"]) == "train" and r["label_code"] in CLASSES and r["is_pseudo_normal"] == "False":
                gt_train[r["image_id"]].append((r["label_code"],
                    (float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"]))))
    print("building deployment-aligned训练集 (jittered GT + background negatives) ...")
    Xj, yj = [], []
    for img_id, items in gt_train.items():
        im = cv2.imread(str(ROOT / f"data/images/{img_id}.jpg"))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes_only = [b for _, b in items]
        for c, (x1, y1, x2, y2) in items:
            bw, bh = x2-x1, y2-y1
            for _ in range(1):
                sf = rng.uniform(0.8, 1.25)
                dx = rng.uniform(-0.15, 0.15)*bw; dy = rng.uniform(-0.15, 0.15)*bh
                nb = ( (x1+x2)/2 + dx - bw*sf/2, (y1+y2)/2 + dy - bh*sf/2,
                       (x1+x2)/2 + dx + bw*sf/2, (y1+y2)/2 + dy + bh*sf/2 )
                bw2, bh2 = nb[2]-nb[0], nb[3]-nb[1]
                if bw2 < 2 or bh2 < 2:
                    continue
                fac, mn = CTX[c]
                half = max(mn/2, fac*max(bw2, bh2)/2)
                cx, cy = (nb[0]+nb[2])/2, (nb[1]+nb[3])/2
                a1, b1 = int(max(0, cx-half)), int(max(0, cy-half))
                a2, b2 = int(min(W, cx+half)), int(min(H, cy+half))
                crop = im[b1:b2, a1:a2]
                if crop.size == 0:
                    continue
                Xj.append(extractor.extract(crop, bbox_width=bw2, bbox_height=bh2,
                    bbox_area=bw2*bh2, patch_area=(a2-a1)*(b2-b1), polygon_area=bw2*bh2))
                yj.append(CLASSES.index(c))
        # background negatives: 4 per image, sized like random GT, IoU<0.05 vs all GT
        for _ in range(4):
            for _try in range(30):
                _, (rx1, ry1, rx2, ry2) = items[rng.randint(len(items))]
                bw2, bh2 = rx2-rx1, ry2-ry1
                cx = rng.uniform(bw2/2, W-bw2/2); cy = rng.uniform(bh2/2, H-bh2/2)
                nb = (cx-bw2/2, cy-bh2/2, cx+bw2/2, cy+bh2/2)
                if all(iou(nb, b) < 0.05 for b in boxes_only):
                    half = max(48, 1.8*max(bw2, bh2)/2)
                    a1, b1 = int(max(0, cx-half)), int(max(0, cy-half))
                    a2, b2 = int(min(W, cx+half)), int(min(H, cy+half))
                    crop = im[b1:b2, a1:a2]
                    if crop.size == 0:
                        break
                    Xj.append(extractor.extract(crop, bbox_width=bw2, bbox_height=bh2,
                        bbox_area=bw2*bh2, patch_area=(a2-a1)*(b2-b1), polygon_area=bw2*bh2))
                    yj.append(6)
                    break
    Xj = np.array(Xj, np.float32); yj = np.array(yj)
    # + 128 N patches as extra normal
    Xtr_al = np.vstack([Xj, Xn[trn]]); ytr_al = np.concatenate([yj, np.full(int(trn.sum()), 6)])
    print(f"aligned train set: {len(ytr_al)} (bg {int((ytr_al==6).sum())})")
    sc_al = StandardScaler().fit(Xtr_al)
    clf_al = make_model().fit(sc_al.transform(Xtr_al), ytr_al)
    proba_al = np.zeros((len(feats), 7))
    proba_al[ok] = clf_al.predict_proba(sc_al.transform(Xp))

    def build_p3(tau_r, tau_v):
        out = defaultdict(list); n_re = n_ve = 0
        for i, (img_id, c, conf, box) in enumerate(meta):
            pr = proba_al[i]
            if pr[6] >= tau_v:
                n_ve += 1; continue
            lab = c
            j = int(np.argmax(pr[:6]))
            if CLASSES[j] != c and pr[j] >= tau_r:
                lab = CLASSES[j]; n_re += 1
            out[img_id].append((lab, conf, box))
        return out, n_re, n_ve

    base_stats = greedy_eval(val_imgs, gt, preds)
    rows = [{"policy": "P0 仅每类阈值", "tau": "-", "macroF1": round(macro_f1(base_stats), 4),
             "relabel": 0, "veto": 0,
             **{f"F1_{c}": round(prf(base_stats[c])[2], 4) for c in CLASSES}}]
    print(f"P0 baseline macroF1={macro_f1(base_stats):.4f}")

    grids = ([("P1", tr_, None) for tr_ in (0.5, 0.6, 0.7, 0.8)] +
             [("P2", None, tv) for tv in (0.5, 0.6, 0.8)] +
             [("P12", 0.6, 0.6), ("P12", 0.7, 0.8)])
    p3_grids = [(0.6, 0.5), (0.6, 0.7), (0.8, 0.7), (0.8, 0.9), (1.1, 0.5), (1.1, 0.7), (1.1, 0.9)]
    for pol, tr_, tv in grids:
        plists, nr, nv = build(pol, tau_r=tr_ or 0.6, tau_v=tv or 0.6)
        st = greedy_eval(val_imgs, gt, plists)
        tag = {"P1": f"P1 重判类 τr={tr_}", "P2": f"P2 背景否决 τv={tv}",
               "P12": f"P1+P2 τr={tr_} τv={tv}"}[pol]
        rows.append({"policy": tag, "tau": f"{tr_}/{tv}", "macroF1": round(macro_f1(st), 4),
                     "relabel": nr, "veto": nv,
                     **{f"F1_{c}": round(prf(st[c])[2], 4) for c in CLASSES}})
        print(f"{tag:<24s} macroF1={macro_f1(st):.4f} (Δ{macro_f1(st)-macro_f1(base_stats):+.4f}) "
              f"relabel={nr} veto={nv}")

    for tau_r, tau_v in p3_grids:
        plists, nr, nv = build_p3(tau_r, tau_v)
        st = greedy_eval(val_imgs, gt, plists)
        mode = "否决" if tau_r > 1 else "重判+否决"
        tag = f"P3 对齐重训 {mode} τr={tau_r} τv={tau_v}"
        rows.append({"policy": tag, "tau": f"{tau_r}/{tau_v}", "macroF1": round(macro_f1(st), 4),
                     "relabel": nr, "veto": nv,
                     **{f"F1_{c}": round(prf(st[c])[2], 4) for c in CLASSES}})
        print(f"{tag:<34s} macroF1={macro_f1(st):.4f} (Δ{macro_f1(st)-macro_f1(base_stats):+.4f}) "
              f"relabel={nr} veto={nv}")

    # ---- relabel correctness audit (τr=0.6): 改判后与空间 GT 一致率
    fix_good = fix_bad = fix_bg = 0
    ae_before = ae_after = 0
    for i, (img_id, c, conf, box) in enumerate(meta):
        sg = spatial_gt_label(box, img_id, gt)
        j = int(np.argmax(proba6[i])); newc = CLASSES[j] if proba6[i][j] >= 0.6 else c
        if sg is not None and c != sg and ((c == "A" and sg == "E") or (c == "E" and sg == "A")):
            ae_before += 1
            if newc == sg:
                pass
            else:
                ae_after += 1
        if newc != c:
            if sg is None:
                fix_bg += 1
            elif newc == sg:
                fix_good += 1
            else:
                fix_bad += 1
    audit = {"relabel_correct(改对)": fix_good, "relabel_wrong(改错)": fix_bad,
             "relabel_on_background(无GT区域)": fix_bg,
             "A_E_spatial_confusions_before": ae_before, "A_E_remaining_after_relabel": ae_after}
    print("relabel audit:", audit)

    with open(OUT / "cascade_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (OUT / "cascade_summary.json").write_text(json.dumps({
        "n_val_preds_after_thr": n_pred, "policies": rows, "relabel_audit_tau0.6": audit,
        "guard_train": {"defect_patches": int(tr6.sum()), "normal_patches": int(trn.sum())},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 检测 × Patch 分类级联融合报告 (val · 150-epoch 基线)", "",
             "阶段1: 检测框 (每类 F1 最优阈值); 阶段2: 框内 patch → 手工特征 → tuned SVM。",
             "P1=分类器重判类; P2=七类守门员(含 N1-N3 正常类)背景否决; 全网格汇报无单点挑选。", "",
             "| 策略 | macro-F1 | 改判数 | 否决数 | " + " | ".join(f"F1_{c}" for c in CLASSES) + " |",
             "|---|---|---|---|" + "---|"*6]
    for r_ in rows:
        lines.append(f"| {r_['policy']} | {r_['macroF1']:.4f} | {r_['relabel']} | {r_['veto']} | "
                     + " | ".join(f"{r_[f'F1_{c}']:.4f}" for c in CLASSES) + " |")
    lines += ["", "## 改判正确性审计 (τr=0.6, 以空间重叠 GT 为参照)", ""]
    for k, v_ in audit.items():
        lines.append(f"- {k}: {v_}")
    (OUT / "cascade_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 {OUT}")


if __name__ == "__main__":
    main()
