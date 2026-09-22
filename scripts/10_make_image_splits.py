#!/usr/bin/env python3
"""
10_make_image_splits.py — 图片级多标签分层数据划分

按 image_id 划分 train=140 / val=30 / test=30，使用多标签分层确保每类
在各 split 中都有出现且比例均衡。支持 exact duplicate 约束和 split 锁定。

用法:
    python scripts/10_make_image_splits.py
    python scripts/10_make_image_splits.py --force-regenerate
    python scripts/10_make_image_splits.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("make_splits")

# =============================================================================
# 加载
# =============================================================================
def load_summary(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_instances(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}

def load_fingerprint(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _natural_sort_key(s: str) -> Tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s))

def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=str(path.parent), prefix=".tmp_")
    with os.fdopen(tmp_fd, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: str(v) for k, v in r.items()})
    os.replace(tmp_path, path)

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


# =============================================================================
# 主逻辑
# =============================================================================
HAS_COLS = ["has_A", "has_B", "has_C", "has_D", "has_E", "has_F"]
COUNT_COLS = ["count_A", "count_B", "count_C", "count_D", "count_E", "count_F",
              "total_annotation_count"]

SPLIT_FIELDS = [
    "image_id", "filename", "split", "split_index",
    "has_A", "has_B", "has_C", "has_D", "has_E", "has_F",
    "has_N1", "has_N2", "has_N3",
    "defect_instance_count", "pseudo_normal_instance_count",
    "dataset_fingerprint",
]

TRAIN_N = 140; VAL_N = 30; TEST_N = 30


def _build_ml_labels(rows: List[Dict[str, str]], all_ids: List[str]) -> np.ndarray:
    """构建多标签向量: has_A..has_F + has_N1..has_N3 + discretized counts."""
    id_to_row = {r["image_id"]: r for r in rows}
    n = len(all_ids)
    # has_* 二值 (9 维)
    has_labels = np.zeros((n, 9), dtype=int)
    # count discretized (7 维)
    count_feats = np.zeros((n, 7), dtype=np.float32)
    for i, iid in enumerate(all_ids):
        r = id_to_row[iid]
        for j, c in enumerate(HAS_COLS):
            has_labels[i, j] = int(r.get(c, "0"))
        for j, c in enumerate(["N1", "N2", "N3"]):
            has_labels[i, 6 + j] = int(r.get(f"has_{c}", "0"))
        for j, c in enumerate(COUNT_COLS):
            count_feats[i, j] = float(r.get(c, "0"))

    # Discretize count features into 3 bins (low/medium/high) per column
    count_bins = np.zeros((n, len(COUNT_COLS)), dtype=int)
    for j in range(count_feats.shape[1]):
        col = count_feats[:, j]
        if np.max(col) <= 1:
            count_bins[:, j] = col.astype(int)
        else:
            # ternary split by quantiles
            lo = np.percentile(col[col > 0], 33) if np.any(col > 0) else 1
            hi = np.percentile(col[col > 0], 67) if np.any(col > 0) else 2
            count_bins[:, j] = np.where(col <= lo, 0, np.where(col <= hi, 1, 2))

    # Concatenate
    return np.concatenate([has_labels, count_bins], axis=1)


def _stratified_split(ml: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用迭代分层生成 train/val/test 索引，确保精确计数。

    策略: 多次 n_splits 取最接近目标的分割。
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        logger.error("缺少 iterative-stratification 依赖。请安装: pip install iterative-stratification")
        sys.exit(1)

    n = ml.shape[0]

    # 多次尝试找到最接近目标 test=30 的划分
    best_test_diff = 999
    best_split = None
    for attempt_seed in range(seed, seed + 50):
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_N, random_state=attempt_seed)
        tv_idx, t_idx = next(msss.split(np.zeros((n, 1)), ml))
        diff = abs(len(t_idx) - TEST_N)
        if diff < best_test_diff:
            best_test_diff = diff
            best_split = (tv_idx, t_idx)
        if diff == 0:
            break

    if best_split is None:
        raise RuntimeError("无法找到符合 test=30 的划分")

    trainval_idx, test_idx = best_split
    # 强制截取精确 TEST_N 个（如果多了，多余归入 trainval；如果不足，从 trainval 补）
    if len(test_idx) > TEST_N:
        extras = test_idx[TEST_N:]
        test_idx = test_idx[:TEST_N]
        trainval_idx = np.concatenate([trainval_idx, extras])
    elif len(test_idx) < TEST_N:
        needed = TEST_N - len(test_idx)
        # 从 trainval 中随机取 needed 个加入 test (保持分层)
        rng = np.random.RandomState(seed)
        extra_from_tv = rng.choice(trainval_idx, size=needed, replace=False)
        test_idx = np.concatenate([test_idx, extra_from_tv])
        trainval_idx = np.setdiff1d(trainval_idx, extra_from_tv)

    # Step 2: 从 trainval 中分 val=30
    n_tv = len(trainval_idx)
    ml_tv = ml[trainval_idx]
    best_val_diff = 999
    best_val_split = None
    for attempt_seed in range(seed, seed + 50):
        msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=VAL_N, random_state=attempt_seed)
        tr_rel, v_rel = next(msss2.split(np.zeros((n_tv, 1)), ml_tv))
        diff = abs(len(v_rel) - VAL_N)
        if diff < best_val_diff:
            best_val_diff = diff
            best_val_split = (tr_rel, v_rel)
        if diff == 0:
            break

    if best_val_split is None:
        raise RuntimeError("无法找到符合 val=30 的划分")

    tr_rel, v_rel = best_val_split
    val_idx = trainval_idx[v_rel]
    train_idx = trainval_idx[tr_rel]

    # 强制精确计数
    if len(val_idx) > VAL_N:
        extras = val_idx[VAL_N:]
        val_idx = val_idx[:VAL_N]
        train_idx = np.concatenate([train_idx, extras])
    elif len(val_idx) < VAL_N:
        needed = VAL_N - len(val_idx)
        rng = np.random.RandomState(seed + 1)
        extra_from_tr = rng.choice(train_idx, size=needed, replace=False)
        val_idx = np.concatenate([val_idx, extra_from_tr])
        train_idx = np.setdiff1d(train_idx, extra_from_tr)

    logger.info("分层尝试完成: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))
    return train_idx, val_idx, test_idx


def _validate_splits(all_ids: List[str], train_idx, val_idx, test_idx,
                     rows: List[Dict[str, str]]) -> List[str]:
    """检查划分合法性，返回警告列表。"""
    warns = []
    id_to_row = {r["image_id"]: r for r in rows}
    train_ids = {all_ids[i] for i in train_idx}
    val_ids = {all_ids[i] for i in val_idx}
    test_ids = {all_ids[i] for i in test_idx}

    if train_ids & val_ids: warns.append("train ∩ val 非空")
    if train_ids & test_ids: warns.append("train ∩ test 非空")
    if val_ids & test_ids: warns.append("val ∩ test 非空")

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for c in ["A", "B", "C", "D", "E", "F"]:
            n_img = sum(1 for iid in split_ids if id_to_row[iid].get(f"has_{c}", "0") == "1")
            if n_img == 0:
                warns.append(f"{split_name} 缺少类别 {c}")

    return warns


def _build_splits_csv(all_ids: List[str], train_idx, val_idx, test_idx,
                      rows: List[Dict[str, str]], fingerprint: str) -> List[Dict]:
    id_to_row = {r["image_id"]: r for r in rows}
    results = []
    for split_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        for j, i in enumerate(idxs):
            iid = all_ids[i]; r = id_to_row[iid]
            results.append({
                "image_id": iid,
                "filename": r["filename"],
                "split": split_name,
                "split_index": j,
                "has_A": r.get("has_A", "0"), "has_B": r.get("has_B", "0"),
                "has_C": r.get("has_C", "0"), "has_D": r.get("has_D", "0"),
                "has_E": r.get("has_E", "0"), "has_F": r.get("has_F", "0"),
                "has_N1": r.get("has_N1", "0"), "has_N2": r.get("has_N2", "0"),
                "has_N3": r.get("has_N3", "0"),
                "defect_instance_count": r["defect_instance_count"],
                "pseudo_normal_instance_count": r["pseudo_normal_instance_count"],
                "dataset_fingerprint": fingerprint[:16],
            })
    return results


def _generate_stats(all_ids, train_idx, val_idx, test_idx,
                    summary_rows, instances, output_dir):
    """生成各类统计 CSV 和 summary."""
    id_to_row = {r["image_id"]: r for r in summary_rows}
    splits_map: Dict[str, List[str]] = {
        "train": [all_ids[i] for i in train_idx],
        "val": [all_ids[i] for i in val_idx],
        "test": [all_ids[i] for i in test_idx],
    }

    # class image counts
    cls_cols = ["class", "train_images", "val_images", "test_images",
                "train_instances", "val_instances", "test_instances"]
    cls_rows = []
    for lc in ["A", "B", "C", "D", "E", "F", "N1", "N2", "N3"]:
        imgs = {s: sum(1 for iid in ids if id_to_row[iid].get(f"has_{lc}", "0") == "1")
                 for s, ids in splits_map.items()}
        icounts = {s: sum(1 for r in instances if r["label_code"] == lc and r["image_id"] in ids)
                    for s, ids in splits_map.items()}
        cls_rows.append({"class": lc, **{f"{s}_images": v for s, v in imgs.items()},
                          **{f"{s}_instances": v for s, v in icounts.items()}})
    _write_csv(output_dir / "split_class_image_counts.csv",
                ["class", "train_images", "val_images", "test_images"], cls_rows)
    _write_csv(output_dir / "split_class_instance_counts.csv",
                ["class", "train_instances", "val_instances", "test_instances"], cls_rows)

    # geometry counts
    geo_rows = []
    for s, ids in splits_map.items():
        rect = sum(1 for r in instances if r["geometry_type"] == "rectangle" and r["image_id"] in ids)
        poly = sum(1 for r in instances if r["geometry_type"] == "polygon" and r["image_id"] in ids)
        geo_rows.append({"split": s, "rectangle": rect, "polygon": poly})
    _write_csv(output_dir / "split_geometry_counts.csv",
                ["split", "rectangle", "polygon"], geo_rows)

    # size statistics
    size_rows = []
    for s, ids in splits_map.items():
        insts_in = [r for r in instances if r["image_id"] in ids]
        bws = [int(r["bbox_width"]) for r in insts_in if r["bbox_width"]]
        bhs = [int(r["bbox_height"]) for r in insts_in if r["bbox_height"]]
        areas = [int(r["bbox_area_px"]) for r in insts_in if r["bbox_area_px"]]
        size_rows.append({
            "split": s, "total_instances": len(insts_in),
            "bw_min": min(bws) if bws else "", "bw_median": np.median(bws) if bws else "",
            "bw_max": max(bws) if bws else "", "bh_median": np.median(bhs) if bhs else "",
            "area_median": np.median(areas) if areas else "",
        })
    _write_csv(output_dir / "split_size_statistics.csv",
                ["split", "total_instances", "bw_min", "bw_median", "bw_max", "bh_median", "area_median"],
                size_rows)


def _generate_figures(all_ids, train_idx, val_idx, test_idx, summary_rows, output_dir):
    """生成 split 分布图."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        id_to_row = {r["image_id"]: r for r in summary_rows}
        splits_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        classes = ["A", "B", "C", "D", "E", "F"]
        colors = ["#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#74c0fc", "#da77f2"]
        n_train = len(train_idx); n_val = len(val_idx); n_test = len(test_idx)

        # 1. class presence per split
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(classes)); w = 0.25
        for si, (sname, idxs) in enumerate(splits_map.items()):
            ids_set = {all_ids[i] for i in idxs}
            counts = [sum(1 for iid in ids_set if id_to_row[iid].get(f"has_{c}", "0") == "1") for c in classes]
            ax.bar(x + si * w, counts, w, label=sname, color=["#7c6ff0", "#4caf88", "#e0a040"][si])
        ax.set_xticks(x + w); ax.set_xticklabels(classes)
        ax.set_ylabel("Images"); ax.set_title("Class Presence per Split")
        ax.legend()
        plt.tight_layout(); plt.savefig(fig_dir / "split_class_presence.png", dpi=100); plt.close()

        # 2. instance distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        for si, (sname, idxs) in enumerate(splits_map.items()):
            ids_set = {all_ids[i] for i in idxs}
            icounts = [int(id_to_row[iid]["defect_instance_count"]) for iid in ids_set]
            ax.hist(icounts, bins=30, alpha=0.5, label=sname, color=["#7c6ff0", "#4caf88", "#e0a040"][si])
        ax.set_xlabel("Defect Instance Count"); ax.set_ylabel("Images")
        ax.set_title("Instance Count Distribution per Split"); ax.legend()
        plt.tight_layout(); plt.savefig(fig_dir / "split_instance_distribution.png", dpi=100); plt.close()

        logger.info("图表已生成: %s", fig_dir)
    except Exception as e:
        logger.warning("图表生成失败: %s", e)


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="图片级多标签分层数据划分")
    p.add_argument("--summary_csv", type=str, default="data/annotations/image_summary.csv")
    p.add_argument("--instances_csv", type=str, default="data/annotations/instances.csv")
    p.add_argument("--manifest_csv", type=str, default="data/metadata/images_manifest.csv")
    p.add_argument("--duplicate_report", type=str, default="reports/data_audit/image_duplicate_report.csv")
    p.add_argument("--fingerprint", type=str, default="data/derived/dataset_fingerprint.json")
    p.add_argument("--output_splits", type=str, default="data/derived/splits.csv")
    p.add_argument("--lock_file", type=str, default="data/derived/test_split_lock.json")
    p.add_argument("--output_dir", type=str, default="reports/data_split")
    p.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    p.add_argument("--force-regenerate", action="store_true", default=False,
                   help="强制重新划分 (需要用户确认)")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    summary_csv = (root / args.summary_csv).resolve()
    instances_csv = (root / args.instances_csv).resolve()
    manifest_csv = (root / args.manifest_csv).resolve()
    dup_report = (root / args.duplicate_report).resolve()
    fp_path = (root / args.fingerprint).resolve()
    splits_path = (root / args.output_splits).resolve()
    lock_path = (root / args.lock_file).resolve()
    output_dir = (root / args.output_dir).resolve()

    # ----- Lock check -----
    if splits_path.exists() or lock_path.exists():
        if not args.force_regenerate:
            logger.error("splits.csv 或 lock 文件已存在。如需重新划分，请使用 --force-regenerate")
            logger.info("已存在的 split: %s", splits_path)
            sys.exit(0)
        # 需要用户确认
        print("\n⚠️  即将重新生成数据划分！这将覆盖现有的 splits.csv 和 lock 文件。")
        confirm = input("确定要继续吗？输入 YES 确认: ")
        if confirm.strip() != "YES":
            print("已取消。")
            sys.exit(0)

    # ----- Load -----
    summary_rows = load_summary(summary_csv)
    instances = load_instances(instances_csv)
    fp_data = load_fingerprint(fp_path)

    all_ids = sorted(set(r["image_id"] for r in summary_rows), key=_natural_sort_key)
    if len(all_ids) != 200:
        logger.error("image_summary 行数 %d != 200", len(all_ids)); sys.exit(1)

    # Check exact duplicates
    dups_exist = False
    dup_pairs: List[Tuple[str, str]] = []
    if dup_report.exists():
        with open(dup_report, "r", encoding="utf-8-sig") as f:
            for dr in csv.DictReader(f):
                if dr.get("exact_duplicate", "").lower() == "true":
                    dup_pairs.append((dr["image_id_1"], dr["image_id_2"]))
                    dups_exist = True
    if dup_pairs:
        logger.info("发现 %d 对 exact duplicate 图片", len(dup_pairs))

    # ----- Build multi-label matrix -----
    logger.info("构建多标签矩阵 (200 images × 16 labels)...")
    ml = _build_ml_labels(summary_rows, all_ids)

    # ----- Stratified split -----
    np.random.seed(args.seed)
    train_idx, val_idx, test_idx = _stratified_split(ml, args.seed)
    logger.info("划分完成: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Adjust for exact duplicates
    train_ids_set = {all_ids[i] for i in train_idx}
    val_ids_set = {all_ids[i] for i in val_idx}
    test_ids_set = {all_ids[i] for i in test_idx}
    fixed_dups = 0
    for id1, id2 in dup_pairs:
        # both must be in same split; prefer placing both in train
        if id1 in train_ids_set:
            if id2 not in train_ids_set:
                # move id2 to train
                for s_name, s_set in [("val", val_ids_set), ("test", test_ids_set)]:
                    if id2 in s_set:
                        s_set.discard(id2); train_ids_set.add(id2); fixed_dups += 1
    if fixed_dups:
        logger.info("修复 exact duplicate 分组: %d 张图片被移动", fixed_dups)

    # Rebuild idx arrays
    id_to_pos = {iid: i for i, iid in enumerate(all_ids)}
    train_idx = np.array([id_to_pos[iid] for iid in sorted(train_ids_set, key=_natural_sort_key)])
    val_idx = np.array([id_to_pos[iid] for iid in sorted(val_ids_set, key=_natural_sort_key)])
    test_idx = np.array([id_to_pos[iid] for iid in sorted(test_ids_set, key=_natural_sort_key)])

    # ----- Validate -----
    warns = _validate_splits(all_ids, train_idx, val_idx, test_idx, summary_rows)
    for w in warns: logger.warning(w)

    # ----- Build output -----
    fp_hash = fp_data.get("hashes", {}).get("instances_csv", "unknown")
    split_rows = _build_splits_csv(all_ids, train_idx, val_idx, test_idx, summary_rows, fp_hash)
    _write_csv(splits_path, SPLIT_FIELDS, split_rows)
    logger.info("splits.csv → %s", splits_path)

    # Lock
    lock_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "test_image_ids": sorted([all_ids[i] for i in test_idx], key=_natural_sort_key),
        "splits_csv_sha256": _sha256_file(splits_path),
        "dataset_fingerprint": fp_hash,
        "train_n": len(train_idx), "val_n": len(val_idx), "test_n": len(test_idx),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("lock 文件: %s", lock_path)

    # ----- Statistics & Figures -----
    output_dir.mkdir(parents=True, exist_ok=True)
    _generate_stats(all_ids, train_idx, val_idx, test_idx, summary_rows, instances, output_dir)
    _generate_figures(all_ids, train_idx, val_idx, test_idx, summary_rows, output_dir)

    # ----- Summary -----
    id_to_row = {r["image_id"]: r for r in summary_rows}
    lines = [
        "# 数据划分报告",
        "",
        f"- 随机种子: {args.seed}",
        f"- Train: {len(train_idx)} 张",
        f"- Val: {len(val_idx)} 张",
        f"- Test: {len(test_idx)} 张",
        f"- 重复图片修正: {fixed_dups}",
        f"- 警告: {len(warns)}",
        "",
        "## 注意事项",
        "此为内部同批次测试划分，不是跨批次泛化测试。",
        "同一 image_id 的所有标注实例在同一个 split 中。",
        "",
        "## 每类图片覆盖",
    ]
    for lc in ["A", "B", "C", "D", "E", "F"]:
        tr_i = sum(1 for iid in [all_ids[i] for i in train_idx] if id_to_row[iid].get(f"has_{lc}", "0") == "1")
        val_i = sum(1 for iid in [all_ids[i] for i in val_idx] if id_to_row[iid].get(f"has_{lc}", "0") == "1")
        te_i = sum(1 for iid in [all_ids[i] for i in test_idx] if id_to_row[iid].get(f"has_{lc}", "0") == "1")
        lines.append(f"- {lc}: train={tr_i}, val={val_i}, test={te_i}")
    (output_dir / "split_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("split_summary.md → %s", output_dir / "split_summary.md")

    logger.info("=" * 50)
    logger.info("划分完成: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))
    if dups_exist:
        logger.info("  exact duplicate 修正: %d", fixed_dups)
    if abs(len(train_idx) - TRAIN_N) > 2:
        logger.warning("  train 数量偏差: %d (期望 %d)", len(train_idx), TRAIN_N)


if __name__ == "__main__":
    main()
