#!/usr/bin/env python3
"""
07_validate_annotations.py — 校验 annotations.json 与导出 CSV 一致性

校验规则:
  1.  image_id 在 manifest 中 | 2. filename 匹配 | 3. label_code 合法
  4.  label 与 label_code 一致 | 5. geometry_type 合法
  6.  rectangle 有合法 bbox | 7. polygon ≥3 点 | 8/9. bbox 不越界
  10. points 不越界 | 11. x_min<x_max, y_min<y_max
  12. bbox_width/height>0 | 13. area_px>0
  14. is_pseudo_normal 与类别一致 | 15. ann_id 唯一
  16. 图片状态合法 | 17. 实例数与验收值一致

用法:
    python scripts/07_validate_annotations.py
"""

from __future__ import annotations

import argparse, csv, json, logging, re, sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("validate")

VALID_DEFECT = {"A","B","C","D","E","F","G"}
VALID_NORMAL = {"N1","N2","N3"}
VALID_CODES = VALID_DEFECT | VALID_NORMAL
VALID_GEOM = {"rectangle","polygon"}
VALID_STATUSES = {"unstarted","in_progress","done","skipped","uncertain"}

EXPECTED_TOTAL_INSTANCES = 4005
EXPECTED_DEFECT = 3807
EXPECTED_NORMAL = 198
EXPECTED_COUNTS = {"A":638,"B":271,"C":1852,"D":388,"E":234,"F":424,"G":0,"N1":26,"N2":135,"N3":37}

def load_json(p: Path) -> Dict: return json.loads(p.read_text(encoding="utf-8"))
def load_manifest(p: Path) -> Dict[str, Dict[str,str]]:
    if not p.exists(): return {}
    with open(p,"r",encoding="utf-8-sig",newline="") as f: return {r["image_id"]:r for r in csv.DictReader(f)}
def load_schema(p: Path) -> Dict: return yaml.safe_load(p.read_text(encoding="utf-8"))
def build_label_map(schema: Dict) -> Dict[str,str]:
    m={}
    for sec in ("defect_classes","normal_classes"):
        for c,info in schema.get(sec,{}).items(): m[c]=info.get("name","")
    return m
def _nk(s:str)->Tuple: return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)",s))


def read_csv_if_exists(p:Path)->Optional[List[Dict[str,str]]]:
    if not p.exists(): return None
    with open(p,"r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def validate_all(ad:Dict, mm:Dict[str,Dict[str,str]], lm:Dict[str,str])->Tuple[List[str],List[str],Dict[str,Any]]:
    errors:List[str]=[]; warnings:List[str]=[]
    images=ad.get("images",{})
    cat_counts:Counter=Counter()
    total_defect=0; total_normal=0; rect_count=0; poly_count=0
    status_counts:Counter=Counter()

    for iid in sorted(images.keys(),key=_nk):
        rec=images[iid]; pfx=f"[{iid}]"
        if not isinstance(rec,dict): errors.append(f"{pfx}: 不是 dict"); continue
        mr=mm.get(iid)
        if mr is None: errors.append(f"{pfx}: 不在 manifest 中")
        mf=mr.get("filename","") if mr else ""
        rf=rec.get("filename","")
        if rf and mf and rf!=mf: errors.append(f"{pfx}: filename 不匹配 '{rf}' vs '{mf}'")
        iw=int(mr.get("width",0)) if mr else 0; ih=int(mr.get("height",0)) if mr else 0
        st=rec.get("status",""); status_counts[st]=status_counts.get(st,0)+1
        if st and st not in VALID_STATUSES: errors.append(f"{pfx}: 无效 status='{st}'")
        anns=rec.get("annotations",[])
        ann_ids_in_img:List[str]=[]
        for ann in anns:
            aid=ann.get("ann_id","?"); apfx=f"{pfx}/{aid}"
            lc=ann.get("label_code","")
            if lc not in VALID_CODES: errors.append(f"{apfx}: 无效 label_code='{lc}'"); continue
            exp=lm.get(lc,""); act=ann.get("label","")
            if exp and act!=exp: errors.append(f"{apfx}: label 不匹配 '{act}' vs '{exp}'")
            if lc in VALID_NORMAL and not ann.get("is_pseudo_normal"): errors.append(f"{apfx}: N* 但 is_pseudo_normal=False")
            if lc in VALID_DEFECT and ann.get("is_pseudo_normal"): errors.append(f"{apfx}: A-G 但 is_pseudo_normal=True")
            cat_counts[lc]+=1
            if ann.get("is_pseudo_normal"): total_normal+=1
            else: total_defect+=1
            gt=ann.get("geometry_type","")
            if gt not in VALID_GEOM: errors.append(f"{apfx}: 无效 geometry_type='{gt}'")
            if gt=="rectangle": rect_count+=1
            elif gt=="polygon": poly_count+=1
            bbox=ann.get("bbox",[])
            if gt=="rectangle" and len(bbox)!=4: errors.append(f"{apfx}: rectangle bbox 长度≠4")
            if len(bbox)==4:
                x1,y1,x2,y2=bbox
                if x1>=x2: errors.append(f"{apfx}: x_min>=x_max")
                if y1>=y2: errors.append(f"{apfx}: y_min>=y_max")
                bw,bh=x2-x1,y2-y1
                if bw<=0 or bh<=0: errors.append(f"{apfx}: bbox 宽高 ≤0")
                if iw>0 and ih>0:
                    if x1<0 or y1<0 or x2>iw or y2>ih: errors.append(f"{apfx}: bbox 越界")
            pts=ann.get("points",[])
            if gt=="polygon" and len(pts)<3: errors.append(f"{apfx}: polygon <3 点")
            if pts and iw>0 and ih>0:
                for pi,p in enumerate(pts):
                    if len(p)==2 and (p[0]<0 or p[1]<0 or p[0]>iw or p[1]>ih): errors.append(f"{apfx}: point[{pi}] 越界")
            area=ann.get("area_px",0)
            if area is not None and float(area)<=0: errors.append(f"{apfx}: area_px ≤0")
            if not aid.startswith(f"{iid}_ann_"): errors.append(f"{apfx}: ann_id 格式异常")
            ann_ids_in_img.append(aid)
        # ann_id 图内唯一
        dupes=[a for a in set(ann_ids_in_img) if ann_ids_in_img.count(a)>1]
        for d in dupes: errors.append(f"{pfx}: 重复 ann_id={d}")
        if st=="done" and not any(not a.get("is_pseudo_normal") for a in anns): warnings.append(f"{pfx}: done 但无缺陷")
        if st=="unstarted" and anns: warnings.append(f"{pfx}: unstarted 但有 {len(anns)} 条标注")

    # 全局唯一 ann_id
    all_aids=[a.get("ann_id","") for rec in images.values() for a in rec.get("annotations",[])]
    for d in sorted({a for a in set(all_aids) if all_aids.count(a)>1}): errors.append(f"全局重复 ann_id: {d}")

    # 验收值检查
    total_instances=total_defect+total_normal
    if total_instances!=EXPECTED_TOTAL_INSTANCES: errors.append(f"总实例数 {total_instances} ≠ 期望 {EXPECTED_TOTAL_INSTANCES}")
    if total_defect!=EXPECTED_DEFECT: errors.append(f"缺陷实例 {total_defect} ≠ 期望 {EXPECTED_DEFECT}")
    if total_normal!=EXPECTED_NORMAL: errors.append(f"pseudo-normal {total_normal} ≠ 期望 {EXPECTED_NORMAL}")
    for c,ev in EXPECTED_COUNTS.items():
        actual=cat_counts.get(c,0)
        if actual!=ev: errors.append(f"类别 {c} 数量 {actual} ≠ 期望 {ev}")

    stats={
        "total_images":len(images),"total_in_manifest":len(mm),
        "status_counts":dict(status_counts),
        "total_instances":total_instances,"total_defect":total_defect,"total_normal":total_normal,
        "rect_count":rect_count,"poly_count":poly_count,
        "category_counts":dict(cat_counts),
        "error_count":len(errors),"warning_count":len(warnings),
    }
    return errors,warnings,stats


def write_report(path:Path, errors:List[str], warnings:List[str], stats:Dict[str,Any],
                 instances_csv_path:Path, summary_csv_path:Path):
    sc=stats["status_counts"]; cc=stats["category_counts"]
    finished=sc.get("done",0)+sc.get("skipped",0)
    lines=[
        "="*60,"  AOI Defect Annotation Validation Report","="*60,"",
        "--- 图片统计 ---",
        f"  总图片数 (manifest):          {stats['total_in_manifest']}",
        f"  总图片数 (annotations):       {stats['total_images']}",
        f"  done 图片数:                   {sc.get('done',0)}",
        f"  skipped 图片数:                {sc.get('skipped',0)}",
        f"  uncertain 图片数:              {sc.get('uncertain',0)}",
        f"  in_progress 图片数:            {sc.get('in_progress',0)}",
        f"  未完成图片数:                  {stats['total_images']-finished}",
        "",
        "--- 实例统计 ---",
        f"  总标注实例数:                  {stats['total_instances']}",
        f"  总缺陷实例数:                  {stats['total_defect']}",
        f"  pseudo-normal 实例数:           {stats['total_normal']}",
        f"  rectangle 数量:                 {stats['rect_count']}",
        f"  polygon 数量:                   {stats['poly_count']}",
        "",
        "--- 每个缺陷类别数量 ---",
    ]
    for c in sorted(VALID_DEFECT): lines.append(f"  {c}: {cc.get(c,0)}")
    lines.append("--- 每个正常区域类别数量 ---")
    for c in sorted(VALID_NORMAL): lines.append(f"  {c}: {cc.get(c,0)}")
    lines.append("")

    # 导出一致性检查
    ilines:List[str]=[]
    ic=read_csv_if_exists(instances_csv_path)
    sc2=read_csv_if_exists(summary_csv_path)
    if ic is not None:
        ilines.append("--- 导出一致性检查 ---")
        ilines.append(f"  instances.csv 行数: {len(ic)}  (期望 {stats['total_instances']})")
        if len(ic)!=stats['total_instances']: ilines.append("  ✗ 不一致!")
        else: ilines.append("  ✓ 一致")
        false_count=sum(1 for r in ic if r.get("is_pseudo_normal","False")=="False")
        true_count=sum(1 for r in ic if r.get("is_pseudo_normal","True")=="True")
        ilines.append(f"  is_pseudo_normal=False: {false_count} (期望 {stats['total_defect']})")
        ilines.append(f"  is_pseudo_normal=True:  {true_count} (期望 {stats['total_normal']})")
    if sc2 is not None:
        ilines.append(f"  image_summary.csv 行数: {len(sc2)} (期望 200)")
        if len(sc2)!=200: ilines.append("  ✗ 行数不一致!")
        # sum all counts
        sum_defect=sum(int(r.get(f"count_{c}",0)) for r in sc2 for c in VALID_DEFECT)
        sum_normal=sum(int(r.get(f"count_{c}",0)) for r in sc2 for c in VALID_NORMAL)
        ilines.append(f"  sum(count_A..G)={sum_defect} (期望 {stats['total_defect']})")
        ilines.append(f"  sum(count_N1..N3)={sum_normal} (期望 {stats['total_normal']})")
        # per-image check
        per_img_errors=0
        for r in sc2:
            iid=r["image_id"]
            nd=int(r.get("defect_instance_count","0"))
            nn=int(r.get("pseudo_normal_instance_count","0"))
            nt=int(r.get("total_annotation_count","0"))
            sdc=sum(int(r.get(f"count_{c}","0")) for c in VALID_DEFECT)
            snc=sum(int(r.get(f"count_{c}","0")) for c in VALID_NORMAL)
            if nd+nn!=nt: per_img_errors+=1
            if nd!=sdc: per_img_errors+=1
            if nn!=snc: per_img_errors+=1
        ilines.append(f"  逐图一致性错误: {per_img_errors}")
    if ilines:
        lines.extend(ilines)
        lines.append("")

    lines.append("="*60)
    lines.append(f"  校验结果: {stats['error_count']} errors, {stats['warning_count']} warnings")
    lines.append("="*60); lines.append("")
    if errors:
        lines.append(f"--- Errors ({len(errors)}) ---")
        for e in errors: lines.append(f"  ERROR: {e}")
        lines.append("")
    if warnings:
        lines.append(f"--- Warnings ({len(warnings)}) ---")
        for w in warnings: lines.append(f"  WARN:  {w}")
        lines.append("")
    if not errors and not warnings: lines.append("  ✓ 所有校验通过。\n")
    lines.append("="*60)
    text="\n".join(lines)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding="utf-8")
    logger.info("报告已保存: %s", path); print(text)


def main():
    p=argparse.ArgumentParser(description="校验 annotations.json")
    p.add_argument("--annotations_json",type=str,default="data/annotations/annotations.json")
    p.add_argument("--manifest_csv",type=str,default="data/metadata/images_manifest.csv")
    p.add_argument("--schema_yaml",type=str,default="configs/label_schema.yaml")
    p.add_argument("--report",type=str,default="reports/annotation_validation_report.txt")
    p.add_argument("--instances_csv",type=str,default="data/annotations/instances.csv")
    p.add_argument("--summary_csv",type=str,default="data/annotations/image_summary.csv")
    args=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    aj=(root/args.annotations_json).resolve(); mc=(root/args.manifest_csv).resolve()
    sy=(root/args.schema_yaml).resolve(); rp=(root/args.report).resolve()
    icp=(root/args.instances_csv).resolve(); scp=(root/args.summary_csv).resolve()
    if not aj.exists(): logger.error("annotations.json 不存在"); sys.exit(1)
    data=load_json(aj); mm=load_manifest(mc); lm=build_label_map(load_schema(sy))
    errors,warnings,stats=validate_all(data,mm,lm)
    write_report(rp,errors,warnings,stats,icp,scp)
    if errors: logger.error("✗ %d errors, %d warnings",len(errors),len(warnings)); sys.exit(1)
    elif warnings: logger.warning("✓ 0 errors, %d warnings",len(warnings)); sys.exit(0)
    else: logger.info("✓ 所有校验通过！"); sys.exit(0)

if __name__=="__main__": main()
