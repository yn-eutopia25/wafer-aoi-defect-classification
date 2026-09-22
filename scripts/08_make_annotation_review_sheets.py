#!/usr/bin/env python3
"""
08_make_annotation_review_sheets.py — 生成标注复核 overlay 图与 contact sheet

输出:
  - reports/annotation_review/overlay_images/   — 单张 overlay 图
  - reports/annotation_review/contact_sheets/   — 4x4 contact sheet
  - reports/annotation_review/index.html        — 浏览页面

用法:
    python scripts/08_make_annotation_review_sheets.py
    python scripts/08_make_annotation_review_sheets.py --only_done
    python scripts/08_make_annotation_review_sheets.py --label_style short
"""

from __future__ import annotations

import argparse, csv, json, logging, math, re, shutil, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("review")

CAT_COLORS: Dict[str, Tuple[int,int,int,int]] = {
    "A":(255,107,107,220),"B":(255,169,77,220),"C":(255,212,59,220),"D":(105,219,124,220),
    "E":(116,192,252,220),"F":(218,119,242,220),"G":(173,181,189,220),
    "N1":(56,217,169,220),"N2":(77,171,247,220),"N3":(177,151,252,220),
}
_FALLBACK = (200,200,200,220)
GRID_COLS, GRID_ROWS, PER_PAGE = 4, 4, 16


def load_ann(p: Path) -> Dict: return json.loads(p.read_text(encoding="utf-8"))
def load_manifest(p: Path) -> Dict[str,Dict[str,str]]:
    if not p.exists(): return {}
    with open(p,"r",encoding="utf-8-sig",newline="") as f: return {r["image_id"]:r for r in csv.DictReader(f)}
def _nk(s:str)->Tuple: return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)",s))
def _font(sz:int): 
    try: return ImageFont.truetype("arial.ttf",size=sz)
    except: return ImageFont.load_default()


def _draw_single(image_path:Path, image_id:str, image_status:str,
                 annotations:List[Dict], output_path:Path, label_style:str):
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    font = _font(10); font_bold = _font(11)

    # 按 area_px 从大到小排序
    anns_sorted = sorted(annotations, key=lambda a: float(a.get("area_px",0) or 0), reverse=True)

    defect_anns = [a for a in annotations if not a.get("is_pseudo_normal")]
    normal_anns = [a for a in annotations if a.get("is_pseudo_normal")]
    nd, nn = len(defect_anns), len(normal_anns)
    present = sorted({a.get("label_code","") for a in defect_anns if a.get("label_code")})

    for ann in anns_sorted:
        lc = ann.get("label_code","?"); color = CAT_COLORS.get(lc, _FALLBACK)
        gt = ann.get("geometry_type","rectangle")

        # 标签文字
        if label_style == "code": lbl = lc
        elif label_style == "short": lbl = f"{lc} {ann.get('label','')}"
        else: lbl = f"{lc} {ann.get('label','')}"

        if gt == "polygon":
            pts_raw = ann.get("points",[])
            if len(pts_raw) >= 3:
                pts = [(p[0],p[1]) for p in pts_raw]
                draw.polygon(pts, fill=(*color[:3], 30))
                draw.polygon(pts, outline=color[:3], width=2)
                for p in pts: draw.ellipse([p[0]-2,p[1]-2,p[0]+2,p[1]+2], fill=color[:3])
                lx, ly = pts[0][0], pts[0][1]-14
            else: lx,ly = 0,0
        else:
            bbox = ann.get("bbox",[])
            if len(bbox)==4:
                x1,y1,x2,y2 = bbox
                draw.rectangle([x1,y1,x2,y2], outline=color[:3], width=2)
                lx, ly = x1, y1-14
            else: lx,ly = 0,0

        # 画标签
        try: tb = draw.textbbox((lx,ly), lbl, font=font)
        except: tb = None
        if tb:
            draw.rectangle([tb[0]-2,tb[1]-1,tb[2]+2,tb[3]+1], fill=color[:3])
            draw.text((lx,ly), lbl, fill=(0,0,0,255), font=font)

    # 信息面板（画在图像上方额外区域或左上角）
    info_w = max(200, draw.textbbox((0,0), f"classes: {','.join(present)}", font=font)[2]+20)
    info_h = 5*18+10
    draw.rectangle([0,0,info_w,info_h], fill=(0,0,0,200))
    lines = [
        image_id, f"status: {image_status}",
        f"defects: {nd}  normal: {nn}",
        f"classes: {','.join(present) if present else '-'}",
    ]
    for i,line in enumerate(lines): draw.text((6,4+i*18), line, fill=(255,255,255,255), font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "JPEG", quality=92)


def _build_contact_sheet(overlay_paths: List[Path], output_path: Path, cell_size: Tuple[int,int]=(400,400)):
    cw,ch=cell_size
    canvas=Image.new("RGB",(GRID_COLS*cw, GRID_ROWS*ch), color=(40,40,50))
    for idx,op in enumerate(overlay_paths):
        if idx>=PER_PAGE: break
        try:
            thumb=Image.open(op).convert("RGB"); thumb.thumbnail((cw,ch), Image.LANCZOS)
        except: thumb=Image.new("RGB",(cw,ch),(60,60,70))
        col,row=idx%GRID_COLS,idx//GRID_COLS
        tx=col*cw+(cw-thumb.width)//2; ty=row*ch+(ch-thumb.height)//2
        canvas.paste(thumb,(tx,ty))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output_path,"JPEG",quality=92)


def _generate_index(overlay_rel:List[str], contact_rel:List[str], output_path:Path):
    parts=[
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='UTF-8'><title>AOI Annotation Review</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#1e1e2e;color:#e0e0e8;margin:0;padding:20px}"
        "h1{color:#7c6ff0}h2{color:#8888a0;margin-top:30px}"
        ".legend{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}"
        ".legend-item{display:flex;align-items:center;gap:4px;font-size:12px}"
        ".legend-dot{width:12px;height:12px;border-radius:3px}"
        ".gallery{display:flex;flex-wrap:wrap;gap:10px}"
        ".gallery img{width:400px;border:1px solid #3e3e56;border-radius:4px;cursor:pointer;transition:transform .15s}"
        ".gallery img:hover{transform:scale(1.02)}</style></head><body>"
        "<h1>AOI Annotation Review Sheets</h1>"
    ]
    # Legend
    parts.append("<div class='legend'>")
    color_map={"A":"#ff6b6b","B":"#ffa94d","C":"#ffd43b","D":"#69db7c","E":"#74c0fc","F":"#da77f2","G":"#adb5bd",
               "N1":"#38d9a9","N2":"#4dabf7","N3":"#b197fc"}
    for c in ["A","B","C","D","E","F","G","N1","N2","N3"]:
        parts.append(f"<div class='legend-item'><span class='legend-dot' style='background:{color_map[c]}'></span>{c}</div>")
    parts.append("</div>")

    parts.append(f"<p>Overlay 图片: {len(overlay_rel)} 张 | Contact Sheets: {len(contact_rel)} 页</p>")
    if contact_rel:
        parts.append("<h2>Contact Sheets</h2><div class='gallery'>")
        for p in contact_rel: parts.append(f'<div><img src="{p}" loading="lazy"><p>{Path(p).name}</p></div>')
        parts.append("</div>")
    if overlay_rel:
        parts.append("<h2>Overlay Images</h2><div class='gallery'>")
        for p in overlay_rel: parts.append(f'<img src="{p}" loading="lazy" title="{Path(p).name}">')
        parts.append("</div>")
    parts.append("</body></html>")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text("\n".join(parts),encoding="utf-8")
    logger.info("index.html → %s", output_path)


def main():
    p = argparse.ArgumentParser(description="生成标注复核 overlay 图")
    p.add_argument("--annotations_json",type=str,default="data/annotations/annotations.json")
    p.add_argument("--manifest_csv",type=str,default="data/metadata/images_manifest.csv")
    p.add_argument("--images_dir",type=str,default="data/images")
    p.add_argument("--output_dir",type=str,default="reports/annotation_review")
    p.add_argument("--only_done",action="store_true",default=False)
    p.add_argument("--include_unstarted",action="store_true",default=False)
    p.add_argument("--cell_size",type=int,default=400)
    p.add_argument("--label_style",type=str,default="code",choices=["code","short","full"])
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    aj = (root/args.annotations_json).resolve(); mc = (root/args.manifest_csv).resolve()
    img_dir = (root/args.images_dir).resolve(); out_dir = (root/args.output_dir).resolve()
    if not aj.exists(): logger.error("annotations.json 不存在"); sys.exit(1)

    data = load_ann(aj); mm = load_manifest(mc)
    images = data.get("images",{})

    # 清理旧输出
    for d in [out_dir/"overlay_images", out_dir/"contact_sheets"]:
        if d.exists(): shutil.rmtree(d)
    idx_path = out_dir/"index.html"
    if idx_path.exists(): idx_path.unlink()

    overlay_dir = out_dir/"overlay_images"; contact_dir = out_dir/"contact_sheets"
    overlay_dir.mkdir(parents=True,exist_ok=True); contact_dir.mkdir(parents=True,exist_ok=True)

    overlay_paths: List[Path] = []; done_count = 0; skipped = 0
    for iid in sorted(images.keys(), key=_nk):
        rec = images[iid]; st = rec.get("status","unstarted")
        if args.only_done and st!="done": skipped+=1; continue
        if not args.include_unstarted and st=="unstarted": skipped+=1; continue
        anns = rec.get("annotations",[])
        fn = rec.get("filename",""); ip = img_dir/fn
        if not ip.exists(): logger.warning("图片不存在: %s", ip); skipped+=1; continue
        if st=="done": done_count+=1
        op = overlay_dir/f"{iid}_overlay.jpg"
        try:
            _draw_single(ip, iid, st, anns, op, args.label_style)
            overlay_paths.append(op)
        except Exception as e: logger.error("失败 %s: %s", iid, e)

    logger.info("Overlay: %d 张, 跳过 %d 张", len(overlay_paths), skipped)

    cs_paths: List[Path] = []
    for pg in range(math.ceil(len(overlay_paths)/PER_PAGE)):
        chunk = overlay_paths[pg*PER_PAGE:(pg+1)*PER_PAGE]
        csp = contact_dir/f"contact_sheet_{pg+1:03d}.jpg"
        _build_contact_sheet(chunk, csp, (args.cell_size,args.cell_size))
        cs_paths.append(csp)
    logger.info("Contact sheet: %d 页", len(cs_paths))

    _generate_index(
        [f"overlay_images/{p.name}" for p in overlay_paths],
        [f"contact_sheets/{p.name}" for p in cs_paths],
        out_dir/"index.html",
    )
    logger.info("完成: %d overlays, %d contact sheets, done=%d", len(overlay_paths), len(cs_paths), done_count)


if __name__=="__main__": main()
