"""web_server.py — AOI 缺陷标注 FastAPI 后端服务。

提供:
  - Jinja2 模板渲染 (GET /)
  - 类别配置 API
  - 图片列表 / 起始图片 / 图片文件 API
  - 标注实例 CRUD API
  - 图片状态 / 备注 API
  - 进度统计 API
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# 将 src/ 加入 sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aoi_defect.annotation_store import AnnotationStore

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_server")


# ---------------------------------------------------------------------------
# 全局状态（由 create_app 设置）
# ---------------------------------------------------------------------------
_manifest: List[Dict[str, str]] = []
_schema: Dict[str, Any] = {}
_store: Optional[AnnotationStore] = None
_image_dir: Path = Path(".")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _natural_sort_key(s: str) -> Tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s))


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"images_manifest.csv 不存在: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"label_schema.yaml 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_image_size(image_id: str) -> Tuple[int, int]:
    """从 manifest 获取图片宽高。"""
    for row in _manifest:
        if row.get("image_id") == image_id:
            return int(row.get("width", 0)), int(row.get("height", 0))
    return 0, 0


def _get_filename(image_id: str) -> str:
    """从 manifest 获取文件名。"""
    for row in _manifest:
        if row.get("image_id") == image_id:
            return row.get("filename", "")
    return ""


# ---------------------------------------------------------------------------
# 创建 app（工厂函数）
# ---------------------------------------------------------------------------
def create_app(
    image_dir: str,
    manifest_csv: str,
    annotations_json: str,
    schema_yaml: str,
    annotator_html: str = "web/templates/annotator.html",
) -> FastAPI:
    """创建并配置 FastAPI 应用。

    Parameters
    ----------
    image_dir : str
        图片目录（相对于 project_root）。
    manifest_csv : str
        images_manifest.csv 路径。
    annotations_json : str
        annotations.json 路径。
    schema_yaml : str
        label_schema.yaml 路径。
    annotator_html : str
        前端模板 HTML 路径。
    """
    global _manifest, _schema, _store, _image_dir

    project_root = Path(__file__).resolve().parent.parent.parent
    _image_dir = (project_root / image_dir).resolve()

    manifest_path = (project_root / manifest_csv).resolve()
    schema_path = (project_root / schema_yaml).resolve()

    # 加载数据
    _manifest = _load_manifest(manifest_path)
    _schema = _load_schema(schema_path)
    _store = AnnotationStore(
        annotations_path=annotations_json,
        backup_dir="data/annotations/backups",
        project_root=project_root,
    )

    logger.info("已加载 manifest (%d 张图片), schema, annotations", len(_manifest))

    # 初始化 AnnotationStore 中的图片记录（按 manifest）
    for row in _manifest:
        iid = row.get("image_id", "")
        fn = row.get("filename", "")
        if iid:
            _store.get_image_record(iid, fn)

    # ------- FastAPI app -------
    app = FastAPI(title="AOI Defect Labeling Tool", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 模板 (纯 HTML，无 Jinja2 变量)
    _template_html_path = project_root / annotator_html

    # 静态文件
    static_dir = project_root / "web" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ---------- 页面 ----------
    @app.get("/")
    def index():
        if _template_html_path.exists():
            return HTMLResponse(content=_template_html_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>annotator.html 未找到</h1>", status_code=404)

    # ---------- 配置 ----------
    @app.get("/api/config")
    def api_config():
        """返回标签 schema。"""
        return {
            "defect_classes": _schema.get("defect_classes", {}),
            "normal_classes": _schema.get("normal_classes", {}),
            "regions": _schema.get("regions", []),
            "severity": _schema.get("severity", {}),
            "quality": _schema.get("quality", []),
            "geometry_types": _schema.get("geometry_types", []),
            "image_status": _schema.get("image_status", []),
        }

    # ---------- 图片列表 ----------
    @app.get("/api/images")
    def api_images():
        """返回 manifest 图片列表（含标注状态和 index）。"""
        images = _store.data.get("images", {})
        result = []
        for idx, row in enumerate(_manifest):
            iid = row.get("image_id", "")
            rec = images.get(iid, {})
            result.append({
                "image_id": iid,
                "filename": row.get("filename", ""),
                "width": int(row.get("width", 0)),
                "height": int(row.get("height", 0)),
                "status": rec.get("status", "unstarted"),
                "index": idx + 1,
            })
        return result

    # ---------- 起始图片 ----------
    @app.get("/api/start")
    def api_start(start_image_id: Optional[str] = Query(None)):
        """返回应打开的 image_id。

        1. 如果传了 start_image_id 且在 manifest 中，返回它。
        2. 否则返回第一个未完成的（status != "done"）。
        3. 全部完成返回第一个，all_done=True。
        """
        valid_ids = [r["image_id"] for r in _manifest]
        images = _store.data.get("images", {})

        if start_image_id and start_image_id in valid_ids:
            return {"image_id": start_image_id, "all_done": False}

        # 找第一个未完成
        for iid in valid_ids:
            rec = images.get(iid)
            if rec is None or rec.get("status") != "done":
                return {"image_id": iid, "all_done": False}

        # 全部完成
        return {"image_id": valid_ids[0] if valid_ids else "", "all_done": True}

    # ---------- 图片文件 ----------
    @app.get("/api/image/{image_id}")
    def api_image_file(image_id: str):
        """返回图片文件。仅允许 manifest 中存在的 image_id。"""
        filename = _get_filename(image_id)
        if not filename:
            raise HTTPException(status_code=404, detail=f"image_id 不在 manifest 中: {image_id}")

        image_path = _image_dir / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"图片文件不存在: {filename}")

        return FileResponse(image_path, media_type="image/jpeg")

    # ---------- 标注 CRUD ----------
    @app.get("/api/annotations/{image_id}")
    def api_get_annotations(image_id: str):
        """返回某图片的标注记录。"""
        if not _get_filename(image_id):
            raise HTTPException(status_code=404, detail=f"image_id 不在 manifest 中: {image_id}")
        anns = _store.list_annotations(image_id)
        return {"image_id": image_id, "annotations": anns}

    @app.post("/api/annotations/{image_id}")
    def api_add_annotation(image_id: str, body: Dict[str, Any]):
        """新增一条标注。

        Request JSON:
          label_code, geometry_type, points 或 bbox,
          region, severity, quality, note
        """
        filename = _get_filename(image_id)
        if not filename:
            raise HTTPException(status_code=404, detail=f"image_id 不在 manifest 中: {image_id}")

        img_w, img_h = _get_image_size(image_id)

        # ---- 坐标合法性检查 ----
        geometry_type = body.get("geometry_type", "rectangle")

        if geometry_type == "rectangle":
            bbox = body.get("bbox")
            if not bbox or len(bbox) != 4:
                raise HTTPException(status_code=400, detail="rectangle 需要 bbox [x_min, y_min, x_max, y_max]")
            try:
                bbox = [int(round(float(v))) for v in bbox]
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="bbox 坐标必须为数字")
            x_min, y_min, x_max, y_max = bbox
            if x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h:
                raise HTTPException(
                    status_code=400,
                    detail=f"bbox 超出图片范围: bbox={bbox}, image={img_w}x{img_h}",
                )
            if x_min >= x_max or y_min >= y_max:
                raise HTTPException(status_code=400, detail="x_min >= x_max 或 y_min >= y_max")

        elif geometry_type == "polygon":
            points = body.get("points")
            if not points or len(points) < 3:
                raise HTTPException(status_code=400, detail="polygon 需要至少 3 个顶点")
            # 检查每个点在图片范围内
            for p in points:
                if len(p) != 2:
                    raise HTTPException(status_code=400, detail="每个顶点需要 [x, y]")
                try:
                    px, py = int(round(float(p[0]))), int(round(float(p[1])))
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail="顶点坐标必须为数字")
                if px < 0 or px > img_w or py < 0 or py > img_h:
                    raise HTTPException(
                        status_code=400,
                        detail=f"顶点超出图片范围: [{px},{py}], image={img_w}x{img_h}",
                    )

        # 构建 annotation_dict
        ann_dict: Dict[str, Any] = {"label_code": body.get("label_code", "")}
        if "geometry_type" in body:
            ann_dict["geometry_type"] = body["geometry_type"]
        if "bbox" in body:
            ann_dict["bbox"] = body["bbox"]
        if "points" in body:
            ann_dict["points"] = body["points"]
        if "region" in body:
            ann_dict["region"] = body["region"]
        if "severity" in body:
            ann_dict["severity"] = body["severity"]
        if "quality" in body:
            ann_dict["quality"] = body["quality"]
        if "note" in body:
            ann_dict["note"] = body["note"]

        ann_id = _store.add_annotation(image_id, filename, ann_dict)
        if ann_id is None:
            raise HTTPException(status_code=400, detail="标注添加失败，请检查 label_code 等参数")
        _store.save()
        # 返回完整 annotation 对象
        anns = _store.list_annotations(image_id)
        new_ann = next((a for a in anns if a.get("ann_id") == ann_id), None)
        return {"status": "ok", "ann_id": ann_id, "annotation": new_ann}

    @app.delete("/api/annotations/{image_id}/{ann_id}")
    def api_delete_annotation(image_id: str, ann_id: str):
        """删除一条标注。"""
        if not _get_filename(image_id):
            raise HTTPException(status_code=404, detail=f"image_id 不存在: {image_id}")
        ok = _store.delete_annotation(image_id, ann_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"ann_id 不存在: {ann_id}")
        _store.save()
        return {"status": "ok"}

    # ---------- 图片状态 & 备注 ----------
    @app.post("/api/image_status/{image_id}")
    def api_update_status(image_id: str, body: Dict[str, Any]):
        """更新图片状态。"""
        if not _get_filename(image_id):
            raise HTTPException(status_code=404, detail=f"image_id 不存在: {image_id}")
        status = body.get("status", "")
        ok = _store.update_image_status(image_id, status)
        if not ok:
            raise HTTPException(status_code=400, detail=f"无效 status: '{status}'")
        _store.save()
        return {"status": "ok"}

    @app.post("/api/image_note/{image_id}")
    def api_update_note(image_id: str, body: Dict[str, Any]):
        """更新图片备注。"""
        if not _get_filename(image_id):
            raise HTTPException(status_code=404, detail=f"image_id 不存在: {image_id}")
        note = body.get("note", "")
        _store.update_image_note(image_id, note)
        _store.save()
        return {"status": "ok"}

    # ---------- 进度 ----------
    @app.get("/api/progress")
    def api_progress():
        """返回标注进度统计。"""
        images = _store.data.get("images", {})
        total = len(_manifest)
        status_counts: Dict[str, int] = {
            "done": 0, "skipped": 0, "uncertain": 0,
            "in_progress": 0, "unstarted": 0,
        }
        for iid in [r["image_id"] for r in _manifest]:
            rec = images.get(iid, {})
            st = rec.get("status", "unstarted")
            if st in status_counts:
                status_counts[st] += 1

        return {
            "total_images": total,
            **status_counts,
        }

    return app
