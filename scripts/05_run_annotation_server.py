#!/usr/bin/env python3
"""
05_run_annotation_server.py — 启动 AOI 标注 Web 服务

用法:
    python scripts/05_run_annotation_server.py
    python scripts/05_run_annotation_server.py --port 8080
    python scripts/05_run_annotation_server.py --start_image_id AOI_NG_0050
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 将 src/ 加入 sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AOI 缺陷标注 Web 服务")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定 IP (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="端口 (默认 8000)")
    parser.add_argument("--image_dir", type=str, default="data/images",
                        help="图片目录 (默认 data/images)")
    parser.add_argument("--manifest_csv", type=str,
                        default="data/metadata/images_manifest.csv",
                        help="Manifest CSV 路径")
    parser.add_argument("--annotations_json", type=str,
                        default="data/annotations/annotations.json",
                        help="标注 JSON 路径")
    parser.add_argument("--schema_yaml", type=str,
                        default="configs/label_schema.yaml",
                        help="标签 schema YAML 路径")
    parser.add_argument("--start_image_id", type=str, default="",
                        help="从指定 image_id 开始 (默认: 自动找第一张未完成)")
    args = parser.parse_args()

    from aoi_defect.web_server import create_app

    app = create_app(
        image_dir=args.image_dir,
        manifest_csv=args.manifest_csv,
        annotations_json=args.annotations_json,
        schema_yaml=args.schema_yaml,
    )

    # 如果有 --start_image_id，写入临时文件供前端读取
    if args.start_image_id:
        project_root = Path(__file__).resolve().parent.parent
        start_file = project_root / "web" / "templates" / ".start_image_id"
        start_file.parent.mkdir(parents=True, exist_ok=True)
        start_file.write_text(args.start_image_id, encoding="utf-8")

    import uvicorn
    print(f"\n{'='*60}")
    print(f"  AOI Defect Labeling Tool")
    print(f"  访问地址: http://{args.host}:{args.port}")
    if args.start_image_id:
        print(f"  起始图片: {args.start_image_id}")
    print(f"  按 Ctrl+C 停止服务")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
