#!/usr/bin/env python3
"""Local UI server: satellite basemap + Enhance for selected bbox.

  python3 scripts/zoom_app_server.py
  open http://127.0.0.1:8877/
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "zoom_strips" / "enhance_ui"
LATEST_PATH = OUT_DIR / "latest.json"
sys.path.insert(0, str(SCRIPTS))

from generate_zoom_strip import (  # noqa: E402
    approx_meters,
    build_edge_guided_prompt,
    download_mapbox,
    gemini_enhance,
    load_env,
    lock_enhance_to_mapbox,
    estimate_detail_energy,
    make_structure_edges,
    sharpen_mapbox,
    mapbox_pixel_size,
)
from PIL import Image  # noqa: E402


def _load() -> None:
    load_env(ROOT / ".env")


class Handler(BaseHTTPRequestHandler):
    server_version = "ZoomApp/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            html = (SCRIPTS / "bbox_selector.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")

        if path == "/api/health":
            return self._send_json(
                200,
                {
                    "ok": True,
                    "mapbox": bool(os.environ.get("MAPBOX_API_KEY")),
                    "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
                },
            )

        if path == "/api/latest":
            if not LATEST_PATH.exists():
                return self._send_json(404, {"error": "no saved enhancement yet"})
            try:
                data = json.loads(LATEST_PATH.read_text())
            except Exception as exc:
                return self._send_json(500, {"error": str(exc)})
            # Drop stale pointers if files were deleted
            enh = OUT_DIR / Path(data.get("enhanced_url", "")).name
            if not enh.exists():
                return self._send_json(404, {"error": "saved enhancement files missing"})
            return self._send_json(200, data)

        # Proxy Mapbox satellite tiles so the secret sk. token never hits the browser.
        if path.startswith("/tiles/mapbox/"):
            return self._proxy_mapbox_tile(path)

        if path.startswith("/outputs/"):
            rel = path[len("/outputs/") :]
            file_path = (OUT_DIR / rel).resolve()
            if not str(file_path).startswith(str(OUT_DIR.resolve())) or not file_path.exists():
                return self._send_json(404, {"error": "not found"})
            ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            return self._send(200, file_path.read_bytes(), ctype)

        if path.startswith("/strip/"):
            # /strip/<dirname>/<filename>
            rel = path[len("/strip/") :]
            file_path = (ROOT / "data" / "zoom_strips" / rel).resolve()
            base = (ROOT / "data" / "zoom_strips").resolve()
            if not str(file_path).startswith(str(base)) or not file_path.exists():
                return self._send_json(404, {"error": "not found"})
            ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            return self._send(200, file_path.read_bytes(), ctype)

        # static scripts fallback
        if path.startswith("/scripts/"):
            file_path = (ROOT / path.lstrip("/")).resolve()
            if file_path.exists() and str(file_path).startswith(str(ROOT.resolve())):
                ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
                return self._send(200, file_path.read_bytes(), ctype)

        return self._send_json(404, {"error": "not found", "path": path})

    def _proxy_mapbox_tile(self, path: str) -> None:
        token = os.environ.get("MAPBOX_API_KEY")
        if not token:
            return self._send_json(500, {"error": "MAPBOX_API_KEY missing"})
        # /tiles/mapbox/{z}/{x}/{y}
        parts = path.strip("/").split("/")
        if len(parts) != 5:
            return self._send_json(400, {"error": "expected /tiles/mapbox/{z}/{x}/{y}"})
        _, _, z, x, y = parts
        y = y.split(".")[0]
        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}@2x"
            f"?access_token={token}"
        )
        try:
            resp = requests.get(url, timeout=30)
        except Exception as exc:
            return self._send_json(502, {"error": str(exc)})
        if resp.status_code >= 400:
            return self._send(resp.status_code, resp.content, "text/plain")
        ctype = resp.headers.get("Content-Type", "image/jpeg")
        self._send(200, resp.content, ctype)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/enhance":
            return self._enhance()
        if parsed.path == "/api/zoom-strip":
            return self._zoom_strip()
        if parsed.path == "/api/latest/clear":
            if LATEST_PATH.exists():
                LATEST_PATH.unlink()
            return self._send_json(200, {"ok": True})
        return self._send_json(404, {"error": "not found"})

    def _parse_bbox(self, data: dict):
        if "bbox" in data and isinstance(data["bbox"], list) and len(data["bbox"]) == 4:
            west, south, east, north = map(float, data["bbox"])
        else:
            west = float(data["west"])
            south = float(data["south"])
            east = float(data["east"])
            north = float(data["north"])
        if not (west < east and south < north):
            raise ValueError("invalid bbox")
        return west, south, east, north

    def _resolve_reference(self, reference_url: str) -> Path:
        """Map /outputs/<file> URLs to files under OUT_DIR."""
        parsed = urllib.parse.urlparse(reference_url)
        path = parsed.path
        if not path.startswith("/outputs/"):
            raise ValueError("reference_url must be an /outputs/... path")
        name = path[len("/outputs/") :]
        file_path = (OUT_DIR / name).resolve()
        if not str(file_path).startswith(str(OUT_DIR.resolve())) or not file_path.exists():
            raise ValueError("reference image not found")
        return file_path

    def _enhance(self) -> None:
        try:
            data = self._read_json()
            west, south, east, north = self._parse_bbox(data)
        except Exception as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        job = uuid.uuid4().hex[:10]
        raw_path = OUT_DIR / f"{job}_viewport.jpg"
        enh_path = OUT_DIR / f"{job}_enhanced.png"

        center = data.get("center") or {
            "lon": (west + east) / 2.0,
            "lat": (south + north) / 2.0,
        }
        detail_hf = None
        client_zoom = float(data.get("zoom") or 0.0)

        try:
            # Prefer live map canvas screenshot from the browser (exact pixels on screen).
            viewport_b64 = data.get("viewport_image") or data.get("screenshot")
            if not viewport_b64:
                return self._send_json(
                    400,
                    {"error": "viewport_image required (screenshot of current map canvas)"},
                )
            if "," in viewport_b64:
                viewport_b64 = viewport_b64.split(",", 1)[1]
            raw_path.write_bytes(base64.b64decode(viewport_b64))
            with Image.open(raw_path) as im:
                rgb = im.convert("RGB")
                iw, ih = rgb.size
                rgb.save(raw_path, format="JPEG", quality=92)

            detail_hf = estimate_detail_energy(raw_path)
            # Fine edges from the screenshot → Gemini(viewport + edges) → lock to screenshot.
            edges_path = OUT_DIR / f"{job}_edges.png"
            make_structure_edges(raw_path, edges_path, style="edges_fine")

            prompt = build_edge_guided_prompt(
                west=west,
                south=south,
                east=east,
                north=north,
                meters_hint=approx_meters(west, south, east, north),
                center=center,
                image_width=iw,
                image_height=ih,
                guide_style="edges_fine",
            )
            (OUT_DIR / f"{job}_prompt.txt").write_text(prompt)

            model = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
            gemini_raw = OUT_DIR / f"{job}_gemini_raw.png"
            try:
                gemini_enhance(
                    model=model,
                    prompt=prompt,
                    reference_paths=[raw_path, edges_path],
                    out_path=gemini_raw,
                    temperature=0.2,
                    top_p=0.85,
                    top_k=32,
                    match_size_to=raw_path,
                )
                locked, ncc, mode = lock_enhance_to_mapbox(
                    raw_path,
                    gemini_raw,
                    enh_path,
                    min_ncc=0.88,
                    creative_freedom="med",
                )
                mode = f"viewport+edges_fine→{mode}"
            except Exception as gemini_exc:
                sharpen_mapbox(raw_path, enh_path)
                ncc = 0.0
                mode = f"viewport_unsharp_after_error:{gemini_exc}"
        except Exception as exc:
            return self._send_json(500, {"error": str(exc)})

        result = {
            "ok": True,
            "bbox": [west, south, east, north],
            "center": center,
            "meters": approx_meters(west, south, east, north),
            "original_url": f"/outputs/{raw_path.name}",
            "edges_url": f"/outputs/{job}_edges.png",
            "enhanced_url": f"/outputs/{enh_path.name}",
            "prompt_url": f"/outputs/{job}_prompt.txt",
            "source": {
                "provider": "map_viewport_screenshot",
                "crs": "EPSG:3857",
                "orientation": "north-up",
                "inclination_deg": 0.0,
                "view": "browser canvas capture",
                "detail_hf": detail_hf,
                "client_zoom": client_zoom,
                "guide": "edges_fine",
                "creative_freedom": "med",
            },
            "align": {"ncc": ncc, "mode": mode},
            "chained": False,
            "saved_at": time.time(),
        }
        LATEST_PATH.write_text(json.dumps(result, indent=2))
        return self._send_json(200, result)

    def _zoom_strip(self) -> None:
        try:
            data = self._read_json()
            west, south, east, north = self._parse_bbox(data)
        except Exception as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

        from generate_zoom_strip import generate_for_bbox

        out_dir = ROOT / "data" / "zoom_strips" / f"ui_{int(time.time())}"
        model = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        try:
            strip = generate_for_bbox(west, south, east, north, out_dir=out_dir, model=model)
        except Exception as exc:
            return self._send_json(500, {"error": str(exc)})
        # expose via /outputs by copying names under OUT_DIR symlink-style: serve from absolute via new route
        return self._send_json(
            200,
            {
                "ok": True,
                "out_dir": str(out_dir),
                "strip_path": str(strip),
                "strip_url": f"/strip/{out_dir.name}/zoom_strip.jpg",
            },
        )


def main() -> int:
    _load()
    host = "127.0.0.1"
    port = int(os.environ.get("ZOOM_APP_PORT", "8877"))
    # free the port if previous python http.server holds it
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Zoom app: http://{host}:{port}/")
    print(f"  MAPBOX={'yes' if os.environ.get('MAPBOX_API_KEY') else 'NO'}")
    print(f"  GEMINI={'yes' if (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')) else 'NO'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
