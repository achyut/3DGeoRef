#!/usr/bin/env python3
"""Generate a progressive satellite zoom strip for a WGS84 bounding box.

Typical flow:
  1. Open scripts/bbox_selector.html in a browser
  2. Draw a rectangle and download bbox.json
  3. python scripts/generate_zoom_strip.py --bbox bbox.json

Requires in .env (or environment):
  MAPBOX_API_KEY
  GEMINI_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / ".env"

# Linear scale relative to the selected bbox extent.
# Selected bbox == "close" (scale 1.0).
# (name, scale_vs_selected_bbox, detail_prompt, offset_xy)
# offset_xy shifts the window inside the selected bbox: (+x east, +y north)
# Keep ultra/extreme on the SAME focus so each red box nests inside the previous panel.
FOCUS_PARKING = (0.0, 0.30)  # north parking lot inside the sample site
LEVELS = [
    ("far", 8.0, "city-scale shapes: urban fabric, major roads, rivers/valleys", (0.0, 0.0)),
    ("mid", 3.0, "neighborhood blocks, campus/complex silhouette, street grid", (0.0, 0.0)),
    ("close", 1.0, "building footprints, courtyards, parking lots, tree crowns", (0.0, 0.0)),
    ("ultra", 0.20, "street borders, lane marks, utility poles/wires, vehicles, people", FOCUS_PARKING),
    ("extreme", 0.04, "fill frame with parked cars and pavement; car body style/color, people, curb stones, cracks, markings, wires", FOCUS_PARKING),
]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_bbox(path: Path) -> Tuple[float, float, float, float]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "bbox" in data:
        west, south, east, north = data["bbox"]
    elif isinstance(data, dict):
        west = data["west"]
        south = data["south"]
        east = data["east"]
        north = data["north"]
    elif isinstance(data, (list, tuple)) and len(data) == 4:
        west, south, east, north = data
    else:
        raise ValueError("bbox JSON must be [west,south,east,north] or an object with those fields")
    west, south, east, north = map(float, (west, south, east, north))
    if not (west < east and south < north):
        raise ValueError("Invalid bbox ordering; need west < east and south < north")
    return west, south, east, north


def scaled_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    scale: float,
    *,
    # Normalized offset from center in selected-bbox units: (+x east, +y north)
    offset_xy: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float, float, float]:
    cx = (west + east) / 2.0 + offset_xy[0] * (east - west)
    cy = (south + north) / 2.0 + offset_xy[1] * (north - south)
    half_w = (east - west) / 2.0 * scale
    half_h = (north - south) / 2.0 * scale
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def mercator_aspect(west: float, south: float, east: float, north: float) -> float:
    """Width/height ratio of bbox in Web Mercator (EPSG:3857)."""
    import math

    def mx(lon: float) -> float:
        return lon * 20037508.342789244 / 180.0

    def my(lat: float) -> float:
        lat = max(min(lat, 85.05112878), -85.05112878)
        return math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * 6378137.0

    w = abs(mx(east) - mx(west))
    h = abs(my(north) - my(south))
    if h <= 0:
        return 1.0
    return w / h


def mapbox_pixel_size(
    west: float,
    south: float,
    east: float,
    north: float,
    max_dim: int = 1280,
) -> Tuple[int, int]:
    """Choose Static API width/height matching mercator aspect (avoids letterboxing)."""
    aspect = mercator_aspect(west, south, east, north)
    max_dim = max(64, min(int(max_dim), 1280))
    if aspect >= 1.0:
        width = max_dim
        height = max(64, int(round(max_dim / aspect)))
    else:
        height = max_dim
        width = max(64, int(round(max_dim * aspect)))
    return width, height


def download_mapbox(
    west: float,
    south: float,
    east: float,
    north: float,
    out_path: Path,
    size: int = 1280,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Path:
    """Download Mapbox satellite static image for bbox.

    Uses Web Mercator aspect ratio by default so the image fills the bbox
    edge-to-edge (required for Leaflet overlay registration / opacity compare).
    """
    token = os.environ.get("MAPBOX_API_KEY")
    if not token:
        raise RuntimeError("MAPBOX_API_KEY is not set")
    if width is None or height is None:
        width, height = mapbox_pixel_size(west, south, east, north, max_dim=size)
    width = max(1, min(int(width), 1280))
    height = max(1, min(int(height), 1280))
    url = (
        "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"[{west},{south},{east},{north}]/{width}x{height}@2x"
        f"?access_token={token}&attribution=false&logo=false"
    )
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def gemini_enhance(
    *,
    model: str,
    prompt: str,
    reference_paths: List[Path],
    out_path: Path,
    temperature: float = 0.0,
    match_size_to: Optional[Path] = None,
) -> Path:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    parts = [{"text": prompt}]
    for ref in reference_paths:
        mime = "image/jpeg" if ref.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime,
                    "data": base64.b64encode(ref.read_bytes()).decode("ascii"),
                }
            }
        )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": temperature,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = requests.post(
        url,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    for cand in data.get("candidates") or []:
        for part in ((cand.get("content") or {}).get("parts")) or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline:
                continue
            raw = base64.b64decode(inline["data"])
            out_path.write_bytes(raw)
            # Force same pixel grid as structural reference (survey overlay alignment).
            if match_size_to and match_size_to.exists():
                src = Image.open(match_size_to).convert("RGB")
                out = Image.open(out_path).convert("RGB")
                if out.size != src.size:
                    out = out.resize(src.size, Image.Resampling.LANCZOS)
                    out.save(out_path)
            return out_path
    raise RuntimeError("Gemini returned no image bytes")


def build_survey_enhance_prompt(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    meters_hint: str,
    center: Optional[dict] = None,
    image_width: int,
    image_height: int,
    chained: bool = False,
) -> str:
    """Prompt for survey-safe sharpening: geometry locked to viewport screenshot."""
    cx = float((center or {}).get("lon", (west + east) / 2.0))
    cy = float((center or {}).get("lat", (south + north) / 2.0))
    chain = ""
    if chained:
        chain = (
            "\nIf a second (prior enhanced) image is attached, use it ONLY for texture/material continuity. "
            "ALL geometry, rotation, and framing MUST follow Image 1 (viewport screenshot) exactly.\n"
        )
    return f"""
TASK: Survey-grade IMAGE ENHANCEMENT of an existing orthophoto (NOT creative generation / NOT a new acquisition).

GOALS:
1) Sharpen & clarify what is ALREADY in Image 1 — the live map viewport screenshot (dehaze, denoise, deblur).
2) Add REALISTIC MICRO-DETAIL only on surfaces/objects that already exist (leaves on existing canopy,
   asphalt grit on existing pavement, roofing texture on existing roofs, paint/glass edges on existing cars).
3) NEVER invent new objects or infrastructure. Prefer under-enhancement over hallucination.

ALLOWED DETAIL (only where the object/surface is already visible in Image 1):
- Existing trees/vegetation: leaf-mass texture, canopy edge crispness, internal canopy variation — do not grow new trees or place plants in empty road/yard.
- Existing roads/pavement/dirt: fine surface grain / wear on that same surface — do not invent sidewalks, curbs, stall grids, or lane paint that are not clearly there.
- Existing roofs/buildings: tile/metal/ridge texture and sharper outlines — same footprint.
- Existing cars: sharper body/window/shadow boundaries — same count and positions.

FORBIDDEN (survey failures):
- New buildings, roofs, roads, sidewalks, parking stalls, lane markings, fences, people, or vehicles.
- Turning ambiguous blur blobs into trees/bushes/objects (especially in the middle of streets/lots).
- Moving, removing, or redesigning layout. No rotate / warp / reframe.
- If unsure a feature exists → leave it soft; do not invent.

SOURCE IMAGE METADATA (must preserve):
- Source: browser map canvas screenshot (same pixels the user sees)
- Product type: web-mercator orthorectified raster tile mosaic (not a raw tilted pushbroom frame)
- CRS / orientation: EPSG:3857 display, north-up, east-right (NO rotation, NO flip, NO skew)
- View geometry: nadir / ortho (sensor inclination ≈ 0° relative to vertical). Do not invent oblique perspective.
- Coverage WGS84 bbox [west,south,east,north]: [{west:.8f}, {south:.8f}, {east:.8f}, {north:.8f}]
- Center lon/lat: {cx:.8f}, {cy:.8f}
- Ground coverage: {meters_hint}
- Input pixel size: {image_width} x {image_height} px — keep the SAME aspect ratio and same scene framing
{chain}
Return a single true-color orthophoto: sharper with richer detail on EXISTING content, zero new objects,
overlay-ready on Image 1 with near-zero misregistration.
""".strip()


def build_prompt(
    *,
    level: str,
    detail: str,
    west: float,
    south: float,
    east: float,
    north: float,
    meters_hint: str,
) -> str:
    """Legacy zoom-strip prompt (less strict than survey enhance)."""
    return build_survey_enhance_prompt(
        west=west,
        south=south,
        east=east,
        north=north,
        meters_hint=meters_hint,
        image_width=1280,
        image_height=1280,
        chained=False,
    ) + f"\n\nZoom level name: {level}\nAdditional focus: {detail}"


def structural_similarity_ncc(mapbox_path: Path, other_path: Path) -> float:
    """Normalized cross-correlation of downscaled grayscale (0..1-ish)."""
    import numpy as np

    a = Image.open(mapbox_path).convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    b = Image.open(other_path).convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    pa = np.asarray(a, dtype=np.float64)
    pb = np.asarray(b, dtype=np.float64)
    pa -= pa.mean()
    pb -= pb.mean()
    denom = float(np.sqrt((pa * pa).sum() * (pb * pb).sum()))
    if denom <= 1e-9:
        return 0.0
    return float((pa * pb).sum() / denom)


def estimate_detail_energy(image_path: Path) -> float:
    """Mean |image - blur| — drops sharply when Mapbox is past native resolution."""
    from PIL import ImageFilter
    import numpy as np

    g = Image.open(image_path).convert("L").resize((512, 512), Image.Resampling.BILINEAR)
    raw = np.asarray(g, dtype=np.float64)
    blur = np.asarray(g.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float64)
    return float(np.mean(np.abs(raw - blur)))


# Below this, Mapbox is mostly upsampled soft tiles (see z18→z22 probes in Kathmandu).
MAPBOX_MIN_DETAIL_FOR_GEMINI = 1.2


def make_structure_edges(
    mapbox_path: Path,
    out_path: Path,
    *,
    style: str = "edges_fine",
) -> Path:
    """Build a structure plate from Mapbox for Gemini guidance.

    Styles:
      - edges_fine (default): soft Sobel magnitude — thin detailed structure lines
      - edges: same pipeline, slightly more blur / stronger noise suppress
      - stencil: high-contrast stamp / photocopy look (bold B&W shapes)
      - bw: autocontrast grayscale (luminance plate)
    """
    from PIL import ImageEnhance, ImageFilter, ImageOps
    import numpy as np

    gray = Image.open(mapbox_path).convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=2)
    style = (style or "edges_fine").lower().strip()
    # Treat plain "edges" as the fine detailed guide (preferred in experiments).
    if style == "edges":
        style = "edges_fine"

    if style == "bw":
        rgb = Image.merge("RGB", (gray, gray, gray))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(out_path)
        return out_path

    if style in {"edges_fine", "edges_soft"}:
        # Fine structural edges (not noisy binary lace):
        # 1) light blur kills roof-tile / asphalt grain
        # 2) Sobel magnitude keeps thin building/road/car outlines
        # 3) keep as soft grayscale (no hard threshold / no thicken)
        blur_r = 0.8 if style == "edges_fine" else 1.35
        base = gray.filter(ImageFilter.GaussianBlur(radius=blur_r))
        arr = np.asarray(base, dtype=np.float32)
        # Sobel kernels
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        def conv2(a: np.ndarray, k: np.ndarray) -> np.ndarray:
            # reflect-pad then scipy-free convolution
            pad = 1
            p = np.pad(a, pad, mode="reflect")
            out = np.zeros_like(a)
            for i in range(k.shape[0]):
                for j in range(k.shape[1]):
                    out += k[i, j] * p[i : i + a.shape[0], j : j + a.shape[1]]
            return out

        gx = conv2(arr, kx)
        gy = conv2(arr, ky)
        mag = np.sqrt(gx * gx + gy * gy)
        # Suppress weak texture; keep mid/strong structure as continuous tones.
        lo = float(np.percentile(mag, 60 if style == "edges_fine" else 68))
        hi = float(np.percentile(mag, 99.2))
        mag = np.clip((mag - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        # Gentle gamma so fine lines stay visible without turning into white noise.
        mag = np.power(mag, 0.85)
        lines = (mag * 255.0).astype(np.uint8)
        # Tiny cleanup only — do NOT MaxFilter (that made edges chunky/noisy).
        lines_img = Image.fromarray(lines, mode="L")
        lines_img = ImageEnhance.Contrast(lines_img).enhance(1.25)
        rgb = Image.merge("RGB", (lines_img, lines_img, lines_img))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(out_path)
        return out_path

    # Default: printable stencil / stamp (like PS Filter Gallery → Sketch → Stamp).
    # Blur slightly, then hard threshold into ink vs paper.
    soft = gray.filter(ImageFilter.GaussianBlur(radius=1.2))
    soft = ImageEnhance.Contrast(soft).enhance(1.8)
    arr = np.asarray(soft, dtype=np.float32)
    # Adaptive-ish threshold around mid luminance.
    thr = float(np.median(arr) * 0.92 + np.percentile(arr, 40) * 0.08)
    ink = (arr < thr).astype(np.uint8) * 255  # black ink on white paper
    stencil = Image.fromarray(255 - ink, mode="L")  # white paper, black shapes
    # Clean speckles
    stencil = stencil.filter(ImageFilter.MedianFilter(3))
    # Optional outline emphasis: combine stamp with edges
    e = gray.filter(ImageFilter.FIND_EDGES)
    e_arr = np.asarray(ImageOps.autocontrast(e), dtype=np.float32)
    e_bin = (e_arr >= np.percentile(e_arr, 82)).astype(np.uint8) * 255
    stamp = np.asarray(stencil, dtype=np.uint8)
    # Burn edge lines into stamp (black where either is ink)
    combined = np.minimum(stamp, 255 - e_bin)
    out_img = Image.fromarray(combined, mode="L")
    rgb = Image.merge("RGB", (out_img, out_img, out_img))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path)
    return out_path


def build_edge_guided_prompt(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    meters_hint: str,
    center: Optional[dict] = None,
    image_width: int,
    image_height: int,
    guide_style: str = "edges_fine",
) -> str:
    """Prompt when Image1=Mapbox color and Image2=structure plate (edges/stencil/bw)."""
    base = build_survey_enhance_prompt(
        west=west,
        south=south,
        east=east,
        north=north,
        meters_hint=meters_hint,
        center=center,
        image_width=image_width,
        image_height=image_height,
        chained=False,
    )
    guide = (guide_style or "edges_fine").lower()
    if guide == "edges":
        guide = "edges_fine"
    guide_blurb = {
        "edges_fine": "Image 2 is a FINE soft EDGE MAP (thin detailed structure lines) derived from Image 1.",
        "edges_soft": "Image 2 is a soft EDGE MAP derived from Image 1.",
        "stencil": "Image 2 is a PRINTABLE STENCIL / STAMP plate (bold B&W shapes) derived from Image 1.",
        "bw": "Image 2 is a high-contrast BLACK-AND-WHITE luminance plate derived from Image 1.",
    }.get(guide, "Image 2 is a fine edge structure plate derived from Image 1.")
    return (
        base
        + f"""

STRUCTURE GUIDE (mandatory — survey / no hallucination):
- Image 1 = VIEWPORT SCREENSHOT of the live map canvas (exact pixels on screen). ONLY source of truth.
- {guide_blurb}
- Align roofs, roads, cars, tree canopies to Image 1 + Image 2 exactly.
- Sharpen / dehaze / add micro-detail ONLY on surfaces already visible in Image 1.
- FORBIDDEN: inventing buildings, roads, sidewalks, cars, trees, markings, pavement lines, or any object not clearly in Image 1.
- Ambiguous blur stays ambiguous — never promote a blotch into a new object.
- Do not change geometry, footprints, road centerlines, or object counts.
- Match Image 1 framing, size, and aspect exactly. Output must overlay Image 1 with near-zero misregistration.
"""
    ).strip()


def sharpen_mapbox(mapbox_path: Path, out_path: Path, *, scale: int = 2) -> Path:
    """Geometry-perfect enhance: bicubic upscale → unsharp → bicubic downscale.

    Survey-safe (no generative fill). Bicubic-at-2× gives the unsharp stage more
    samples to work on; downscaling back keeps the same pixel grid / georef.
    """
    from PIL import ImageFilter

    img = Image.open(mapbox_path).convert("RGB")
    w, h = img.size
    scale = max(2, int(scale))
    up = img.resize((w * scale, h * scale), Image.Resampling.BICUBIC)
    # Sharpen in the upscaled domain (mid edges then fine grit).
    sharp = up.filter(ImageFilter.UnsharpMask(radius=2.4 * scale / 2, percent=210, threshold=1))
    sharp = sharp.filter(ImageFilter.UnsharpMask(radius=1.1 * scale / 2, percent=150, threshold=2))
    # Return to original size with bicubic (Photoshop-style "bicubic sharper" cycle).
    out = sharp.resize((w, h), Image.Resampling.BICUBIC)
    # Light final bite at native size.
    out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


def lock_enhance_to_mapbox(
    mapbox_path: Path,
    candidate_path: Path,
    out_path: Path,
    *,
    min_ncc: float = 0.88,
) -> tuple[Path, float, str]:
    """Survey-safe sharpen: Mapbox geometry + gated high-frequency clarity.

    Strategy (keeps roofs/cars in the same places):
    1. Low frequencies always come from Mapbox (layout cannot drift).
    2. Boost Mapbox's own edges (makes existing car blobs crisper).
    3. Add Gemini high-pass detail only where local gradients agree with Mapbox
       (asphalt/tree/car micro-texture without inventing new buildings).
    """
    from PIL import ImageFilter
    import numpy as np

    ncc = structural_similarity_ncc(mapbox_path, candidate_path)
    src = Image.open(mapbox_path).convert("RGB")
    cand = Image.open(candidate_path).convert("RGB")
    if cand.size != src.size:
        cand = cand.resize(src.size, Image.Resampling.LANCZOS)

    if ncc < 0.70:
        sharpen_mapbox(mapbox_path, out_path)
        return out_path, ncc, "mapbox_unsharp_fallback"

    sa = np.asarray(src, dtype=np.float32)
    ca = np.asarray(cand, dtype=np.float32)

    # Soft gate from gradient agreement (cars/roads/roofs that exist in Mapbox).
    gray_s = sa.mean(axis=2)
    gray_c = ca.mean(axis=2)
    gsx = np.gradient(gray_s, axis=1)
    gsy = np.gradient(gray_s, axis=0)
    gcx = np.gradient(gray_c, axis=1)
    gcy = np.gradient(gray_c, axis=0)
    dot = gsx * gcx + gsy * gcy
    denom = np.sqrt((gsx * gsx + gsy * gsy) * (gcx * gcx + gcy * gcy)) + 1e-6
    agree = np.clip(dot / denom, 0.0, 1.0)
    edge_s = np.sqrt(gsx * gsx + gsy * gsy)
    edge_c = np.sqrt(gcx * gcx + gcy * gcy)
    edge_s_n = edge_s / (np.percentile(edge_s, 90) + 1e-6)
    edge_c_n = edge_c / (np.percentile(edge_c, 90) + 1e-6)
    gate = np.clip(0.35 + 0.65 * agree * np.clip(np.maximum(edge_s_n, edge_c_n), 0, 1), 0.35, 1.0)
    gate = gate[..., None].astype(np.float32)

    # When Gemini already matches Mapbox closely (edge-guided runs often ~0.95+),
    # only keep VERY coarse layout from Mapbox and let Gemini supply mid+high
    # detail — otherwise the lock re-blurs everything back to soft web tiles.
    if ncc >= 0.93:
        radius = 7.0
        mb_gain, ge_gain = 0.35, 1.05
        mode = "mapbox_coarse+gemini_detail_hi_ncc"
        unsharp = (1.2, 140)
    elif ncc >= min_ncc:
        radius = 3.0
        mb_gain, ge_gain = 0.85, 1.45
        mode = "mapbox_structure+gated_gemini_detail"
        unsharp = (1.4, 170)
    else:
        radius = 2.5
        mb_gain, ge_gain = 1.10, 0.90
        mode = "mapbox_structure+weak_gated_gemini_detail"
        unsharp = (1.5, 180)

    blur = ImageFilter.GaussianBlur(radius=radius)
    sb = np.asarray(src.filter(blur), dtype=np.float32)
    cb = np.asarray(cand.filter(blur), dtype=np.float32)
    mapbox_hf = np.clip(sa - sb, -64.0, 64.0)
    gemini_hf = np.clip(ca - cb, -80.0, 80.0)

    # Mapbox coarse + little Mapbox HF + gated Gemini mid/high detail.
    out = sb + mb_gain * mapbox_hf + ge_gain * gate * gemini_hf
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out, mode="RGB")
    out_img = out_img.filter(
        ImageFilter.UnsharpMask(radius=unsharp[0], percent=unsharp[1], threshold=2)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    return out_path, ncc, mode


def approx_meters(west: float, south: float, east: float, north: float) -> str:
    # rough degrees→meters at mid latitude
    cy = (south + north) / 2.0
    import math

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(cy))
    w = abs(east - west) * m_per_deg_lon
    h = abs(north - south) * m_per_deg_lat
    return f"about {w:.0f} m × {h:.0f} m"


def draw_geo_box(
    image: Image.Image,
    parent_bbox: Tuple[float, float, float, float],
    child_bbox: Tuple[float, float, float, float],
    color=(224, 49, 49),
    width: int = 5,
) -> Image.Image:
    """Draw child WGS84 bbox as a red rectangle inside a north-up parent image."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    pw, ph = img.size
    west, south, east, north = parent_bbox
    cwest, csouth, ceast, cnorth = child_bbox
    if east == west or north == south:
        return img

    def to_px(lon: float, lat: float) -> Tuple[float, float]:
        x = (lon - west) / (east - west) * pw
        y = (north - lat) / (north - south) * ph
        return x, y

    x0, y0 = to_px(cwest, cnorth)
    x1, y1 = to_px(ceast, csouth)
    x0, x1 = sorted((max(0, min(pw - 1, x0)), max(0, min(pw - 1, x1))))
    y0, y1 = sorted((max(0, min(ph - 1, y0)), max(0, min(ph - 1, y1))))
    if x1 - x0 < 4 or y1 - y0 < 4:
        # Keep a visible marker even if the child window is tiny.
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        x0, x1, y0, y1 = cx - 8, cx + 8, cy - 8, cy + 8
    for i in range(width):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)
    return img


def make_strip(panels: List[Tuple[str, Image.Image]], out_path: Path, caption: str) -> Path:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = font_sm = ImageFont.load_default()

    H = 420
    fitted = []
    for label, im in panels:
        im = im.convert("RGB")
        fitted.append((label, im.resize((int(im.width * H / im.height), H), Image.Resampling.LANCZOS)))

    pad = 12
    label_h = 36
    caption_h = 28
    width = sum(im.width for _, im in fitted) + pad * (len(fitted) + 1)
    canvas = Image.new("RGB", (width, H + label_h + caption_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    x = pad
    for label, im in fitted:
        draw.text((x, 8), label, fill=(255, 255, 255), font=font)
        canvas.paste(im, (x, label_h))
        x += im.width + pad
    draw.text((pad, H + label_h + 6), caption, fill=(180, 180, 180), font=font_sm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return out_path


def generate_for_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    out_dir: Path,
    model: str,
    skip_gemini: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "bbox": [west, south, east, north],
        "center": {"lon": (west + east) / 2.0, "lat": (south + north) / 2.0},
        "levels": [],
    }

    level_images: Dict[str, Path] = {}
    level_bboxes: Dict[str, Tuple[float, float, float, float]] = {}

    for name, scale, detail, offset_xy in LEVELS:
        b = scaled_bbox(west, south, east, north, scale, offset_xy=offset_xy)
        level_bboxes[name] = b
        mapbox_path = out_dir / f"mapbox_{name}.jpg"
        print(f"[mapbox] {name} scale={scale} offset={offset_xy} bbox={b}")
        download_mapbox(*b, mapbox_path)

        final_path = out_dir / f"zoom_{name}.png"
        if skip_gemini:
            Image.open(mapbox_path).convert("RGB").save(final_path)
        else:
            # Only the same-level Mapbox crop as reference keeps FOV honest.
            refs = [mapbox_path]
            prompt = build_prompt(
                level=name,
                detail=detail,
                west=b[0],
                south=b[1],
                east=b[2],
                north=b[3],
                meters_hint=approx_meters(*b),
            )
            print(f"[gemini] {name} via {model}")
            try:
                gemini_enhance(model=model, prompt=prompt, reference_paths=refs, out_path=final_path)
            except Exception as exc:
                print(f"[warn] Gemini failed for {name}: {exc}; using Mapbox")
                Image.open(mapbox_path).convert("RGB").save(final_path)

        level_images[name] = final_path
        meta["levels"].append(
            {
                "name": name,
                "scale": scale,
                "offset_xy": list(offset_xy),
                "bbox": list(b),
                "meters": approx_meters(*b),
                "file": final_path.name,
            }
        )

    # Build annotated strip with red square = next zoom FOV (geographic).
    panels = []
    for idx, (name, scale, _, _) in enumerate(LEVELS):
        img = Image.open(level_images[name]).convert("RGB")
        if idx + 1 < len(LEVELS):
            next_name = LEVELS[idx + 1][0]
            img = draw_geo_box(img, level_bboxes[name], level_bboxes[next_name])
            label = f"{name} -> {next_name}"
        else:
            label = f"{name} (max)"
        panels.append((label, img))

    strip_path = out_dir / "zoom_strip.jpg"
    make_strip(
        panels,
        strip_path,
        caption="Red square = next zoom target | Mapbox reference + Gemini enhancement",
    )
    preview = out_dir / "zoom_strip_preview.jpg"
    Image.open(strip_path).resize(
        (Image.open(strip_path).width // 2, Image.open(strip_path).height // 2),
        Image.Resampling.LANCZOS,
    ).save(preview, quality=85)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] strip -> {strip_path}")
    return strip_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate satellite zoom strip for a selected bbox")
    parser.add_argument("--bbox", type=Path, required=True, help="Path to bbox.json from bbox_selector.html")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/zoom_strips/<timestamp-or-name>)",
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV, help="Path to .env")
    parser.add_argument("--model", default=os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"))
    parser.add_argument(
        "--mapbox-only",
        action="store_true",
        help="Skip Gemini and only compose Mapbox tiles (useful for offline tests)",
    )
    args = parser.parse_args(argv)

    load_env(args.env)
    west, south, east, north = parse_bbox(args.bbox)

    out_dir = args.out
    if out_dir is None:
        out_dir = ROOT / "data" / "zoom_strips" / "latest"
    out_dir = out_dir.resolve()

    generate_for_bbox(
        west,
        south,
        east,
        north,
        out_dir=out_dir,
        model=args.model,
        skip_gemini=args.mapbox_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
