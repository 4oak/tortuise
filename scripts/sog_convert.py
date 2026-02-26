#!/usr/bin/env python3
"""Convert PlayCanvas SOG assets (meta.json + WebP files) to binary little-endian PLY.

Usage:
    python3 sog_convert.py <input_path_or_url> <output.ply>

Input may be:
- Local directory containing meta.json and WebP files
- Local path to meta.json
- Scene ID (8 hex chars) for the SuperSplat CDN
- superspl.at URL containing a scene ID
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import struct
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    print("error: numpy is required. Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("error: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
    print("  Or: pip install -r scripts/requirements.txt", file=sys.stderr)
    sys.exit(1)

CDN_BASE = "https://d28zzqy0iyovbz.cloudfront.net"
SCENE_ID_RE = re.compile(r"^[0-9a-fA-F]{8}$")
SCENE_ID_SEARCH_RE = re.compile(r"([0-9a-fA-F]{8})")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def usage() -> int:
    print("Usage: python3 sog_convert.py <input_path_or_url> <output.ply>", file=sys.stderr)
    return 2


def is_scene_id(text: str) -> bool:
    return SCENE_ID_RE.fullmatch(text) is not None


def extract_scene_id_from_url(text: str) -> str | None:
    parsed = urllib.parse.urlparse(text)
    if not parsed.scheme or not parsed.netloc:
        return None
    haystack = f"{parsed.path}/{parsed.query}/{parsed.fragment}"
    match = SCENE_ID_SEARCH_RE.search(haystack)
    return match.group(1).lower() if match else None


def fetch_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error fetching {url}: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("meta.json root must be an object")
    return data


def collect_file_references(obj: Any) -> List[str]:
    found: List[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            files = node.get("files")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, str) and item.lower().endswith(".webp"):
                        found.append(item)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(obj)
    deduped = sorted(set(found))
    return deduped


def download_scene(scene_id: str, temp_dir: str) -> str:
    scene_id = scene_id.lower()
    meta_url = f"{CDN_BASE}/{scene_id}/v3/meta.json"
    log(f"Downloading meta.json: {meta_url}")
    meta_bytes = fetch_bytes(meta_url)

    meta_path = os.path.join(temp_dir, "meta.json")
    with open(meta_path, "wb") as f:
        f.write(meta_bytes)

    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid remote meta.json for scene {scene_id}: {exc}") from exc

    file_names = collect_file_references(meta)
    for filename in file_names:
        file_url = f"{CDN_BASE}/{scene_id}/v3/{filename}"
        log(f"Downloading {filename}: {file_url}")
        content = fetch_bytes(file_url)
        out_path = os.path.join(temp_dir, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(content)

    return meta_path


def resolve_input_to_meta_path(input_arg: str) -> Tuple[str, tempfile.TemporaryDirectory[str] | None]:
    if os.path.isdir(input_arg):
        meta_path = os.path.join(input_arg, "meta.json")
        if not os.path.isfile(meta_path):
            raise RuntimeError(f"Directory does not contain meta.json: {input_arg}")
        return meta_path, None

    if os.path.isfile(input_arg):
        return input_arg, None

    scene_id: str | None = None
    if is_scene_id(input_arg):
        scene_id = input_arg.lower()
    else:
        scene_id = extract_scene_id_from_url(input_arg)

    if scene_id is None:
        raise RuntimeError(
            "Input must be a directory, meta.json path, scene ID, or superspl.at URL"
        )

    tmp = tempfile.TemporaryDirectory(prefix="sog_convert_")
    meta_path = download_scene(scene_id, tmp.name)
    return meta_path, tmp


def require_keys(obj: Dict[str, Any], keys: Sequence[str], ctx: str) -> None:
    for key in keys:
        if key not in obj:
            raise ValueError(f"Missing key '{key}' in {ctx}")


def validate_meta(meta: Dict[str, Any]) -> None:
    require_keys(meta, ["count", "means", "scales", "quats", "sh0"], "meta.json")

    count = meta["count"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("meta.count must be a positive integer")

    means = meta["means"]
    scales = meta["scales"]
    quats = meta["quats"]
    sh0 = meta["sh0"]
    if not isinstance(means, dict) or not isinstance(scales, dict):
        raise ValueError("meta.means and meta.scales must be objects")
    if not isinstance(quats, dict) or not isinstance(sh0, dict):
        raise ValueError("meta.quats and meta.sh0 must be objects")

    require_keys(means, ["mins", "maxs", "files"], "means")
    require_keys(scales, ["codebook", "files"], "scales")
    require_keys(quats, ["files"], "quats")
    require_keys(sh0, ["codebook", "files"], "sh0")

    if not isinstance(means["mins"], list) or len(means["mins"]) != 3:
        raise ValueError("means.mins must be length-3 array")
    if not isinstance(means["maxs"], list) or len(means["maxs"]) != 3:
        raise ValueError("means.maxs must be length-3 array")

    if not isinstance(means["files"], list) or len(means["files"]) < 2:
        raise ValueError("means.files must include at least [means_l.webp, means_u.webp]")
    if not isinstance(scales["files"], list) or len(scales["files"]) < 1:
        raise ValueError("scales.files must include scales.webp")
    if not isinstance(quats["files"], list) or len(quats["files"]) < 1:
        raise ValueError("quats.files must include quats.webp")
    if not isinstance(sh0["files"], list) or len(sh0["files"]) < 1:
        raise ValueError("sh0.files must include sh0.webp")

    if not isinstance(scales["codebook"], list) or len(scales["codebook"]) != 256:
        raise ValueError("scales.codebook must have 256 entries")
    if not isinstance(sh0["codebook"], list) or len(sh0["codebook"]) != 256:
        raise ValueError("sh0.codebook must have 256 entries")


def load_rgba_pixels(path: str) -> np.ndarray:
    log(f"Loading image: {path}")
    try:
        with open(path, "rb") as f:
            raw = f.read()
        with Image.open(io.BytesIO(raw)) as img:
            rgba = img.convert("RGBA")
            arr = np.asarray(rgba, dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(f"Failed to decode image {path}: {exc}") from exc

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise RuntimeError(f"Decoded image {path} is not RGBA")
    flat = arr.reshape(-1, 4)
    return flat


def inv_log_transform(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * (np.exp(np.abs(v)) - 1.0)


def sigmoid_inv(y: np.ndarray) -> np.ndarray:
    e = np.clip(y, 1e-6, 1.0 - 1e-6)
    return np.log(e / (1.0 - e))


def decode_positions(
    lo: np.ndarray, hi: np.ndarray, mins: Sequence[float], maxs: Sequence[float], count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo_c = lo[:count]
    hi_c = hi[:count]

    xs = lo_c[:, 0].astype(np.uint16) | (hi_c[:, 0].astype(np.uint16) << 8)
    ys = lo_c[:, 1].astype(np.uint16) | (hi_c[:, 1].astype(np.uint16) << 8)
    zs = lo_c[:, 2].astype(np.uint16) | (hi_c[:, 2].astype(np.uint16) << 8)

    mins_f = np.asarray(mins, dtype=np.float32)
    maxs_f = np.asarray(maxs, dtype=np.float32)
    scales = maxs_f - mins_f
    scales = np.where(scales == 0.0, 1.0, scales)

    unit = np.float32(1.0 / 65535.0)
    lx = mins_f[0] + scales[0] * (xs.astype(np.float32) * unit)
    ly = mins_f[1] + scales[1] * (ys.astype(np.float32) * unit)
    lz = mins_f[2] + scales[2] * (zs.astype(np.float32) * unit)

    x = inv_log_transform(lx).astype(np.float32)
    y = inv_log_transform(ly).astype(np.float32)
    z = inv_log_transform(lz).astype(np.float32)
    return x, y, z


def decode_quats(qr: np.ndarray, count: int) -> np.ndarray:
    qr_c = qr[:count]
    tags = qr_c[:, 3]
    valid = (tags >= 252) & (tags <= 255)

    rot = np.zeros((count, 4), dtype=np.float32)
    rot[:, 3] = 1.0

    if not np.any(valid):
        return rot

    idx = np.nonzero(valid)[0]
    tag_v = tags[valid].astype(np.int32)
    max_comp = tag_v - 252

    qv = qr_c[valid, :3].astype(np.float32)
    a = qv[:, 0] / 255.0 * 2.0 - 1.0
    b = qv[:, 1] / 255.0 * 2.0 - 1.0
    c = qv[:, 2] / 255.0 * 2.0 - 1.0

    sqrt2 = np.float32(math.sqrt(2.0))
    abc = np.stack((a / sqrt2, b / sqrt2, c / sqrt2), axis=1)

    # For each max-component tag (0..3), map a/b/c to the remaining 3 quaternion indices.
    idx_map = np.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64
    )
    target_cols = idx_map[max_comp]

    rv = np.zeros((idx.size, 4), dtype=np.float32)
    row_ids = np.arange(idx.size)[:, None]
    rv[row_ids, target_cols] = abc

    t = 1.0 - np.sum(rv * rv, axis=1)
    rv[np.arange(idx.size), max_comp] = np.sqrt(np.maximum(0.0, t))
    rot[idx] = rv
    return rot


def decode_sog(meta_path: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log(f"Loading meta.json: {meta_path}")
    meta = read_json(meta_path)
    validate_meta(meta)

    base_dir = os.path.dirname(os.path.abspath(meta_path))
    count = int(meta["count"])

    means = meta["means"]
    scales_meta = meta["scales"]
    quats_meta = meta["quats"]
    sh0_meta = meta["sh0"]

    means_lo_path = os.path.join(base_dir, means["files"][0])
    means_hi_path = os.path.join(base_dir, means["files"][1])
    scales_path = os.path.join(base_dir, scales_meta["files"][0])
    quats_path = os.path.join(base_dir, quats_meta["files"][0])
    sh0_path = os.path.join(base_dir, sh0_meta["files"][0])

    lo = load_rgba_pixels(means_lo_path)
    hi = load_rgba_pixels(means_hi_path)
    sl = load_rgba_pixels(scales_path)
    qr = load_rgba_pixels(quats_path)
    c0 = load_rgba_pixels(sh0_path)

    for path, pixels in (
        (means_lo_path, lo),
        (means_hi_path, hi),
        (scales_path, sl),
        (quats_path, qr),
        (sh0_path, c0),
    ):
        if pixels.shape[0] < count:
            raise RuntimeError(
                f"Image has too few pixels for {count} gaussians: {path} ({pixels.shape[0]})"
            )

    log(f"Decoding means for {count} gaussians")
    x, y, z = decode_positions(lo, hi, means["mins"], means["maxs"], count)

    log(f"Decoding scales for {count} gaussians")
    scale_codebook = np.asarray(scales_meta["codebook"], dtype=np.float32)
    sl_c = sl[:count]
    scale_0 = scale_codebook[sl_c[:, 0]]
    scale_1 = scale_codebook[sl_c[:, 1]]
    scale_2 = scale_codebook[sl_c[:, 2]]

    log(f"Decoding SH0 + opacity for {count} gaussians")
    sh0_codebook = np.asarray(sh0_meta["codebook"], dtype=np.float32)
    c0_c = c0[:count]
    f_dc_0 = sh0_codebook[c0_c[:, 0]]
    f_dc_1 = sh0_codebook[c0_c[:, 1]]
    f_dc_2 = sh0_codebook[c0_c[:, 2]]
    opacity = sigmoid_inv(c0_c[:, 3].astype(np.float32) / 255.0).astype(np.float32)

    log(f"Decoding quaternions for {count} gaussians")
    rot = decode_quats(qr, count)

    return count, x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, np.stack((scale_0, scale_1, scale_2), axis=1), rot


def write_ply(
    output_path: str,
    count: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    f_dc_0: np.ndarray,
    f_dc_1: np.ndarray,
    f_dc_2: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    rot: np.ndarray,
) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    log(f"Writing PLY: {output_path}")
    vertices = np.empty((count, 14), dtype="<f4")
    vertices[:, 0] = x
    vertices[:, 1] = y
    vertices[:, 2] = z
    vertices[:, 3] = f_dc_0
    vertices[:, 4] = f_dc_1
    vertices[:, 5] = f_dc_2
    vertices[:, 6] = opacity
    vertices[:, 7] = scales[:, 0]
    vertices[:, 8] = scales[:, 1]
    vertices[:, 9] = scales[:, 2]
    vertices[:, 10] = rot[:, 0]
    vertices[:, 11] = rot[:, 1]
    vertices[:, 12] = rot[:, 2]
    vertices[:, 13] = rot[:, 3]

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes(order="C"))


def main(argv: Sequence[str]) -> int:
    if len(argv) != 3:
        return usage()

    input_arg = argv[1]
    output_path = argv[2]

    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    try:
        meta_path, temp_ctx = resolve_input_to_meta_path(input_arg)
        (
            count,
            x,
            y,
            z,
            f_dc_0,
            f_dc_1,
            f_dc_2,
            opacity,
            scales,
            rot,
        ) = decode_sog(meta_path)
        write_ply(output_path, count, x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, scales, rot)
        size = os.path.getsize(output_path)
        log(f"Done: {output_path} ({size} bytes)")
        return 0
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
