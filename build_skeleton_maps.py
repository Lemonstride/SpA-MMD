from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SESSION_DIR = Path("F:/datasets/SpA-MMD/processed/S01/walk")
COCO17_LIMBS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 2D/3D pose keypoints into SkeletonGait-style skeleton maps."
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help="Processed session directory containing labels/kpt2d.npy or labels/kpt3d.npy.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to a keypoint npy file. Defaults to labels/kpt2d.npy or labels/kpt3d.npy.",
    )
    parser.add_argument(
        "--kind",
        choices=["auto", "2d", "3d"],
        default="auto",
        help="Input keypoint kind. Use auto to infer from the file name or tensor shape.",
    )
    parser.add_argument(
        "--projection",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Projection plane used when the input is 3D.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Skeleton map height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Skeleton map width.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.8,
        help="Gaussian blur sigma used to smooth joints and limbs.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.2,
        help="Confidence threshold used to keep joints.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Relative padding added around the normalized skeleton.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Save per-frame PNG previews in labels/skeleton_map/png.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def infer_input_file(session_dir: Path) -> tuple[Path, str]:
    labels_dir = session_dir / "labels"
    kpt2d = labels_dir / "kpt2d.npy"
    kpt3d = labels_dir / "kpt3d.npy"
    if kpt2d.exists():
        return kpt2d, "2d"
    if kpt3d.exists():
        return kpt3d, "3d"
    raise SystemExit(f"未找到关键点文件: {kpt2d} 或 {kpt3d}")


def ensure_output_dirs(base_dir: Path, save_png: bool, overwrite: bool) -> tuple[Path, Path | None]:
    if base_dir.exists() and overwrite:
        for child in base_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested_dir in sorted([p for p in child.rglob("*") if p.is_dir()], reverse=True):
                    nested_dir.rmdir()
                child.rmdir()
    base_dir.mkdir(parents=True, exist_ok=True)
    png_dir = None
    if save_png:
        png_dir = base_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, png_dir


def load_keypoints(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 3:
        raise SystemExit(f"关键点数组形状错误，应为 [T, J, C]，实际为 {data.shape}")
    return data.astype(np.float32)


def detect_kind(path: Path, data: np.ndarray, kind: str) -> str:
    if kind != "auto":
        return kind
    name = path.name.lower()
    if "3d" in name:
        return "3d"
    if "2d" in name:
        return "2d"
    return "3d" if data.shape[-1] >= 4 else "2d"


def extract_coords_and_conf(data: np.ndarray, kind: str, projection: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "2d":
        coords = data[..., :2]
        conf = data[..., 2] if data.shape[-1] >= 3 else np.ones(data.shape[:2], dtype=np.float32)
        return coords, conf

    axis_map = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }
    axes = axis_map[projection]
    coords = data[..., list(axes)]
    conf = data[..., 3] if data.shape[-1] >= 4 else np.ones(data.shape[:2], dtype=np.float32)
    return coords, conf


def normalize_coords(coords: np.ndarray, conf: np.ndarray, width: int, height: int, padding: float) -> np.ndarray:
    valid = conf > 0
    if not np.any(valid):
        return np.zeros_like(coords)

    flat_valid = coords[valid]
    mins = flat_valid.min(axis=0)
    maxs = flat_valid.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)

    inner_width = width * (1.0 - 2.0 * padding)
    inner_height = height * (1.0 - 2.0 * padding)
    scale = min(inner_width / span[0], inner_height / span[1])

    normalized = (coords - mins) * scale
    bbox_size = span * scale
    offset_x = (width - bbox_size[0]) * 0.5
    offset_y = (height - bbox_size[1]) * 0.5
    normalized[..., 0] += offset_x
    normalized[..., 1] += offset_y
    return normalized


def draw_gaussian(canvas: np.ndarray, center: tuple[float, float], sigma: float) -> None:
    x, y = int(round(center[0])), int(round(center[1]))
    if x < 0 or y < 0 or x >= canvas.shape[1] or y >= canvas.shape[0]:
        return
    cv2.circle(canvas, (x, y), max(1, int(round(sigma * 1.5))), 255, thickness=-1)


def render_frame(coords: np.ndarray, conf: np.ndarray, width: int, height: int, sigma: float) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)

    if coords.shape[0] == 17:
        limbs = COCO17_LIMBS
    else:
        limbs = []

    for start, end in limbs:
        if conf[start] <= 0 or conf[end] <= 0:
            continue
        p1 = tuple(int(round(v)) for v in coords[start])
        p2 = tuple(int(round(v)) for v in coords[end])
        cv2.line(canvas, p1, p2, 180, thickness=2)

    for joint_index in range(coords.shape[0]):
        if conf[joint_index] <= 0:
            continue
        draw_gaussian(canvas, tuple(coords[joint_index]), sigma)

    canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return canvas.astype(np.uint8)


def build_maps(
    coords: np.ndarray,
    conf: np.ndarray,
    width: int,
    height: int,
    sigma: float,
    conf_thresh: float,
) -> np.ndarray:
    visible = np.where(conf >= conf_thresh, conf, 0.0)
    maps = np.zeros((coords.shape[0], height, width), dtype=np.uint8)
    for frame_index in range(coords.shape[0]):
        maps[frame_index] = render_frame(coords[frame_index], visible[frame_index], width, height, sigma)
    return maps


def save_pngs(png_dir: Path, maps: np.ndarray) -> None:
    for index, frame in enumerate(maps, start=1):
        cv2.imwrite(str(png_dir / f"frame_{index:06d}.png"), frame)


def write_meta(
    path: Path,
    source_file: Path,
    kind: str,
    projection: str,
    maps: np.ndarray,
    args: argparse.Namespace,
) -> None:
    payload = {
        "source_keypoints": str(source_file),
        "input_kind": kind,
        "projection": projection if kind == "3d" else "xy",
        "shape": list(maps.shape),
        "height": args.height,
        "width": args.width,
        "sigma": args.sigma,
        "conf_thresh": args.conf_thresh,
        "padding": args.padding,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir.expanduser().resolve()
    input_file, inferred_kind = infer_input_file(session_dir) if args.input is None else (args.input.expanduser().resolve(), args.kind)
    data = load_keypoints(input_file)
    kind = detect_kind(input_file, data, inferred_kind if args.input is None else args.kind)

    coords, conf = extract_coords_and_conf(data, kind, args.projection)
    coords = normalize_coords(coords, conf, args.width, args.height, args.padding)

    out_dir = session_dir / "labels" / "skeleton_map"
    out_dir, png_dir = ensure_output_dirs(out_dir, args.save_png, args.overwrite)
    maps = build_maps(coords, conf, args.width, args.height, args.sigma, args.conf_thresh)

    np.save(out_dir / "skeleton_maps.npy", maps)
    write_meta(out_dir / "meta.json", input_file, kind, args.projection, maps, args)

    if png_dir is not None:
        save_pngs(png_dir, maps)

    print(f"Skeleton maps 已生成: {out_dir}")
    print(f"输入关键点: {input_file}")
    print(f"输入类型: {kind}")
    print(f"输出形状: {maps.shape}")


if __name__ == "__main__":
    main()
