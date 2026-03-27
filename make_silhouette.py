from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SESSION_DIR = Path("/your/own/path/output/session1")
HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate binary silhouettes from exported RealSense RGB/depth images."
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help="Session directory that contains depth/ and where silhouette/ will be created.",
    )
    parser.add_argument(
        "--depth-margin-mm",
        type=int,
        default=180,
        help="Foreground threshold in millimeters compared with estimated background depth.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=48,
        help="Number of sampled depth frames used to estimate the static background.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=3000,
        help="Minimum connected-component area kept as a person silhouette.",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=int,
        default=6000,
        help="Ignore depth values farther than this distance.",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Also save a foreground preview image for quick checking.",
    )
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=0,
        help="Temporal smoothing radius. 1 means use previous/current/next frames.",
    )
    parser.add_argument(
        "--use-hog-detector",
        action="store_true",
        help="Use OpenCV HOG person detector to refine the depth box. Slower and disabled by default.",
    )
    return parser.parse_args()


def list_depth_images(depth_dir: Path) -> list[Path]:
    images = sorted(depth_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"没有找到深度图: {depth_dir}")
    return images


def list_rgb_images(rgb_dir: Path) -> list[Path]:
    images = sorted(rgb_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"没有找到 RGB 图像: {rgb_dir}")
    return images


def uniform_sample(paths: list[Path], sample_count: int) -> list[Path]:
    if len(paths) <= sample_count:
        return paths
    indices = np.linspace(0, len(paths) - 1, sample_count, dtype=int)
    return [paths[i] for i in indices]


def load_depth(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"读取失败: {path}")
    if image.dtype != np.uint16:
        raise RuntimeError(f"深度图不是 uint16: {path}")
    return image


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"读取失败: {path}")
    return image


def smooth_depth(depth: np.ndarray) -> np.ndarray:
    # Median filtering removes isolated depth spikes while preserving body boundaries.
    return cv2.medianBlur(depth, 5)


def estimate_background(sample_paths: list[Path], max_depth_mm: int) -> np.ndarray:
    stack = []
    for path in sample_paths:
        depth = smooth_depth(load_depth(path))
        valid = (depth > 0) & (depth <= max_depth_mm)
        stack.append(np.where(valid, depth, 0).astype(np.uint16))

    background = np.max(np.stack(stack, axis=0), axis=0)
    return background


def largest_component(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, dict[str, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_label = 0
    best_area = 0
    best_box = {"x": 0, "y": 0, "w": 0, "h": 0}

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area <= best_area:
            continue

        best_label = label
        best_area = area
        best_box = {
            "x": int(stats[label, cv2.CC_STAT_LEFT]),
            "y": int(stats[label, cv2.CC_STAT_TOP]),
            "w": int(stats[label, cv2.CC_STAT_WIDTH]),
            "h": int(stats[label, cv2.CC_STAT_HEIGHT]),
        }

    if best_label == 0:
        return np.zeros_like(mask), {"x": 0, "y": 0, "w": 0, "h": 0, "area": 0}

    person_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    best_box["area"] = best_area
    return person_mask, best_box


def clip_box(box: dict[str, int], shape: tuple[int, int]) -> dict[str, int]:
    height, width = shape
    x = max(0, min(width - 1, box["x"]))
    y = max(0, min(height - 1, box["y"]))
    w = max(1, min(width - x, box["w"]))
    h = max(1, min(height - y, box["h"]))
    return {"x": x, "y": y, "w": w, "h": h, "area": box.get("area", w * h)}


def expand_box(box: dict[str, int], shape: tuple[int, int], pad_ratio: float = 0.12) -> dict[str, int]:
    pad_x = max(20, int(box["w"] * pad_ratio))
    pad_y = max(20, int(box["h"] * pad_ratio))
    expanded = {
        "x": box["x"] - pad_x,
        "y": box["y"] - pad_y,
        "w": box["w"] + pad_x * 2,
        "h": box["h"] + pad_y * 2,
    }
    return clip_box(expanded, shape)


def iou(box_a: dict[str, int], box_b: dict[str, int]) -> float:
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["w"], ay1 + box_a["h"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["w"], by1 + box_b["h"]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = box_a["w"] * box_a["h"] + box_b["w"] * box_b["h"] - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(primary: dict[str, int], secondary: dict[str, int], shape: tuple[int, int]) -> dict[str, int]:
    x1 = min(primary["x"], secondary["x"])
    y1 = min(primary["y"], secondary["y"])
    x2 = max(primary["x"] + primary["w"], secondary["x"] + secondary["w"])
    y2 = max(primary["y"] + primary["h"], secondary["y"] + secondary["h"])
    return clip_box({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}, shape)


def box_to_mask(box: dict[str, int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if box["area"] == 0:
        return mask
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    mask[y:y + h, x:x + w] = 255
    return mask


def fallback_foreground(depth: np.ndarray, max_depth_mm: int, min_area: int) -> np.ndarray:
    valid = depth[(depth > 0) & (depth <= max_depth_mm)]
    if valid.size == 0:
        return np.zeros(depth.shape, dtype=np.uint8)

    near_threshold = int(np.percentile(valid, 15))
    mask = np.where((depth > 0) & (depth <= near_threshold), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    person_mask, _ = largest_component(mask, min_area)
    return person_mask


def detect_person_box(rgb: np.ndarray, fallback_box: dict[str, int], use_hog: bool) -> dict[str, int]:
    if not use_hog:
        return fallback_box

    height, width = rgb.shape[:2]
    scale = min(1.0, 960.0 / width)
    resized = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    rects, weights = HOG.detectMultiScale(
        resized,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.03,
    )

    if len(rects) == 0:
        return fallback_box

    candidates = []
    for rect, weight in zip(rects, weights):
        x, y, w, h = rect
        box = {
            "x": int(x / scale),
            "y": int(y / scale),
            "w": int(w / scale),
            "h": int(h / scale),
            "area": int((w / scale) * (h / scale)),
        }
        box = clip_box(box, (height, width))
        score = float(weight)
        if fallback_box["area"] > 0:
            score += 2.0 * iou(box, fallback_box)
        candidates.append((score, box))

    _, best_box = max(candidates, key=lambda item: item[0])
    if fallback_box["area"] == 0:
        return best_box
    if iou(best_box, fallback_box) < 0.1:
        return fallback_box
    return merge_boxes(fallback_box, best_box, (height, width))


def refine_silhouette(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, dict[str, int]]:
    if mask.max() == 0:
        return np.zeros_like(mask), {"x": 0, "y": 0, "w": 0, "h": 0, "area": 0}

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_kernel)
    height, width = refined.shape
    scaled_size = (max(1, width // 2), max(1, height // 2))
    refined = cv2.resize(refined, scaled_size, interpolation=cv2.INTER_AREA)
    refined = cv2.GaussianBlur(refined, (5, 5), sigmaX=0)
    refined = cv2.resize(refined, (width, height), interpolation=cv2.INTER_LINEAR)
    _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask), {"x": 0, "y": 0, "w": 0, "h": 0, "area": 0}

    contour = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, close_kernel)
    clean, box = largest_component(clean, min_area)
    return clean, box


def depth_foreground_mask(
    depth: np.ndarray,
    background: np.ndarray,
    depth_margin_mm: int,
    max_depth_mm: int,
    min_area: int,
) -> tuple[np.ndarray, dict[str, int]]:
    depth = smooth_depth(depth)
    valid = (depth > 0) & (depth <= max_depth_mm)
    bg_valid = background > 0
    delta = background.astype(np.int32) - depth.astype(np.int32)
    mask = np.where(valid & bg_valid & (delta >= depth_margin_mm), 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    person_mask, box = largest_component(mask, min_area)
    if box["area"] == 0:
        person_mask = fallback_foreground(depth, max_depth_mm, min_area)
        person_mask, box = largest_component(person_mask, min_area)

    return person_mask, box


def local_depth_gate(
    depth: np.ndarray,
    rough_mask: np.ndarray,
    box: dict[str, int],
    max_depth_mm: int,
) -> np.ndarray:
    return rough_mask


def suppress_ground(mask: np.ndarray, box: dict[str, int]) -> np.ndarray:
    if box["area"] == 0:
        return mask

    cleaned = mask.copy()
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    foot_band_top = y + int(h * 0.82)
    foot_band_top = min(max(y, foot_band_top), y + h - 1)

    bottom_band = cleaned[foot_band_top:y + h, x:x + w]
    if bottom_band.size == 0:
        return cleaned

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bottom_band, connectivity=8)
    for label in range(1, num_labels):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        touches_bottom = top + height >= bottom_band.shape[0] - 1
        too_wide = width >= int(w * 0.42)
        too_flat = width >= max(12, height * 2)
        if touches_bottom and too_wide and too_flat and area > 300:
            bottom_band[labels == label] = 0

    cleaned[foot_band_top:y + h, x:x + w] = bottom_band
    return cleaned


def grabcut_refine(rgb: np.ndarray, rough_mask: np.ndarray, box: dict[str, int]) -> np.ndarray:
    if box["area"] == 0:
        return rough_mask

    h, w = rough_mask.shape
    expanded = expand_box(box, (h, w))
    x1, y1, x2, y2 = (
        expanded["x"],
        expanded["y"],
        expanded["x"] + expanded["w"],
        expanded["y"] + expanded["h"],
    )

    roi_rgb = rgb[y1:y2, x1:x2].copy()
    roi_mask = rough_mask[y1:y2, x1:x2]
    gc_mask = np.full(roi_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)

    probable_fg = cv2.dilate(
        roi_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    sure_fg = cv2.erode(
        roi_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )

    gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(roi_rgb, gc_mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)

    roi_refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    refined = np.zeros_like(rough_mask)
    refined[y1:y2, x1:x2] = roi_refined
    return refined


def build_silhouette(
    rgb: np.ndarray,
    depth: np.ndarray,
    background: np.ndarray,
    depth_margin_mm: int,
    max_depth_mm: int,
    min_area: int,
    use_hog: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    rough_mask, box = depth_foreground_mask(
        depth=depth,
        background=background,
        depth_margin_mm=depth_margin_mm,
        max_depth_mm=max_depth_mm,
        min_area=min_area,
    )
    detected_box = detect_person_box(rgb, box, use_hog)
    gated_mask = local_depth_gate(depth, rough_mask, detected_box, max_depth_mm)
    seed_mask = cv2.bitwise_and(
        cv2.dilate(rough_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))),
        gated_mask,
    )
    fused_mask = grabcut_refine(rgb, seed_mask, detected_box)
    fused_mask = cv2.bitwise_and(
        fused_mask,
        box_to_mask(expand_box(detected_box, fused_mask.shape), fused_mask.shape),
    )
    person_mask, box = refine_silhouette(fused_mask, min_area)
    if box["area"] == 0:
        person_mask, box = refine_silhouette(rough_mask, min_area)
    person_mask = suppress_ground(person_mask, box)
    person_mask, box = refine_silhouette(person_mask, min_area)

    return person_mask, box


def temporal_smooth(masks: list[np.ndarray], radius: int) -> list[np.ndarray]:
    if radius <= 0 or len(masks) < 3:
        return masks

    smoothed: list[np.ndarray] = []
    for index in range(len(masks)):
        start = max(0, index - radius)
        end = min(len(masks), index + radius + 1)
        window = np.stack([(m > 0).astype(np.uint8) for m in masks[start:end]], axis=0)
        votes = np.sum(window, axis=0)
        threshold = (window.shape[0] // 2) + 1
        smoothed.append(np.where(votes >= threshold, 255, 0).astype(np.uint8))
    return smoothed


def write_rows(csv_path: Path, rows: list[dict[str, int | str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "area", "x", "y", "w", "h"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    session_dir = args.session_dir.expanduser().resolve()
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"
    silhouette_dir = session_dir / "silhouette"
    meta_dir = session_dir / "meta"
    preview_dir = session_dir / "silhouette_preview"

    rgb_paths = list_rgb_images(rgb_dir)
    depth_paths = list_depth_images(depth_dir)
    if len(rgb_paths) != len(depth_paths):
        raise SystemExit("RGB 和 depth 帧数不一致，先检查导出结果。")
    silhouette_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    background = estimate_background(
        uniform_sample(depth_paths, args.sample_count),
        args.max_depth_mm,
    )

    raw_masks: list[np.ndarray] = []
    boxes: list[dict[str, int]] = []
    rows: list[dict[str, int | str]] = []
    for rgb_path, depth_path in zip(rgb_paths, depth_paths):
        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_path)
        silhouette, box = build_silhouette(
            rgb=rgb,
            depth=depth,
            background=background,
            depth_margin_mm=args.depth_margin_mm,
            max_depth_mm=args.max_depth_mm,
            min_area=args.min_area,
            use_hog=args.use_hog_detector,
        )
        raw_masks.append(silhouette)
        boxes.append(box)

    final_masks = temporal_smooth(raw_masks, args.temporal_radius)

    for (rgb_path, depth_path, silhouette, rough_box) in zip(rgb_paths, depth_paths, final_masks, boxes):
        rgb = load_rgb(rgb_path)
        silhouette, box = refine_silhouette(silhouette, args.min_area)
        if box["area"] == 0:
            box = rough_box

        out_path = silhouette_dir / depth_path.name.replace("depth_", "silhouette_")
        cv2.imwrite(str(out_path), silhouette)

        if args.save_preview:
            preview = rgb.copy()
            preview[silhouette > 0] = (0.5 * preview[silhouette > 0] + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            if box["area"] > 0:
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(str(preview_dir / out_path.name), preview)

        rows.append(
            {
                "file_name": out_path.name,
                "area": int(box["area"]),
                "x": int(box["x"]),
                "y": int(box["y"]),
                "w": int(box["w"]),
                "h": int(box["h"]),
            }
        )

    write_rows(meta_dir / "silhouette_frames.csv", rows)
    print(f"silhouette 输出目录: {silhouette_dir}")
    print(f"生成数量: {len(rows)}")


if __name__ == "__main__":
    main()
