from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SESSION_DIR = Path("F:/SpA-MMD/processed/S01/head_turn")
DEFAULT_ROOT_DIR = Path("F:/SpA-MMD/processed")
LANDMARK_INDEX = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate head-turn state and peak frames from kpt2d + aligned depth."
    )
    parser.add_argument("--session-dir", type=Path, default=None, help="Processed head_turn session directory.")
    parser.add_argument("--root-dir", type=Path, default=None, help="Processed dataset root for batch processing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--save-vis", action="store_true", help="Save annotated RGB previews.")
    parser.add_argument("--conf-thresh", type=float, default=0.35, help="Minimum 2D landmark confidence.")
    parser.add_argument("--depth-window", type=int, default=7, help="Odd window size for depth sampling.")
    parser.add_argument("--max-depth-m", type=float, default=3.5, help="Ignore larger sampled depths.")
    parser.add_argument("--smooth-window", type=int, default=9, help="Odd smoothing window for proxy score.")
    parser.add_argument(
        "--state-threshold",
        type=float,
        default=0.18,
        help="Absolute relative proxy threshold used for non-center classification.",
    )
    return parser.parse_args()


def iter_head_turn_sessions(root_dir: Path) -> list[Path]:
    session_dirs: list[Path] = []
    for subject_dir in sorted(root_dir.glob("S*")):
        if not subject_dir.is_dir():
            continue
        session_dir = subject_dir / "head_turn"
        if session_dir.is_dir():
            session_dirs.append(session_dir)
    if not session_dirs:
        raise SystemExit(f"在 {root_dir} 下没有找到 head_turn session")
    return session_dirs


def load_kpt2d(session_dir: Path) -> np.ndarray:
    candidates = [
        session_dir / "skeleton" / "kpt2d" / "kpt2d.npy",
        session_dir / "labels" / "kpt2d.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path).astype(np.float32)
    raise SystemExit("没有找到 kpt2d.npy")


def list_images(path: Path) -> list[Path]:
    files = sorted(path.glob("*.png"))
    if not files:
        raise SystemExit(f"目录为空: {path}")
    return files


def load_depth_scale(session_dir: Path) -> float:
    path = session_dir / "calib" / "depth_scale.txt"
    if not path.exists():
        raise SystemExit(f"缺少 depth_scale.txt: {path}")
    return float(path.read_text(encoding="utf-8").strip())


def ensure_output_dir(session_dir: Path, overwrite: bool, save_vis: bool) -> tuple[Path, Path | None]:
    output_dir = session_dir / "labels" / "head_turn_state"
    csv_path = output_dir / "states.csv"
    json_path = output_dir / "summary.json"
    if not overwrite and (csv_path.exists() or json_path.exists()):
        existing = csv_path if csv_path.exists() else json_path
        raise SystemExit(f"输出文件已存在，请使用 --overwrite 覆盖: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = None
    if save_vis:
        vis_dir = output_dir / "vis"
        if vis_dir.exists() and overwrite:
            for file_path in vis_dir.glob("*.png"):
                file_path.unlink()
        vis_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, vis_dir


def export_key_frame(
    source_rgb: Path,
    destination: Path,
    score: float | None,
    relative_score: float | None,
    image_state: str,
    subject_state: str,
    quality: str,
) -> None:
    image = cv2.imread(str(source_rgb), cv2.IMREAD_COLOR)
    if image is None:
        return
    annotated = annotate_frame(
        image=image,
        score=score,
        relative_score=relative_score,
        image_state=image_state,
        subject_state=subject_state,
        quality=quality,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), annotated)


def sample_depth_m(
    depth_raw: np.ndarray,
    u: float,
    v: float,
    window: int,
    depth_scale: float,
    max_depth_m: float,
) -> float | None:
    if window % 2 == 0:
        window += 1
    x = int(round(u))
    y = int(round(v))
    radius = window // 2
    depth_m = depth_raw.astype(np.float32) * depth_scale
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(depth_m.shape[1], x + radius + 1)
    y2 = min(depth_m.shape[0], y + radius + 1)
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[(patch > 0.0) & (patch <= max_depth_m)]
    if valid.size == 0:
        return None
    lo = np.percentile(valid, 20)
    hi = np.percentile(valid, 80)
    robust = valid[(valid >= lo) & (valid <= hi)]
    if robust.size == 0:
        robust = valid
    return float(np.median(robust))


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    output = values.copy()
    for index in range(values.shape[0]):
        start = max(0, index - radius)
        end = min(values.shape[0], index + radius + 1)
        chunk = values[start:end]
        valid = chunk[np.isfinite(chunk)]
        if valid.size == 0:
            output[index] = np.nan
        else:
            output[index] = float(valid.mean())
    return output


def classify_state(relative_score: float | None, threshold: float) -> str:
    if relative_score is None or not np.isfinite(relative_score):
        return "unknown"
    if relative_score >= threshold:
        return "image_left"
    if relative_score <= -threshold:
        return "image_right"
    return "center"


def image_state_to_subject_state(state: str) -> str:
    mapping = {
        "image_left": "subject_right",
        "image_right": "subject_left",
        "center": "center",
        "unknown": "unknown",
    }
    return mapping.get(state, "unknown")


def find_neutral_frame_index(relative_scores: np.ndarray) -> int | None:
    valid_indices = np.where(np.isfinite(relative_scores))[0]
    if valid_indices.size == 0:
        return None
    best_pos = int(valid_indices[np.argmin(np.abs(relative_scores[valid_indices]))])
    return best_pos + 1


def annotate_frame(
    image: np.ndarray,
    score: float | None,
    relative_score: float | None,
    image_state: str,
    subject_state: str,
    quality: str,
) -> np.ndarray:
    canvas = image.copy()
    score_text = f"{score:.3f}" if score is not None and np.isfinite(score) else "N/A"
    relative_text = f"{relative_score:.3f}" if relative_score is not None and np.isfinite(relative_score) else "N/A"
    lines = [
        f"score: {score_text}",
        f"relative: {relative_text}",
        f"image_state: {image_state}",
        f"subject_state: {subject_state}",
        f"quality: {quality}",
    ]
    y = 32
    for line in lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
        y += 32
    return canvas


def process_session(session_dir: Path, args: argparse.Namespace) -> None:
    kpt2d = load_kpt2d(session_dir)
    rgb_files = list_images(session_dir / "rgb")
    depth_files = list_images(session_dir / "depth")
    if len(rgb_files) != len(depth_files) or len(rgb_files) != kpt2d.shape[0]:
        raise SystemExit("rgb/depth/kpt2d 帧数不一致")

    output_dir, vis_dir = ensure_output_dir(session_dir, args.overwrite, args.save_vis)
    depth_scale = load_depth_scale(session_dir)

    raw_scores = np.full((kpt2d.shape[0],), np.nan, dtype=np.float32)
    nose_offset_norm = np.full((kpt2d.shape[0],), np.nan, dtype=np.float32)
    ear_depth_diff = np.full((kpt2d.shape[0],), np.nan, dtype=np.float32)
    quality = ["invalid"] * kpt2d.shape[0]

    for index in range(kpt2d.shape[0]):
        frame = kpt2d[index]
        nose = frame[LANDMARK_INDEX["nose"]]
        left_ear = frame[LANDMARK_INDEX["left_ear"]]
        right_ear = frame[LANDMARK_INDEX["right_ear"]]
        left_shoulder = frame[LANDMARK_INDEX["left_shoulder"]]
        right_shoulder = frame[LANDMARK_INDEX["right_shoulder"]]

        if min(nose[2], left_ear[2], right_ear[2], left_shoulder[2], right_shoulder[2]) < args.conf_thresh:
            quality[index] = "low_conf"
            continue

        depth_raw = cv2.imread(str(depth_files[index]), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            quality[index] = "missing_depth"
            continue

        left_depth = sample_depth_m(depth_raw, float(left_ear[0]), float(left_ear[1]), args.depth_window, depth_scale, args.max_depth_m)
        right_depth = sample_depth_m(depth_raw, float(right_ear[0]), float(right_ear[1]), args.depth_window, depth_scale, args.max_depth_m)
        nose_depth = sample_depth_m(depth_raw, float(nose[0]), float(nose[1]), args.depth_window, depth_scale, args.max_depth_m)
        if left_depth is None or right_depth is None or nose_depth is None:
            quality[index] = "missing_depth_samples"
            continue

        ear_span_px = abs(float(right_ear[0] - left_ear[0]))
        if ear_span_px < 8.0:
            quality[index] = "small_ear_span"
            continue

        offset_norm = float(nose[0] - (left_ear[0] + right_ear[0]) * 0.5) / ear_span_px
        depth_diff = float(right_depth - left_depth)
        # Nose offset is the dominant cue. Ear depth difference adds sign support near side views.
        score = offset_norm + 8.0 * depth_diff

        raw_scores[index] = score
        nose_offset_norm[index] = offset_norm
        ear_depth_diff[index] = depth_diff
        quality[index] = "ok"

    smoothed_scores = smooth_series(raw_scores, args.smooth_window)
    baseline_pool = smoothed_scores[np.isfinite(smoothed_scores)][:15]
    if baseline_pool.size == 0:
        raise SystemExit("没有足够的有效帧用于估计转头状态")
    baseline = float(np.median(baseline_pool))
    relative_scores = smoothed_scores - baseline
    image_states = [
        classify_state(float(value), args.state_threshold) if np.isfinite(value) else "unknown"
        for value in relative_scores
    ]
    subject_states = [image_state_to_subject_state(state) for state in image_states]

    valid_mask = np.isfinite(relative_scores)
    valid_relative = relative_scores[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    image_left_peak_index = int(valid_indices[np.argmax(valid_relative)]) + 1
    image_right_peak_index = int(valid_indices[np.argmin(valid_relative)]) + 1
    neutral_frame_index = find_neutral_frame_index(relative_scores)
    relative_rom_score = float(np.nanmax(relative_scores) - np.nanmin(relative_scores))
    asymmetry_score = float(abs(np.nanmax(relative_scores) - abs(np.nanmin(relative_scores))))

    rows: list[dict[str, Any]] = []
    for index in range(kpt2d.shape[0]):
        rows.append(
            {
                "frame_index": index + 1,
                "raw_score": float(raw_scores[index]) if np.isfinite(raw_scores[index]) else float("nan"),
                "smoothed_score": float(smoothed_scores[index]) if np.isfinite(smoothed_scores[index]) else float("nan"),
                "relative_score": float(relative_scores[index]) if np.isfinite(relative_scores[index]) else float("nan"),
                "image_state": image_states[index],
                "subject_state": subject_states[index],
                "quality": quality[index],
                "nose_offset_norm": float(nose_offset_norm[index]) if np.isfinite(nose_offset_norm[index]) else float("nan"),
                "ear_depth_diff_m": float(ear_depth_diff[index]) if np.isfinite(ear_depth_diff[index]) else float("nan"),
            }
        )

    with (output_dir / "states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "raw_score",
                "smoothed_score",
                "relative_score",
                "image_state",
                "subject_state",
                "quality",
                "nose_offset_norm",
                "ear_depth_diff_m",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "baseline_score": baseline,
        "state_threshold": float(args.state_threshold),
        "neutral_frame_index": neutral_frame_index,
        "neutral_relative_score": float(relative_scores[neutral_frame_index - 1]) if neutral_frame_index is not None else None,
        "image_left_peak_frame_index": image_left_peak_index,
        "image_left_peak_score": float(np.nanmax(relative_scores)),
        "image_right_peak_frame_index": image_right_peak_index,
        "image_right_peak_score": float(abs(np.nanmin(relative_scores))),
        "subject_right_peak_frame_index": image_left_peak_index,
        "subject_right_peak_score": float(np.nanmax(relative_scores)),
        "subject_left_peak_frame_index": image_right_peak_index,
        "subject_left_peak_score": float(abs(np.nanmin(relative_scores))),
        "relative_rom_score": relative_rom_score,
        "asymmetry_score": asymmetry_score,
        "left_max_angle_deg": float(abs(np.nanmin(relative_scores))),
        "right_max_angle_deg": float(np.nanmax(relative_scores)),
        "total_rom_deg": relative_rom_score,
        "asymmetry_deg": asymmetry_score,
        "valid_frame_count": int(valid_mask.sum()),
        "state_counts": {
            "center": int(sum(1 for state in subject_states if state == "center")),
            "subject_left": int(sum(1 for state in subject_states if state == "subject_left")),
            "subject_right": int(sum(1 for state in subject_states if state == "subject_right")),
            "unknown": int(sum(1 for state in subject_states if state == "unknown")),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if vis_dir is not None:
        for index, rgb_path in enumerate(rgb_files):
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            annotated = annotate_frame(
                image=image,
                score=float(smoothed_scores[index]) if np.isfinite(smoothed_scores[index]) else None,
                relative_score=float(relative_scores[index]) if np.isfinite(relative_scores[index]) else None,
                image_state=image_states[index],
                subject_state=subject_states[index],
                quality=quality[index],
            )
            cv2.imwrite(str(vis_dir / f"frame_{index + 1:06d}.png"), annotated)

    key_frames_dir = output_dir / "key_frames"
    if neutral_frame_index is not None:
        neutral_idx = neutral_frame_index - 1
        export_key_frame(
            source_rgb=rgb_files[neutral_idx],
            destination=key_frames_dir / "neutral.png",
            score=float(smoothed_scores[neutral_idx]) if np.isfinite(smoothed_scores[neutral_idx]) else None,
            relative_score=float(relative_scores[neutral_idx]) if np.isfinite(relative_scores[neutral_idx]) else None,
            image_state=image_states[neutral_idx],
            subject_state=subject_states[neutral_idx],
            quality=quality[neutral_idx],
        )

    left_idx = image_right_peak_index - 1
    export_key_frame(
        source_rgb=rgb_files[left_idx],
        destination=key_frames_dir / "subject_left_peak.png",
        score=float(smoothed_scores[left_idx]) if np.isfinite(smoothed_scores[left_idx]) else None,
        relative_score=float(relative_scores[left_idx]) if np.isfinite(relative_scores[left_idx]) else None,
        image_state=image_states[left_idx],
        subject_state=subject_states[left_idx],
        quality=quality[left_idx],
    )

    right_idx = image_left_peak_index - 1
    export_key_frame(
        source_rgb=rgb_files[right_idx],
        destination=key_frames_dir / "subject_right_peak.png",
        score=float(smoothed_scores[right_idx]) if np.isfinite(smoothed_scores[right_idx]) else None,
        relative_score=float(relative_scores[right_idx]) if np.isfinite(relative_scores[right_idx]) else None,
        image_state=image_states[right_idx],
        subject_state=subject_states[right_idx],
        quality=quality[right_idx],
    )

    print(f"状态结果已保存: {output_dir / 'states.csv'}")
    print(f"汇总结果已保存: {output_dir / 'summary.json'}")
    if vis_dir is not None:
        print(f"可视化已保存: {vis_dir}")


def main() -> None:
    args = parse_args()

    if args.session_dir is not None and args.root_dir is not None:
        raise SystemExit("--session-dir 和 --root-dir 只能二选一")

    if args.session_dir is None and args.root_dir is None:
        session_dirs = [DEFAULT_SESSION_DIR]
    elif args.root_dir is not None:
        session_dirs = iter_head_turn_sessions(args.root_dir.expanduser().resolve())
    else:
        session_dirs = [args.session_dir.expanduser().resolve()]

    failures: list[tuple[Path, str]] = []
    for index, session_dir in enumerate(session_dirs, start=1):
        print(f"[{index}/{len(session_dirs)}] 处理: {session_dir}")
        try:
            process_session(session_dir, args)
        except KeyboardInterrupt:
            raise
        except SystemExit as exc:
            failures.append((session_dir, str(exc)))
            print(f"处理失败: {session_dir}")
            print(f"原因: {exc}")
        except Exception as exc:
            failures.append((session_dir, str(exc)))
            print(f"处理失败: {session_dir}")
            print(f"原因: {exc}")

    if failures:
        print("\n以下 session 处理失败:")
        for session_dir, message in failures:
            print(f"- {session_dir}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
