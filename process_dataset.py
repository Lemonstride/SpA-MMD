from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_RAW_ROOT = Path("/your/own/path/raw")
DEFAULT_OUTPUT_ROOT = Path("/your/own/path/processed")
DEFAULT_SESSION_MAP = {"session1": "walk", "session2": "head_turn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw subject folders into the processed training layout."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory containing subject folders such as S01/S02.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the processed/ tree will be created.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs to process, for example: S01 S02",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for exported RGB/depth frames per session.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing seq001 folder if it already exists.",
    )
    return parser.parse_args()


def load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import pyrealsense2 as rs  # type: ignore
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown package"
        raise SystemExit(
            "缺少依赖包: "
            f"{missing}\n"
            "请先在对应环境安装:\n"
            "python -m pip install -r requirements.txt"
        ) from exc

    return rs, np, cv2


def subject_dirs(raw_root: Path, subjects: list[str] | None) -> list[Path]:
    candidates = [path for path in sorted(raw_root.iterdir()) if path.is_dir() and path.name.startswith("S")]
    if subjects is None:
        return candidates
    allowed = {item.upper() for item in subjects}
    return [path for path in candidates if path.name.upper() in allowed]


def session_dirs(subject_dir: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for path in sorted(subject_dir.iterdir()):
        if not path.is_dir():
            continue
        session_type = DEFAULT_SESSION_MAP.get(path.name)
        if session_type is None:
            continue
        pairs.append((path, session_type))
    return pairs


def find_first(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def prepare_dirs(seq_dir: Path, overwrite: bool) -> dict[str, Path]:
    if seq_dir.exists() and overwrite:
        shutil.rmtree(seq_dir)
    elif seq_dir.exists():
        raise FileExistsError(f"输出目录已存在: {seq_dir}")

    paths = {
        "seq": seq_dir,
        "rgb": seq_dir / "rgb",
        "depth": seq_dir / "depth",
        "imu": seq_dir / "imu",
        "mmwave_raw": seq_dir / "mmwave" / "raw",
        "mmwave_rdmap": seq_dir / "mmwave" / "rdmap",
        "mmwave_pointcloud": seq_dir / "mmwave" / "pointcloud",
        "labels": seq_dir / "labels",
        "labels_kpt2d": seq_dir / "labels" / "kpt2d",
        "labels_kpt3d": seq_dir / "labels" / "kpt3d",
        "labels_skeleton_map": seq_dir / "labels" / "skeleton_map",
        "calib": seq_dir / "calib",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_frame_profile(frame: Any) -> Any:
    return frame.profile if hasattr(frame, "profile") else frame.get_profile()


def intrinsics_to_dict(video_profile: Any) -> dict[str, Any]:
    intr = video_profile.get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def extrinsics_to_dict(extrinsics: Any) -> dict[str, Any]:
    return {
        "rotation": list(extrinsics.rotation),
        "translation": list(extrinsics.translation),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def motion_rows_from_frames(rs: Any, frames: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        iterable = list(frames)
    except TypeError:
        iterable = []

    for frame in iterable:
        if not frame or not frame.is_motion_frame():
            continue

        motion = frame.as_motion_frame().get_motion_data()
        profile = get_frame_profile(frame)
        stream_type = profile.stream_type()

        if stream_type == rs.stream.accel:
            sensor_name = "accel"
        elif stream_type == rs.stream.gyro:
            sensor_name = "gyro"
        else:
            sensor_name = str(stream_type)

        rows.append(
            {
                "sensor": sensor_name,
                "frame_number": int(frame.frame_number),
                "timestamp_ms": float(frame.timestamp),
                "x": float(motion.x),
                "y": float(motion.y),
                "z": float(motion.z),
            }
        )
    return rows


def paired_timestamp_rows(
    rgb_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    size = max(len(rgb_rows), len(depth_rows))
    rows: list[dict[str, Any]] = []
    for index in range(size):
        rgb = rgb_rows[index] if index < len(rgb_rows) else {}
        depth = depth_rows[index] if index < len(depth_rows) else {}
        rows.append(
            {
                "index": index + 1,
                "rgb_frame_number": rgb.get("frame_number", ""),
                "rgb_timestamp_ms": rgb.get("timestamp_ms", ""),
                "rgb_file": rgb.get("file_name", ""),
                "depth_frame_number": depth.get("frame_number", ""),
                "depth_timestamp_ms": depth.get("timestamp_ms", ""),
                "depth_file": depth.get("file_name", ""),
            }
        )
    return rows


def initialize_label_files(labels_dir: Path) -> None:
    (labels_dir / "binary_label.txt").write_text("unknown\n", encoding="utf-8")
    (labels_dir / "severity_label.txt").write_text("unknown\n", encoding="utf-8")
    write_csv(labels_dir / "gait_phase.csv", [], ["frame_index", "phase"])
    write_json(
        labels_dir / "disease_annotations.json",
        {
            "binary_label": None,
            "severity_label": None,
            "notes": "",
        },
    )


def initialize_mmwave_placeholders(mmwave_rdmap: Path, mmwave_pointcloud: Path) -> None:
    (mmwave_rdmap / ".keep").write_text("", encoding="utf-8")
    (mmwave_pointcloud / ".keep").write_text("", encoding="utf-8")


def initialize_calib_placeholders(calib_dir: Path) -> None:
    write_json(
        calib_dir / "extrinsics_d455_to_mmwave.json",
        {
            "status": "unknown",
            "message": "Fill this file after completing camera-radar calibration.",
        },
    )


def copy_mmwave_files(session_dir: Path, mmwave_raw_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    pattern_map = {"*.bin": "radar_bin", "*.cfg": "radar_cfg"}
    for pattern, key in pattern_map.items():
        source = find_first(session_dir, pattern)
        if source is None:
            continue
        destination = mmwave_raw_dir / source.name
        shutil.copy2(source, destination)
        copied[key] = str(destination)
    return copied


def export_bag(
    rs: Any,
    np: Any,
    cv2: Any,
    bag_path: Path,
    dirs: dict[str, Path],
    max_frames: int | None,
) -> dict[str, Any]:
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align_to_color = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()

    rgb_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    imu_rows: list[dict[str, Any]] = []
    color_intrinsics: dict[str, Any] | None = None
    depth_intrinsics: dict[str, Any] | None = None
    depth_to_color_extrinsics: dict[str, Any] | None = None

    rgb_index = 0
    depth_index = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(1000)
            except RuntimeError:
                break

            imu_rows.extend(motion_rows_from_frames(rs, frames))
            aligned_frames = align_to_color.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if color_frame:
                rgb_index += 1
                rgb_path = dirs["rgb"] / f"frame_{rgb_index:06d}.png"
                rgb_image = np.asanyarray(color_frame.get_data())
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                rgb_rows.append(
                    {
                        "index": rgb_index,
                        "file_name": rgb_path.name,
                        "frame_number": int(color_frame.frame_number),
                        "timestamp_ms": float(color_frame.timestamp),
                    }
                )
                if color_intrinsics is None:
                    color_profile = get_frame_profile(color_frame).as_video_stream_profile()
                    color_intrinsics = intrinsics_to_dict(color_profile)

            if depth_frame:
                depth_index += 1
                depth_path = dirs["depth"] / f"frame_{depth_index:06d}.png"
                depth_image = np.asanyarray(depth_frame.get_data())
                cv2.imwrite(str(depth_path), depth_image)
                depth_rows.append(
                    {
                        "index": depth_index,
                        "file_name": depth_path.name,
                        "frame_number": int(depth_frame.frame_number),
                        "timestamp_ms": float(depth_frame.timestamp),
                    }
                )
                if depth_intrinsics is None:
                    depth_profile = get_frame_profile(depth_frame).as_video_stream_profile()
                    color_profile = get_frame_profile(color_frame).as_video_stream_profile() if color_frame else None
                    depth_intrinsics = intrinsics_to_dict(depth_profile)
                    if color_profile is not None:
                        depth_to_color_extrinsics = extrinsics_to_dict(
                            depth_profile.get_extrinsics_to(color_profile)
                        )

            if max_frames is not None and rgb_index >= max_frames:
                break
    finally:
        pipeline.stop()

    write_csv(
        dirs["imu"] / "imu.csv",
        imu_rows,
        ["sensor", "frame_number", "timestamp_ms", "x", "y", "z"],
    )
    write_csv(
        dirs["seq"] / "timestamps.csv",
        paired_timestamp_rows(rgb_rows, depth_rows),
        [
            "index",
            "rgb_frame_number",
            "rgb_timestamp_ms",
            "rgb_file",
            "depth_frame_number",
            "depth_timestamp_ms",
            "depth_file",
        ],
    )

    if color_intrinsics is not None:
        write_json(dirs["calib"] / "intrinsics_d455_color.json", color_intrinsics)
    if depth_intrinsics is not None:
        write_json(dirs["calib"] / "intrinsics_d455_depth.json", depth_intrinsics)
    if depth_to_color_extrinsics is not None:
        write_json(dirs["calib"] / "extrinsics_d455_depth_to_color.json", depth_to_color_extrinsics)
    (dirs["calib"] / "depth_scale.txt").write_text(f"{depth_sensor.get_depth_scale()}\n", encoding="utf-8")

    return {
        "rgb_frames": len(rgb_rows),
        "depth_frames": len(depth_rows),
        "imu_rows": len(imu_rows),
        "color_intrinsics": color_intrinsics,
        "depth_intrinsics": depth_intrinsics,
    }


def process_session(
    rs: Any,
    np: Any,
    cv2: Any,
    subject_id: str,
    session_dir: Path,
    session_type: str,
    output_root: Path,
    max_frames: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    bag_path = find_first(session_dir, "*.bag")
    if bag_path is None:
        raise FileNotFoundError(f"未找到 bag 文件: {session_dir}")

    session_output_dir = output_root / subject_id / session_type
    dirs = prepare_dirs(session_output_dir, overwrite)
    initialize_label_files(dirs["labels"])
    initialize_mmwave_placeholders(dirs["mmwave_rdmap"], dirs["mmwave_pointcloud"])
    initialize_calib_placeholders(dirs["calib"])
    mmwave_paths = copy_mmwave_files(session_dir, dirs["mmwave_raw"])
    export_summary = export_bag(rs, np, cv2, bag_path, dirs, max_frames)

    meta = {
        "subject_id": subject_id,
        "session_name": session_dir.name,
        "session_type": session_type,
        "source_bag": str(bag_path),
        "source_session_dir": str(session_dir),
        "source_mmwave": mmwave_paths,
        "summary": export_summary,
    }
    write_json(dirs["seq"] / "meta.json", meta)
    write_json(
        dirs["seq"] / "session_meta.json",
        {
            "subject_id": subject_id,
            "session_name": session_dir.name,
            "session_type": session_type,
        },
    )
    return meta


def main() -> None:
    args = parse_args()
    rs, np, cv2 = load_dependencies()

    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise SystemExit(f"找不到原始数据根目录: {raw_root}")

    subjects = subject_dirs(raw_root, args.subjects)
    if not subjects:
        raise SystemExit("没有找到可处理的受试者目录。")

    summaries: list[dict[str, Any]] = []
    for subject_dir in subjects:
        subject_id = subject_dir.name
        sessions = session_dirs(subject_dir)
        if not sessions:
            print(f"跳过 {subject_id}: 没有找到 session1/session2")
            continue

        for session_dir, session_type in sessions:
            print(f"处理 {subject_id} {session_dir.name} -> {session_type}")
            summary = process_session(
                rs=rs,
                np=np,
                cv2=cv2,
                subject_id=subject_id,
                session_dir=session_dir,
                session_type=session_type,
                output_root=output_root,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
            summaries.append(summary)

    write_json(output_root / "processed_summary.json", {"sessions": summaries})
    print(f"处理完成，输出目录: {output_root}")


if __name__ == "__main__":
    main()
