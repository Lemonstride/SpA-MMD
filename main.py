from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_BAG_PATH = Path("/Users/ice/Desktop/S01/session1/20251215_094856.bag")
DEFAULT_OUTPUT_DIR = Path("/Users/ice/Documents/SpA-MMD/output/session1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RGB, depth and motion data from a RealSense .bag file."
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=DEFAULT_BAG_PATH,
        help="Path to the RealSense .bag file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to store extracted files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for the number of RGB/depth frames to export.",
    )
    parser.add_argument(
        "--save-depth-preview",
        action="store_true",
        help="Also save a colorized preview for each depth image.",
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
            "请先在 PyCharm 终端安装:\n"
            "pip install pyrealsense2 numpy opencv-python"
        ) from exc

    return rs, np, cv2


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "rgb": output_dir / "rgb",
        "depth": output_dir / "depth",
        "depth_preview": output_dir / "depth_preview",
        "meta": output_dir / "meta",
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


def frame_record(index: int, frame: Any, relative_path: str) -> dict[str, Any]:
    return {
        "index": index,
        "frame_number": int(frame.frame_number),
        "timestamp_ms": float(frame.timestamp),
        "relative_path": relative_path,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    rs, np, cv2 = load_dependencies()

    bag_path = args.bag.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not bag_path.exists():
        raise SystemExit(f"找不到 bag 文件: {bag_path}")

    paths = ensure_dirs(output_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align_to_color = rs.align(rs.stream.color)

    rgb_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "bag_path": str(bag_path),
        "output_dir": str(output_dir),
        "streams": {},
    }

    rgb_index = 0
    depth_index = 0
    saved_rgb_intrinsics = False
    saved_depth_intrinsics = False

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(1000)
            except RuntimeError:
                break

            motion_rows.extend(motion_rows_from_frames(rs, frames))
            aligned_frames = align_to_color.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if color_frame:
                rgb_index += 1
                rgb_image = np.asanyarray(color_frame.get_data())
                rgb_path = paths["rgb"] / f"rgb_{rgb_index:06d}.png"
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                rgb_rows.append(frame_record(rgb_index, color_frame, rgb_path.name))

                if not saved_rgb_intrinsics:
                    video_profile = get_frame_profile(color_frame).as_video_stream_profile()
                    metadata["streams"]["color"] = intrinsics_to_dict(video_profile)
                    saved_rgb_intrinsics = True

            if depth_frame:
                depth_index += 1
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_path = paths["depth"] / f"depth_{depth_index:06d}.png"
                cv2.imwrite(str(depth_path), depth_image)
                depth_rows.append(frame_record(depth_index, depth_frame, depth_path.name))

                if args.save_depth_preview:
                    preview = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET,
                    )
                    preview_path = paths["depth_preview"] / f"depth_{depth_index:06d}.png"
                    cv2.imwrite(str(preview_path), preview)

                if not saved_depth_intrinsics:
                    video_profile = get_frame_profile(depth_frame).as_video_stream_profile()
                    metadata["streams"]["depth"] = intrinsics_to_dict(video_profile)
                    saved_depth_intrinsics = True

            if args.max_frames is not None and rgb_index >= args.max_frames:
                break

    finally:
        pipeline.stop()

    write_csv(
        paths["meta"] / "rgb_frames.csv",
        rgb_rows,
        ["index", "frame_number", "timestamp_ms", "relative_path"],
    )
    write_csv(
        paths["meta"] / "depth_frames.csv",
        depth_rows,
        ["index", "frame_number", "timestamp_ms", "relative_path"],
    )
    write_csv(
        paths["meta"] / "motion.csv",
        motion_rows,
        ["sensor", "frame_number", "timestamp_ms", "x", "y", "z"],
    )

    metadata["summary"] = {
        "rgb_frames": len(rgb_rows),
        "depth_frames": len(depth_rows),
        "motion_rows": len(motion_rows),
    }
    save_json(paths["meta"] / "metadata.json", metadata)

    print("处理完成")
    print(f"bag 文件: {bag_path}")
    print(f"输出目录: {output_dir}")
    print(f"RGB 帧数量: {len(rgb_rows)}")
    print(f"深度帧数量: {len(depth_rows)}")
    print(f"动作/IMU 行数: {len(motion_rows)}")


if __name__ == "__main__":
    main()
