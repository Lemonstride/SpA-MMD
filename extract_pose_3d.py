from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SESSION_DIR = Path("/your/own/path/processed/S01/walk")
DEFAULT_ROOT_DIR = Path("/your/own/path/processed")
DEFAULT_MODEL_ASSET = Path("/your/own/path/models/pose_landmarker_full.task")
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 2D and 3D body pose from RGB/depth sequences."
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Processed session directory containing rgb/, depth/, calib/ and labels/.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Processed dataset root. If provided, batch-process all S*/walk and S*/head_turn sessions under this directory.",
    )
    parser.add_argument(
        "--model-asset",
        type=Path,
        default=DEFAULT_MODEL_ASSET,
        help="Path to the MediaPipe Pose Landmarker .task model file. Used by newer MediaPipe versions.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="MediaPipe model complexity. 2 is the highest quality.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for MediaPipe Pose.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence for MediaPipe Pose.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.35,
        help="Minimum landmark confidence used to keep a point.",
    )
    parser.add_argument(
        "--depth-window",
        type=int,
        default=7,
        help="Odd window size for depth sampling around each landmark.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=6.0,
        help="Ignore sampled depth larger than this value in meters.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window used after interpolation.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save 2D overlay visualizations to labels/pose_vis.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing kpt2d.npy/kpt3d.npy and visualizations.",
    )
    return parser.parse_args()


def load_mediapipe() -> Any:
    try:
        import mediapipe as mp  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少 mediapipe。请先安装后再运行:\n"
            "python -m pip install mediapipe"
        ) from exc
    return mp


class LegacyPoseBackend:
    def __init__(self, mp: Any, args: argparse.Namespace) -> None:
        self.backend_name = "mediapipe_solutions_pose"
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[Any]:
        result = self._pose.process(rgb)
        return [] if result.pose_landmarks is None else list(result.pose_landmarks.landmark)

    def close(self) -> None:
        self._pose.close()


class TasksPoseBackend:
    def __init__(self, mp: Any, args: argparse.Namespace) -> None:
        self.backend_name = "mediapipe_tasks_pose_landmarker"
        model_asset = args.model_asset.expanduser().resolve()
        if not model_asset.exists():
            raise SystemExit(
                "当前 mediapipe 版本使用 Tasks API，需要提供 Pose Landmarker 模型文件。\n"
                f"未找到模型: {model_asset}\n"
                "请下载 pose_landmarker_full.task 后通过 --model-asset 指定。"
            )

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_asset)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=args.min_detection_confidence,
            min_pose_presence_confidence=args.conf_thresh,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        self._mp = mp
        self._landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[Any]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return []
        return list(result.pose_landmarks[0])

    def close(self) -> None:
        self._landmarker.close()


def create_pose_backend(mp: Any, args: argparse.Namespace) -> Any:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return LegacyPoseBackend(mp, args)
    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        return TasksPoseBackend(mp, args)
    raise SystemExit(
        "当前 mediapipe 版本既不支持旧版 mp.solutions.pose，也无法找到 Tasks API。"
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_outputs(labels_dir: Path, save_vis: bool, overwrite: bool) -> Path | None:
    targets = [labels_dir / "kpt2d.npy", labels_dir / "kpt3d.npy", labels_dir / "kpt3d.csv", labels_dir / "pose_meta.json"]
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise SystemExit(f"输出文件已存在，请使用 --overwrite 覆盖: {existing[0]}")

    vis_dir = None
    if save_vis:
        vis_dir = labels_dir / "pose_vis"
        if vis_dir.exists() and overwrite:
            for file_path in vis_dir.glob("*.png"):
                file_path.unlink()
        vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir


def load_intrinsics(calib_dir: Path) -> dict[str, float]:
    intrinsics = read_json(calib_dir / "intrinsics_d455_depth.json")
    return {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["ppx"]),
        "cy": float(intrinsics["ppy"]),
    }


def load_depth_scale(calib_dir: Path) -> float:
    text = (calib_dir / "depth_scale.txt").read_text(encoding="utf-8").strip()
    return float(text)


def sorted_images(path: Path) -> list[Path]:
    files = sorted(path.glob("*.png"))
    if not files:
        raise SystemExit(f"目录为空: {path}")
    return files


def landmark_confidence(landmark: Any) -> float:
    visibility = float(getattr(landmark, "visibility", 1.0))
    presence = float(getattr(landmark, "presence", 1.0))
    return max(0.0, min(1.0, visibility * presence))


def sample_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    window: int,
    max_depth_m: float,
) -> float:
    x = int(round(u))
    y = int(round(v))
    radius = window // 2
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(depth_m.shape[1], x + radius + 1)
    y2 = min(depth_m.shape[0], y + radius + 1)
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[(patch > 0) & (patch <= max_depth_m)]
    if valid.size == 0:
        return float("nan")
    lo = np.percentile(valid, 20)
    hi = np.percentile(valid, 80)
    robust = valid[(valid >= lo) & (valid <= hi)]
    if robust.size == 0:
        robust = valid
    return float(np.median(robust))


def deproject(u: float, v: float, z: float, intr: dict[str, float]) -> tuple[float, float, float]:
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return x, y, z


def interpolate_series(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    result = values.copy()
    indices = np.arange(values.shape[0])
    for joint in range(values.shape[1]):
        for channel in range(values.shape[2]):
            valid = valid_mask[:, joint]
            if valid.sum() == 0:
                continue
            if valid.sum() == 1:
                result[:, joint, channel] = values[valid, joint, channel][0]
                continue
            result[:, joint, channel] = np.interp(
                indices,
                indices[valid],
                values[valid, joint, channel],
            )
    return result


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    window = max(1, window)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    smoothed = values.copy()
    for t in range(values.shape[0]):
        start = max(0, t - radius)
        end = min(values.shape[0], t + radius + 1)
        smoothed[t] = values[start:end].mean(axis=0)
    return smoothed


def smooth_conf(conf: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return conf
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out = conf.copy()
    for t in range(conf.shape[0]):
        start = max(0, t - radius)
        end = min(conf.shape[0], t + radius + 1)
        out[t] = conf[start:end].mean(axis=0)
    return out


def draw_pose_overlay(image: np.ndarray, coords: np.ndarray, conf: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    edges = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    ]
    for start, end in edges:
        if conf[start] <= 0 or conf[end] <= 0:
            continue
        p1 = tuple(int(round(v)) for v in coords[start])
        p2 = tuple(int(round(v)) for v in coords[end])
        cv2.line(canvas, p1, p2, (0, 255, 0), 2)
    for idx in range(coords.shape[0]):
        if conf[idx] <= 0:
            continue
        point = tuple(int(round(v)) for v in coords[idx])
        cv2.circle(canvas, point, 3, (0, 180, 255), -1)
    return canvas


def write_kpt3d_csv(path: Path, coords3d: np.ndarray, conf: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "joint_index", "joint_name", "x_m", "y_m", "z_m", "confidence"])
        for frame_index in range(coords3d.shape[0]):
            for joint_index in range(coords3d.shape[1]):
                x, y, z = coords3d[frame_index, joint_index]
                writer.writerow(
                    [
                        frame_index + 1,
                        joint_index,
                        LANDMARK_NAMES[joint_index],
                        float(x),
                        float(y),
                        float(z),
                        float(conf[frame_index, joint_index]),
                    ]
                )


def iter_session_dirs(root_dir: Path) -> list[Path]:
    session_dirs: list[Path] = []
    for subject_dir in sorted(root_dir.glob("S*")):
        if not subject_dir.is_dir():
            continue
        for session_name in ("walk", "head_turn"):
            session_dir = subject_dir / session_name
            if not session_dir.is_dir():
                continue
            if all((session_dir / name).exists() for name in ("rgb", "depth", "calib", "labels")):
                session_dirs.append(session_dir)
    if not session_dirs:
        raise SystemExit(f"在 {root_dir} 下没有找到可处理的 session")
    return session_dirs


def process_session(session_dir: Path, args: argparse.Namespace, mp: Any) -> None:
    labels_dir = session_dir / "labels"
    calib_dir = session_dir / "calib"
    rgb_dir = session_dir / "rgb"
    depth_dir = session_dir / "depth"

    if args.depth_window % 2 == 0:
        raise SystemExit("--depth-window 必须是奇数")

    vis_dir = ensure_outputs(labels_dir, args.save_vis, args.overwrite)
    mp = load_mediapipe()
    intr = load_intrinsics(calib_dir)
    depth_scale = load_depth_scale(calib_dir)

    rgb_files = sorted_images(rgb_dir)
    depth_files = sorted_images(depth_dir)
    if len(rgb_files) != len(depth_files):
        raise SystemExit("rgb 和 depth 帧数不一致")

    pose_backend = create_pose_backend(mp, args)

    num_frames = len(rgb_files)
    num_joints = len(LANDMARK_NAMES)
    kpt2d = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    kpt3d = np.full((num_frames, num_joints, 4), np.nan, dtype=np.float32)
    valid_3d = np.zeros((num_frames, num_joints), dtype=bool)

    try:
        for frame_index, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth_raw is None:
                raise SystemExit(f"读取失败: {rgb_path} 或 {depth_path}")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            landmarks = pose_backend.detect(rgb, frame_index * 33)
            if not landmarks:
                continue

            depth_m = depth_raw.astype(np.float32) * depth_scale
            height, width = bgr.shape[:2]
            frame_coords = np.zeros((num_joints, 2), dtype=np.float32)
            frame_conf = np.zeros((num_joints,), dtype=np.float32)

            for joint_index, landmark in enumerate(landmarks):
                u = float(np.clip(landmark.x * width, 0, width - 1))
                v = float(np.clip(landmark.y * height, 0, height - 1))
                conf = landmark_confidence(landmark)
                frame_coords[joint_index] = (u, v)
                frame_conf[joint_index] = conf
                kpt2d[frame_index, joint_index] = [u, v, conf]

                if conf < args.conf_thresh:
                    continue
                z = sample_depth(depth_m, u, v, args.depth_window, args.max_depth_m)
                if np.isnan(z):
                    continue
                x, y, z = deproject(u, v, z, intr)
                kpt3d[frame_index, joint_index] = [x, y, z, conf]
                valid_3d[frame_index, joint_index] = True

            if vis_dir is not None:
                overlay = draw_pose_overlay(bgr, frame_coords, np.where(frame_conf >= args.conf_thresh, frame_conf, 0.0))
                cv2.imwrite(str(vis_dir / f"frame_{frame_index + 1:06d}.png"), overlay)
    finally:
        pose_backend.close()

    # 2D smoothing uses confidence as availability mask.
    conf2d = kpt2d[..., 2]
    valid_2d = conf2d >= args.conf_thresh
    kpt2d_xy = interpolate_series(kpt2d[..., :2], valid_2d)
    kpt2d_xy = smooth_series(kpt2d_xy, args.smooth_window)
    conf2d = smooth_conf(conf2d, args.smooth_window)
    kpt2d = np.concatenate([kpt2d_xy, conf2d[..., None]], axis=-1).astype(np.float32)

    xyz = np.where(np.isnan(kpt3d[..., :3]), 0.0, kpt3d[..., :3])
    xyz = interpolate_series(xyz, valid_3d)
    xyz = smooth_series(xyz, args.smooth_window)
    conf3d = np.where(np.isnan(kpt3d[..., 3]), 0.0, kpt3d[..., 3])
    conf3d = smooth_conf(conf3d, args.smooth_window)
    kpt3d = np.concatenate([xyz, conf3d[..., None]], axis=-1).astype(np.float32)

    np.save(labels_dir / "kpt2d.npy", kpt2d)
    np.save(labels_dir / "kpt3d.npy", kpt3d)
    write_kpt3d_csv(labels_dir / "kpt3d.csv", kpt3d[..., :3], kpt3d[..., 3])

    pose_meta = {
        "backend": pose_backend.backend_name,
        "num_frames": num_frames,
        "num_joints": num_joints,
        "joint_names": LANDMARK_NAMES,
        "units_3d": "meters",
        "model_complexity": args.model_complexity,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "conf_thresh": args.conf_thresh,
        "depth_window": args.depth_window,
        "max_depth_m": args.max_depth_m,
        "smooth_window": args.smooth_window,
        "model_asset": str(args.model_asset.expanduser().resolve()),
        "session_dir": str(session_dir),
    }
    with (labels_dir / "pose_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(pose_meta, handle, indent=2, ensure_ascii=False)

    print(f"2D 关键点已保存: {labels_dir / 'kpt2d.npy'}")
    print(f"3D 关键点已保存: {labels_dir / 'kpt3d.npy'}")
    print(f"3D CSV 已保存: {labels_dir / 'kpt3d.csv'}")
    if vis_dir is not None:
        print(f"可视化已保存: {vis_dir}")


def main() -> None:
    args = parse_args()

    if args.session_dir is not None and args.root_dir is not None:
        raise SystemExit("--session-dir 和 --root-dir 只能二选一")

    if args.session_dir is None and args.root_dir is None:
        session_dirs = [DEFAULT_SESSION_DIR]
    elif args.root_dir is not None:
        root_dir = args.root_dir.expanduser().resolve()
        session_dirs = iter_session_dirs(root_dir)
    else:
        session_dirs = [args.session_dir.expanduser().resolve()]

    if args.depth_window % 2 == 0:
        raise SystemExit("--depth-window 必须是奇数")

    mp = load_mediapipe()
    failures: list[tuple[Path, str]] = []

    for index, session_dir in enumerate(session_dirs, start=1):
        print(f"[{index}/{len(session_dirs)}] 处理: {session_dir}")
        try:
            process_session(session_dir, args, mp)
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
