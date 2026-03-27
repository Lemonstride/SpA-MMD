from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOT_DIR = Path("/your/own/path/processed")
SESSION_NAMES = ("walk", "head_turn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move pose/skeleton outputs from labels/ into session-level skeleton/ folders."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Processed dataset root, for example /your/own/path/processed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files/directories inside skeleton/.",
    )
    return parser.parse_args()


def iter_session_dirs(root_dir: Path) -> list[Path]:
    session_dirs: list[Path] = []
    for subject_dir in sorted(root_dir.glob("S*")):
        if not subject_dir.is_dir():
            continue
        for session_name in SESSION_NAMES:
            session_dir = subject_dir / session_name
            if session_dir.is_dir() and (session_dir / "labels").is_dir():
                session_dirs.append(session_dir)
    if not session_dirs:
        raise SystemExit(f"在 {root_dir} 下没有找到可处理的 session")
    return session_dirs


def safe_move(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> None:
    if not src.exists():
        return

    if dst.exists():
        if not overwrite:
            raise SystemExit(f"目标已存在，请使用 --overwrite: {dst}")
        if not dry_run:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

    print(f"MOVE {src} -> {dst}")
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def move_if_present(
    labels_dir: Path,
    skeleton_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> int:
    moves = 0

    mapping = [
        (labels_dir / "kpt2d", skeleton_dir / "kpt2d"),
        (labels_dir / "kpt2d.npy", skeleton_dir / "kpt2d" / "kpt2d.npy"),
        (labels_dir / "kpt3d", skeleton_dir / "kpt3d"),
        (labels_dir / "kpt3d.npy", skeleton_dir / "kpt3d" / "kpt3d.npy"),
        (labels_dir / "kpt3d.csv", skeleton_dir / "kpt3d" / "kpt3d.csv"),
        (labels_dir / "pose_meta.json", skeleton_dir / "kpt3d" / "pose_meta.json"),
        (labels_dir / "pose_vis", skeleton_dir / "kpt3d" / "pose_vis"),
        (labels_dir / "skeleton_map", skeleton_dir / "skeleton_map"),
    ]

    for src, dst in mapping:
        if src.exists():
            safe_move(src, dst, overwrite=overwrite, dry_run=dry_run)
            moves += 1

    return moves


def remove_empty_dir(path: Path, dry_run: bool) -> None:
    if not path.exists() or not path.is_dir():
        return
    if any(path.iterdir()):
        return
    print(f"RMDIR {path}")
    if not dry_run:
        path.rmdir()


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    session_dirs = iter_session_dirs(root_dir)

    total_moves = 0
    touched_sessions = 0

    for session_dir in session_dirs:
        labels_dir = session_dir / "labels"
        skeleton_dir = session_dir / "skeleton"
        print(f"\n[{session_dir}]")
        moves = move_if_present(
            labels_dir=labels_dir,
            skeleton_dir=skeleton_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if moves > 0:
            touched_sessions += 1
            total_moves += moves
            if not args.dry_run:
                for subdir in (labels_dir / "kpt2d", labels_dir / "kpt3d", labels_dir / "pose_vis"):
                    remove_empty_dir(subdir, dry_run=args.dry_run)
                remove_empty_dir(labels_dir, dry_run=args.dry_run)
        else:
            print("没有发现需要迁移的骨架相关文件。")

    print(f"\n完成。处理 session 数: {touched_sessions}, 迁移项数: {total_moves}")


if __name__ == "__main__":
    main()
