from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate windowed session segments for SpA-MMD without leakage."
    )
    parser.add_argument(
        "--processed_root",
        type=Path,
        required=True,
        help="Path to SpA-MMD processed root (contains SXX folders).",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="walk,head_turn",
        help="Comma-separated sessions to include.",
    )
    parser.add_argument(
        "--window",
        type=int,
        required=True,
        help="Window length in frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        required=True,
        help="Stride in frames.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output CSV file with window segments.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=16,
        help="Minimum frames required to keep a session.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a session is missing rgb/frames.",
    )
    return parser.parse_args()


def count_frames(rgb_dir: Path) -> int:
    if not rgb_dir.exists():
        return 0
    return len([p for p in rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])


def main() -> None:
    args = parse_args()
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    rows: list[dict[str, object]] = []

    for subject_dir in sorted([p for p in args.processed_root.iterdir() if p.is_dir()]):
        subject_id = subject_dir.name
        for session in sessions:
            session_dir = subject_dir / session
            if not session_dir.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing session: {session_dir}")
                continue
            rgb_dir = session_dir / "rgb"
            total_frames = count_frames(rgb_dir)
            if total_frames < max(args.min_frames, args.window):
                continue

            start = 0
            window_id = 0
            while start + args.window <= total_frames:
                rows.append(
                    {
                        "subject_id": subject_id,
                        "session": session,
                        "window_id": window_id,
                        "start_frame": start,
                        "end_frame": start + args.window - 1,
                        "total_frames": total_frames,
                    }
                )
                window_id += 1
                start += args.stride

    if not rows:
        raise SystemExit("No window segments generated.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} windows to {args.output_csv}")


if __name__ == "__main__":
    main()
