# SpA-MMD

SpA-MMD is a multi-modal dataset project for Spondyloarthritis (SpA) assessment, with a focus on gait analysis, cervical mobility analysis, and multi-modal representation learning.

This repository contains:

- dataset processing scripts
- dataset format specification
- skeleton extraction and conversion tools
- project documentation

This repository does **not** contain the dataset itself.

## Closed-Source Notice

The **SpA-MMD dataset is closed-source**.

The dataset is not publicly released because it contains patient-related clinical and motion data. In addition, data sharing is restricted by:

- patient privacy protection requirements
- hospital and institutional data management requirements
- clinical data usage constraints

This repository is therefore limited to code, format description, and processing workflow documentation.

## Dataset Design

Each subject contains two recording sessions:

- `walk`: gait recording session
- `head_turn`: cervical rotation / head-turn recording session

The dataset currently includes the following modalities:

- `rgb`
- `depth`
- `imu`
- `mmwave`
- `skeleton`

## Processed Dataset Structure

The processed dataset is organized by subject and session.

```text
SpA-MMD/
└── processed/
    ├── S01/
    │   ├── walk/
    │   └── head_turn/
    ├── S02/
    │   ├── walk/
    │   └── head_turn/
    └── ...
```

- `SXX` denotes one subject.
- Each subject has two session folders:
  - `walk`
  - `head_turn`

## Session Directory Format

Each session directory uses the following format:

```text
walk/
├── calib/
├── depth/
├── imu/
├── labels/
├── mmwave/
├── rgb/
├── skeleton/
├── meta.json
├── session_meta.json
└── timestamps.csv
```

The same structure is used for `head_turn/`.

## Folder Specification

### `rgb/`

RGB image sequence exported from the RealSense D455 color stream.

Example:

```text
rgb/
├── frame_000001.png
├── frame_000002.png
└── ...
```

### `depth/`

Aligned depth image sequence exported from the RealSense D455 depth stream.

Example:

```text
depth/
├── frame_000001.png
├── frame_000002.png
└── ...
```

Notes:

- 16-bit PNG
- aligned with RGB frames
- used for depth-aware processing and 3D keypoint reconstruction

### `imu/`

IMU data exported from D455.

Example:

```text
imu/
└── imu.csv
```

### `mmwave/`

Millimeter-wave radar data directory.

```text
mmwave/
├── raw/
├── rdmap/
└── pointcloud/
```

- `raw/`: raw radar files such as `.bin` and radar config files
- `rdmap/`: reserved for range-Doppler map outputs
- `pointcloud/`: reserved for radar point cloud outputs

### `labels/`

Session-level labels and annotations.

```text
labels/
├── binary_label.txt
├── severity_label.txt
├── gait_phase.csv
└── disease_annotations.json
```

- `binary_label.txt`: binary label, for example healthy vs. SpA
- `severity_label.txt`: severity grade label
- `gait_phase.csv`: optional gait-phase annotation file
- `disease_annotations.json`: disease-specific notes or annotations

### `skeleton/`

Pose-derived skeleton modality generated from RGB and Depth.

```text
skeleton/
├── kpt2d/
│   └── kpt2d.npy
├── kpt3d/
│   ├── kpt3d.npy
│   ├── kpt3d.csv
│   ├── pose_meta.json
│   └── pose_vis/
└── skeleton_map/
    ├── skeleton_maps.npy
    ├── meta.json
    └── png/
```

- `kpt2d/`: 2D body keypoints extracted from RGB
- `kpt3d/`: 3D body keypoints reconstructed from RGB + Depth
- `skeleton_map/`: SkeletonGait-style skeleton map representation

### `calib/`

Calibration and camera parameter files.

```text
calib/
├── intrinsics_d455_color.json
├── intrinsics_d455_depth.json
├── extrinsics_d455_depth_to_color.json
├── extrinsics_d455_to_mmwave.json
└── depth_scale.txt
```

These files are used for:

- RGB-depth alignment
- depth deprojection
- future camera-radar fusion

### `timestamps.csv`

Frame and timestamp correspondence file for the session.

### `meta.json`

Detailed session metadata.

### `session_meta.json`

Compact session summary metadata.

## Modalities

### RGB

- color image sequence
- used for video modeling and 2D pose estimation

### Depth

- aligned depth sequence
- used for depth-aware analysis and 3D pose lifting

### IMU

- inertial motion data from D455
- can be used as an auxiliary time-series modality

### mmWave

- raw radar modality
- can be further converted into radar features such as:
  - range-Doppler maps
  - point clouds
  - other radar representations

### Skeleton

- derived modality built from RGB and Depth
- includes:
  - 2D keypoints
  - 3D keypoints
  - skeleton maps

## Processing Scripts

This repository currently includes:

- `process_dataset.py`
  - convert raw recordings into the processed dataset structure
- `extract_pose_3d.py`
  - generate 2D and 3D keypoints from RGB and Depth
- `build_skeleton_maps.py`
  - convert keypoint sequences into skeleton-map representations
- `reorganize_skeleton_outputs.py`
  - reorganize skeleton-related outputs into `session/skeleton/`

## Head-Turn State Script

For `head_turn` sessions, this repository provides a state-based script:

- `estimate_head_turn_state.py`
  - estimate head-turn state and peak frames from `kpt2d` + aligned `depth`
  - export per-frame state labels, summary metrics, and key-frame previews

This script is intended for robust `head_turn` state/peak detection from the current far-view recordings.
It outputs relative proxy scores rather than clinically calibrated absolute cervical angles.

### Input Requirements

The script expects a processed `head_turn` session containing:

- `rgb/`
- `depth/`
- `calib/depth_scale.txt`
- `skeleton/kpt2d/kpt2d.npy` or `labels/kpt2d.npy`

### Output Files

The script writes outputs to:

```text
head_turn/
└── labels/
    └── head_turn_state/
        ├── states.csv
        ├── summary.json
        ├── key_frames/
        │   ├── neutral.png
        │   ├── subject_left_peak.png
        │   └── subject_right_peak.png
        └── vis/
```

Key fields in `summary.json` include:

- `neutral_frame_index`
- `subject_left_peak_frame_index`
- `subject_right_peak_frame_index`
- `left_max_angle_deg`
- `right_max_angle_deg`
- `total_rom_deg`
- `asymmetry_deg`

### Usage

Single session:

```bash
python estimate_head_turn_state.py --session-dir /your/own/path/processed/S01/head_turn --overwrite --save-vis
```

Batch process a processed dataset root:

```bash
python estimate_head_turn_state.py --root-dir /your/own/path/processed --overwrite --save-vis
```

## Metadata Sync Script

To merge patient information, severity labels, and processed `head_turn` metrics into both session JSON files and an Excel summary table, use:

- `sync_head_turn_metadata.py`
  - read patient metadata from a workbook such as `total.xlsx`
  - update `head_turn` and `walk` label files
  - write `clinical_metadata`, `severity_label`, and `binary_label` into session JSON files
  - write processed head-turn summary metrics back into the workbook

### Input Requirements

The script expects:

- a processed dataset root such as `/your/own/path/processed`
- an Excel workbook containing at least:
  - `subject_id`
  - `severity`

If present, the following workbook columns will be updated with processed head-turn results:

- `left_max_angle`
- `right_max_angle`
- `total_rom_deg`
- `asymmetry_deg`

### Updated JSON Files

For each available `SXX/head_turn` and `SXX/walk`, the script updates:

- `meta.json`
- `session_meta.json`
- `labels/disease_annotations.json`
- `labels/binary_label.txt`
- `labels/severity_label.txt`

For each available `SXX/head_turn`, it also updates:

- `labels/head_turn_state/summary.json`

### Usage

Overwrite the original workbook:

```bash
python sync_head_turn_metadata.py --root-dir /your/own/path/processed --xlsx /your/own/path/total.xlsx
```

Write to a new workbook:

```bash
python sync_head_turn_metadata.py \
  --root-dir /your/own/path/processed \
  --xlsx /your/own/path/total.xlsx \
  --output-xlsx /your/own/path/total_with_head_turn.xlsx
```

## Notes

- The dataset itself is not distributed in this repository.
- The file and folder names in this README follow the current processed dataset format used by this project.
- If the dataset structure is updated later, this README should be updated accordingly.

## License

The **code in this repository** is released under the **MIT License**. See [LICENSE](LICENSE).

The **SpA-MMD dataset itself** is **not covered by the MIT License**. It remains closed-source and is not publicly redistributed through this repository.
