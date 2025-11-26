# SpA-MMD: Multi-Modal Dataset for Spondyloarthritis (SpA)

> ⚠️ **IMPORTANT NOTICE**  
> This repository contains only the **processing scripts**, dataset **format specification**, and project documentation.  
> The **SpA-MMD dataset itself is NOT included here**, is **NOT open-source**, and is protected under  
> **Creative Commons BY-NC-ND 4.0** — **non-commercial use only**, **no derivatives**, **no redistribution**.  
>  
> To request dataset access, please contact the authors and sign the **Data Usage Agreement (DUA)**.

---

# 📘 Overview

**SpA-MMD** is a *multi-modal pathological gait dataset* for **Spondyloarthritis (SpA)** patients, collected using:

- **RGB camera** (30 FPS)
- **Intel RealSense D455** (RGB, 16-bit depth, IMU)
- **TI mmWave Radar** (IWR6843 / AWR6843)

The dataset is designed for:

- Pathological gait analysis  
- Spinal mobility evaluation  
- Pose & motion estimation  
- Radar–vision fusion  
- Clinical assessment & machine learning research  

This repository includes:

- `process.py` and related tools  
- Dataset folder structure specification  
- Synchronization method  
- Metadata format  
- Licensing and dataset access instructions  

---

# 📁 Dataset Structure (Per Subject → Per Session)

Each subject (SXX) may contain multiple recording sessions.

GaitMultiModalDataset/
├── S01/
│ ├── session_01/
│ ├── session_02/
│ └── ...
├── S02/
└── ...

yaml
Copy code

---

# 📂 Session Directory Specification

Each session folder contains **aligned RGB, Depth, IMU, and Radar data**:

session_01/
├── rgb/
│ └── frame_000001.jpg
│ └── frame_000002.jpg
│
├── depth/
│ └── frame_000001.png # 16-bit depth, mm
│ └── frame_000002.png
│
├── imu/
│ └── imu.csv # ax, ay, az, gx, gy, gz (aligned to 30fps)
│
├── mmwave/
│ ├── raw/ # radar ADC / raw frames
│ └── pointcloud/ # JSON/PLY
│
├── labels/
│ ├── kpt2d/ # 2D keypoints (optional)
│ ├── kpt3d/ # 3D keypoints (optional)
│ ├── gait_phase/ # HS/TO/stance/swing
│ └── disease_annotations.json
│
├── calib/
│ ├── intrinsics_rgb.json
│ ├── intrinsics_d455_color.json
│ ├── intrinsics_d455_depth.json
│ ├── extrinsics_rgb_to_d455.json
│ ├── extrinsics_d455_to_mmwave.json
│ └── depth_scale.txt
│
└── session_meta.json

yaml
Copy code

---

# 🎯 Modalities

### **RGB**
- 30 FPS  
- `frame_XXXXX.jpg`

### **Depth (D455)**
- 16-bit PNG (uint16)  
- Unit: **mm**  
- Best format for 3D skeleton, reconstruction, and medical-grade processing  

### **IMU (D455)**
CSV format:

frame,timestamp,ax,ay,az,gx,gy,gz

yaml
Copy code

### **mmWave Radar**
- `raw/*.bin`
- `pointcloud/*.json` or `.ply`

---

# 🕒 Multi-Modal Synchronization (Manual but Reliable)

Since the devices cannot be hardware-synchronized, we use a **standardized physical event**:

# ✋ **Synchronization Gesture: Raise-to-Head Motion**
This gesture is visible in **all three modalities**.

### **Procedure (3 seconds):**
1. Stand still for 1 second  
2. Raise both hands above head  
3. Hold (freeze) for 1 second  
4. Lower hands  

### Why this gesture?
- RGB: keypoints move significantly  
- Depth: silhouette expands  
- IMU: acceleration spike  
- Radar: point cloud volume increases  

**We align all modalities to RGB 30 FPS timeline based on the detected peak of this gesture.**

---

# 📐 Calibration (Required for Fusion)

Store all calibration files in:

session/calib/

markdown
Copy code

Includes:

- RGB intrinsics  
- D455 color/depth intrinsics  
- Depth → Color extrinsics  
- RGB → Radar extrinsics (optional)  
- Depth scale  

---

# 🔒 Licensing

### ✔ Code License (scripts)
**MIT License**  
Free to use & modify.

### ✔ Dataset License (data files)
**Creative Commons BY-NC-ND 4.0**

- **BY（署名）**  
- **NC（非商业用途）**  
- **ND（禁止修改/衍生）**  
- **No redistribution**  

### ✔ Dataset is *NOT* included or open-source
It is distributed **only upon request + signed DUA**.

---

# 📧 Dataset Access

To obtain SpA-MMD dataset:

1. Email the authors  
2. Sign the **Data Usage Agreement (DUA)**  
3. Receive download link  

Contact: <your-email-here>
Subject: Request for SpA-MMD Dataset Access

yaml
Copy code

---

# 📚 Citation

@dataset{spa_mmd_2025,
title={SpA-MMD: Multi-Modal Dataset for Spondyloarthritis Gait Analysis},
author={Your Name},
year={2025},
description={A synchronized RGB–Depth–IMU–mmWave dataset for pathological gait and spinal mobility assessment.}
}

yaml
Copy code

---

# 🛠 Tools in This Repository

- `process.py`: Convert bag files → RGB / Depth / IMU / PointCloud  
- Automatic session folder generation  
- mmWave + D455 + RGB timestamp alignment  
- Depth 16-bit processing utilities  
- Visualization scripts (optional)

---

# 🚀 Roadmap

- Add 3D skeleton extraction  
- Add Radar-visual fusion demo  
- Add clinical annotation toolkit  
- Release sample sessions (if allowed by IRB)

---

# ❤️ Acknowledgements

Thanks to all SpA patients and medical collaborators who contributed to this project.
