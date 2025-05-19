# Alkaline Platform 270 – Robotic Catalyst Testing

## Overview
This repository provides a Jupyter notebook (`alkaline_platform_270.ipynb`) that orchestrates a **six‑axis robotic arm**, liquid‑handling peripherals, and an electrochemical workstation to enable fully‑automated, high‑throughput screening of alkaline OER catalysts.  The workflow is powered by the in‑house `robotic_testing` Python package and exposes a **single‑line API** (`init_alkaline_platform_270`) that spins up the complete experimental stack—robot, pumps, furnace, potentiostat, and data pipeline.

---

## Repository Layout
```
.
├── notebooks/
│   └── alkaline_platform_270.ipynb  # ← this notebook
├── robotic_testing/
│   ├── platforms/
│   │   ├── __init__.py
│   │   └── alkaline_platform_270.py
│   ├── arm/
│   │   ├── gripper_driver.py
│   │   └── motion_primitives.py
│   ├── data_pipeline/
│   │   └── analysis.py
│   └── utils/
│       └── config.py
└── requirements.txt
```

---

## Hardware Requirements
| Subsystem | Model | Notes |
|-----------|-------|-------|
| **Robot arm** | Dorna 2 / Dobot M1 (6‑DOF) | End‑effector fitted with 2‑finger gripper |
| **Potentiostat** | Bio‑Logic VSP‑300 | Ethernet control |
| **Pump rack** | Tecan Cavro XLP | Addressed via RS‑485‑USB |
| **Flask station** | Custom 270‑position carousel | 3‑neck glass flasks |
| **Vision** | Intel RealSense D435 | Depth camera for pose correction |

Ensure all devices are on the same LAN segment and reachable from the host PC.

---

## Software Prerequisites
- **Python ≥ 3.9** (tested with 3.11)
- `robotic_testing` (this repo) – install editable
- `pyserial`, `opencv‑python`, `scipy`, `pandas`, `matplotlib`
- **Vendor SDKs**  
  - Dorna API v3  
  - Bio‑Logic EC‑Lab SDK

```bash
conda create -n rtest python=3.9
conda activate rtest
pip install -r requirements.txt
pip install -e .   # installs robotic_testing in editable mode
```

---

## Quick Start

1. **Connect hardware & export IPs**

   ```bash
   export ROBOT_IP=192.168.1.20
   export POTENTIOSTAT_IP=192.168.1.30
   ```

2. **Launch the notebook**

   ```bash
   jupyter lab notebooks/alkaline_platform_270.ipynb
   ```

3. **Initialize the platform**

   ```python
   from robotic_testing.platforms import init_alkaline_platform_270
   rt = init_alkaline_platform_270(exp_name="demo_run")
   ```

4. **Run a canned experiment**

   ```python
   run_cfg = dict(
       sequence=True,
       sample_id_list=["NiFeOx_001", "NiFeOx_002"],
       benchmark={30: "Pt_electrode"},
       immerse_option="flask_contact_immersed",
       operator_name="zhen",
   )
   rt.run(**run_cfg)
   ```

5. **Manual arm commands**

   ```python
   rt.arm.home()
   rt.arm.pick_up_sample(8)
   rt.arm.rinsing()
   rt.arm.put_sample_back(8)
   ```

6. **Post‑experiment analysis**

   ```python
   rt.data_analyze(sample_id=262,
                   test_name="alk_mor_PtRuSc#262_2025-05-18_21-51-36",
                   channel_id=1)
   ```

---

## Runtime Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `sequence` | Run predefined sequence vs. single-job mode | `False` |
| `sample_id_list` | Samples to test in order | `[]` |
| `benchmark` | Dict mapping cycle # → benchmark electrode | `{}` |
| `immerse_option` | `"flask_contact_immersed"` / `"no_immerse"` | `"flask_contact_immersed"` |
| `data_analysis` | Perform inline ECSA/Tafel extraction | `False` |
| `recording` | Capture RGB‑D timelapse | `False` |

---

## Creating New Workflows

1. **Subclass `BasePlatform`**

   ```python
   from robotic_testing.base import BasePlatform

   class AcidicPlatform(BasePlatform):
       ...
   ```

2. **Register in `robotic_testing.platforms.__init__.py`**

3. **Attach YAML config** under `robotic_testing/utils/config.py`.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `SerialException: device not found` | COM port wrong | `dmesg | grep ttyUSB` then update `config.py` |
| Robot overshoots rack position | Homing lost | `rt.arm.home()` then recalibrate vision offset |
| EC‑Lab times out | IP mismatch / firewall | Verify IP, disable Windows firewall for EC‑Lab |

---

## License
Distributed under the MIT License. See `LICENSE`.

## Authors
- **Zhen Zhang** – Electrochemical Automation Lab  
- Contributors: Chu Li, Wei Liu

---

## Citation
> Zhang, Z. *et al.* “Autonomous High‑Throughput Alkaline OER Screening with a Robotic Arm Platform 270.” (2025).

---

## Acknowledgements
Supported by NSF award #XXXXXXX and Dorna Robotics academic grant.
