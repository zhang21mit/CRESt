# Robotic Catalyst Testing

## Overview
This repository provides a Jupyter notebook (`alkaline_platform_270.ipynb`) that orchestrates a **six‑axis robotic arm**, liquid‑handling peripherals, and an electrochemical workstation to enable fully‑automated, high‑throughput screening of alkaline OER catalysts.  The workflow is powered by the in‑house `robotic_testing` Python package and exposes a **single‑line API** (`init_alkaline_platform_270`) that spins up the complete experimental stack—robot, pumps, furnace, potentiostat, and data pipeline.

---

## Quick Start

1. **Connect hardware & export IPs**

   ```bash
   export ROBOT_IP=192.xxx
   export POTENTIOSTAT_IP=192.xxx
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
