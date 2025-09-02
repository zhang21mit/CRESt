# OT-2 Experiment Runner

## Overview
This repository contains a Jupyter notebook (`ot2_launcher_template.ipynb`) that provides a high-level interface for executing predefined liquid‑handling workflows on an **Opentrons OT‑2** robot. The notebook wraps the custom `liquid_handling_robot.ot2_env` API and pulls task definitions through `utils.utils.get_tasks`, enabling rapid and reproducible execution of complex pipetting protocols used in electrochemical catalyst screening and other high‑throughput experiments.

---

## Quick Start
1. **Clone & create environment**
   ```bash
   git clone <your-fork-url> ot2_runner
   cd ot2_runner
   conda create -n ot2_env python=3.9
   conda activate ot2_env
   pip install -r requirements.txt  # installs opentrons + helpers
   pip install -e .                 # installs liquid_handling_robot in editable mode
   ```
2. **Configure robot IP**
   - Option A Set environment variable once per session:
     ```bash
     export OT2_API_HOST=192.168.x.x  # replace with your robot IP
     ```
   - Option B Hard‑code the address in `liquid_handling_robot/ot2_env.py`.
3. **Launch notebook**
   ```bash
   jupyter lab notebooks/run_ot2_experiment.ipynb
   ```
4. **Select experiment** Edit the `exp_name` variable in the first code cell, e.g.
   ```python
   exp_name = "DFFC_8D_4"
   ```
5. **Load a task list**
   ```python
   from utils.utils import get_tasks
   ot2_tasks = get_tasks(exp_name, task_name="default")
   ```
6. **Run the protocol**
   ```python
   from liquid_handling_robot.ot2_env import OT2_Env_Linux

   ot2_env = OT2_Env_Linux(exp_name)
   ot2_env.run(ot2_tasks, operator_name="zhen", run=True, recording=False)
   ```

---

## Runtime Controls
| Action | Method | Example |
|--------|--------|---------|
| Reset tip count for a rack | `reset_tip_index(pipette, idx)` | `ot2_env.reset_tip_index("20ul", 215)` |
| Reset mixing‑well pointer | `reset_mixing_well_index(idx)` | `ot2_env.reset_mixing_well_index(0)` |
| Update reservoir volumes | `reset_reservoir(reservoir_name_list)` | see code in notebook |
| Manual task execution | Load tasks → `ot2_env.run(..., run=False)` | dry‑run without liquid transfer |

---

## Creating & Editing Tasks
Task files (`.yaml`) live under `tasks/<experiment_name>/`. Each file encodes a sequence of transfers, mixes, delays, module commands, and camera markers. Use an existing template as a starting point:

```yaml
# tasks/DFFC_8D_4/default.yaml (excerpt)
- step: transfer
  pipette: p20_single
  source: reservoir_1/A1
  dest: plate_1/B3
  volume: 10.0
  mix_after:
    volume: 15
    repetitions: 3
```

### Naming Conventions
- **Folders** `<project>_<nD>` where *n* denotes dimensionality of composition space.
- **Files** `<task_name>.yaml` (`default`, `temp`, `manual`, …).

---

## Video Recording & Hyperlapse
Set `recording=True` in `ot2_env.run` to capture raw video frames to `raw_path`. Afterwards convert to a time‑lapse:
```python
from utils.utils import hyperlapse_video
hyperlapse_video(raw_path="/media/li/HDD/ot2_recordings/raw",
                 hyperlapse_path="/media/li/HDD/ot2_recordings/hyperlapse",
                 speed=20)  # 20× real‑time
```



