import os
import sys
import copy
import uuid
import itertools
import datetime
import random
from functools import reduce
from pathlib import Path
from experiments import get_launch_args, sweep_with_devices, launch, exp_id, parse_task_ids, launch_jobs, get_local_storage_path

import multiprocessing

def _get_result_dir(debug):
  base_dir = "debug" if debug else "results"
  return base_dir

if __name__ == '__main__':
  experiment = f"{os.path.basename(os.path.dirname(Path(__file__)))}"
  launch_args = get_launch_args(experiment)

  # Hyperparameters
  algos_scripts = [
    {"script": "{prefix} python main_hebo.py",},
  ]

  # Hyperparameters
  upsi_mains = [
    5.0,
    2.0,
    1.0,
    0.5,
  ]

  upsi_refs = [
    5.0,
    2.0,
    1.0,
    0.5,
    0.1,
    0.01,
  ]
  
  seeds = [seed for seed in range(10)]

  common_exp_args = [
    f"--optim=adarefmax_mixhebo",
  ]

  
  all_job_args = []
  for job_idx, (n_tasks, device,
              algo_script,
              upsi_main,
              upsi_ref,
              seed,
          ) in enumerate(
          sweep_with_devices(itertools.product(
              algos_scripts,
              upsi_mains,
              upsi_refs,
              seeds,
            ),
            devices=launch_args.gpus,
            n_jobs=launch_args.n_jobs,
            n_parallel_task=launch_args.n_parallel_task, shuffle=False)):
    job_args = []
    for task_idx in range(n_tasks):
      wandb_mode = "WANDB_MODE=disabled" if launch_args.debug else ""      
      
      args = [
        algo_script[task_idx]["script"].format(
          prefix=f"{wandb_mode}"),
      ] + common_exp_args
  
      args.append(f"--seed={seed[task_idx]}")
      args.append(f"--upsi_main={upsi_main[task_idx]}")
      args.append(f"--upsi_ref={upsi_ref[task_idx]}")
      args.append(f"--output={_get_result_dir(launch_args.debug)}/adarefmax_mixhebo/{upsi_main[task_idx]}_{upsi_ref[task_idx]}/{seed[task_idx]}")

      job_args.append(" ".join(args))
    
    all_job_args += (job_args)

    if launch_args.debug:
      break

  print(f"Total: {len(all_job_args)}")

  launch_jobs((experiment + f"_{launch_args.run_name}" if launch_args.run_name else experiment),
    all_job_args,
    *parse_task_ids(launch_args.task_id),
    n_jobs=launch_args.n_jobs,
    n_parallel_task=launch_args.n_parallel_task,
    mode=launch_args.mode,
    script="")

  print(f"Total: {len(all_job_args)}, num_gpus={len(launch_args.gpus)}")