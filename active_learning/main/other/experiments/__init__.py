import os
import math
import itertools
import random

# SLURM_TASK_BASE_SCRIPT = "python3 train.py {args}"
#SLURM_TASK_BASE_SCRIPT = "ts -g 0 {entry_script} {args}"
SLURM_TASK_BASE_SCRIPT = "{entry_script} {args}"
#SLURM_TASK_CPU_BASE_SCRIPT = "ts-cpu python3 {entry_script} {args}"
SLURM_TASK_CPU_BASE_SCRIPT = "{entry_script} {args}"

TIG_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}
#SBATCH --exclude=tig-slurm-4

{job_cmds}
wait < <(jobs -p)
"""

TIG_SLURM_CPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}
#SBATCH --exclude=tig-slurm-4

{job_cmds}
wait < <(jobs -p)
"""

IMP_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=imp
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

{job_cmds}
wait < <(jobs -p)
"""

SUPERCLOUD_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

source setup.sh
{job_cmds}
jobs -p
for job in `jobs -p`
do
	echo $job
	wait $job
done

"""

VISION_2080_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=vision-pulkitag-2080
#SBATCH -q vision-pulkitag-main
#SBATCH -t 2-00:00:0
#SBATCH --gres=gpu:10
#SBATCH --exclusive
#SBATCH --mem 700000
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

source setup.sh
{job_cmds}
jobs -p
for job in `jobs -p`
do
	echo $job
	wait $job
  sleep 120
done
"""

# sitzmann-a100-1,
# gpu19-2.drl
FREE_2080_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=csail-shared
#SBATCH --exclude=agrawal-h100-1,sitzmann-h100-1,pizzelle,strudel,tart
#SBATCH -q lab-free
#SBATCH -t 1-00:00:0
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --mem 32G
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}
#SBATCH --requeue

source setup.sh
{job_cmds}
jobs -p
for job in `jobs -p`
do
	echo $job
	wait $job
  sleep 120
done
"""

IMPX_3090_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH -t 2-00:00:0
#SBATCH --gres=gpu:2
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --mem 200000
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

source setup.sh
{job_cmds}
jobs -p
for job in `jobs -p`
do
	echo $job
	wait $job
  sleep 120
done
"""

IMPX_A6000_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH -t 2-00:00:0
#SBATCH --gres=gpu:1
#SBATCH --mem 32G
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

source setup.sh
{job_cmds}
sleep 360
jobs -p
for job in `jobs -p`
do
  echo $job
	wait $job
  sleep 360
done

"""

IMPX_V100_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH -t 2-00:00:0
#SBATCH --gres=gpu:6
#SBATCH --nodes 1
#SBATCH --ntasks={ntasks}
#SBATCH --exclusive
#SBATCH --mem 300000
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

source setup.sh
{job_cmds}
jobs -p
for job in `jobs -p`
do
	echo $job
	wait $job
  sleep 360
done
"""

IMPX_A100_SLURM_GPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH -t 2-00:00:0
#SBATCH --gres=gpu:1
#SBATCH --ntasks={ntasks}
#SBATCH --mem 300000
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

source setup.sh
{job_cmds}
jobs -p
for job in `jobs -p`
do
	echo $job
	wait $job
  sleep 360
done
"""

SUPERCLOUD_SLURM_CPU_BASE_SCRIPT = """#!/bin/bash
#SBATCH --partition=xeon-p8
#SBATCH -c 16
#SBATCH --ntasks={ntasks}
#SBATCH --output=slurm_outputs/{job_name}_%A_%a.out
#SBATCH --error=slurm_outputs/{job_name}_%A_%a.err
#SBATCH --job-name={job_name}

module load anaconda/2023b
{job_cmds}
"""


BASE_LOCAL_SCRIPT = """{entry_script} {args}"""

def exp_id():
  import uuid
  return uuid.uuid4()

def hostname():
  import subprocess
  cmd = 'hostname -f'
  try:
      p = subprocess.check_output(cmd, shell=True)  # Save git diff to experiment directory
      return p.decode('utf-8').strip()
  except subprocess.CalledProcessError as e:
      print(f"can not get obtain hostname via `{cmd}` due to exception: {e}")
      return None


def get_local_storage_path():
  return {
    "visiongpu50.csail.mit.edu": "/mnt/scratch/zwhong",
    "improbablex001.csail.mit.edu": "/scratch/zwhong",
    "improbablex002.csail.mit.edu": "/scratch/zwhong",
    "improbablex003.csail.mit.edu": "/scratch/zwhong",
    "improbablex004.csail.mit.edu": "/scratch/zwhong",
    "improbablex005.csail.mit.edu": "/mnt/ssd1/zwhong",
    "improbablex006.csail.mit.edu": "/mnt/ssd1/zwhong",
    "improbablex007.csail.mit.edu": "/mnt/ssd1/zwhong",
    # "improbablex009.csail.mit.edu": "/mnt/ssd1/zwhong",
    "katsudon.csail.mit.edu": "/home/zwhong"
  }.get(hostname(), ".")


def _screen(job_name, run_cmd, verbose=True):
    cmd = f"screen -S {job_name} -dm bash -c \"{run_cmd}\""
    if verbose:
      print(cmd)
    os.system(cmd)

def _jaynes_entry(entry_script, args):
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"{script}"
    print(cmd)
    os.system(cmd)

def gcp(job_name, args, entry_script, verbose=False, **kwargs):
    import jaynes

    jaynes.config(
      # "local",
      name=f"{job_name}",
      launch={'name': f"drr_{job_name}"},
      verbose=True)
    instances = jaynes.run(_jaynes_entry,
      entry_script,
      args=args
    )

def sbatch(job_name, args, entry_script, gpu=0, n_parallel_task=1, verbose=False, slurm_script=None, **kwargs):
    task_script = ""
    task_size = math.ceil((len(args) / n_parallel_task) if len(args) >= 1 else 1)
    print("n_parallel_task: ", n_parallel_task)
    for sub_task_idx in (range(0, len(args), task_size)):
      end_sub_task_idx = sub_task_idx + task_size
    
      if gpu == -1:
        task_script += ";".join([SLURM_TASK_CPU_BASE_SCRIPT.format(args=task_args, entry_script=entry_script) for task_args in args[sub_task_idx:end_sub_task_idx]])
      else:
        task_script += ";".join([SLURM_TASK_BASE_SCRIPT.format(args=task_args, entry_script=entry_script) for task_args in args[sub_task_idx:end_sub_task_idx]])
      
      if n_parallel_task > 1:
        task_script += " & \n"
      else:
        task_script += "\n"
    
    if slurm_script is not None:
      script_mapping = {
        "2080": VISION_2080_SLURM_GPU_BASE_SCRIPT,
        "3090": IMPX_3090_SLURM_GPU_BASE_SCRIPT,
        "a6000": IMPX_A6000_SLURM_GPU_BASE_SCRIPT,
        "v100": IMPX_V100_SLURM_GPU_BASE_SCRIPT,
        "a100": IMPX_A100_SLURM_GPU_BASE_SCRIPT,
        "free": FREE_2080_SLURM_GPU_BASE_SCRIPT,
        "xeonp8": SUPERCLOUD_SLURM_CPU_BASE_SCRIPT,
      }
      script = script_mapping[slurm_script].format(job_name=job_name, job_cmds=task_script, ntasks=n_parallel_task)
      print(f"Use {slurm_script}'s script")
    else:
      _hostname = hostname()
      if _hostname.startswith("slurm-control"):
        script = TIG_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
        print("Use TIG Slurm script")
      elif _hostname.startswith("improbable"):
        script = IMP_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
        print("Use improbable script")
      elif _hostname.startswith("login"):
        if gpu == -1:
          script = SUPERCLOUD_SLURM_CPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
          print("Use supercloud CPU script")
        else:
          script = SUPERCLOUD_SLURM_GPU_BASE_SCRIPT.format(job_name=job_name, job_cmds=task_script)
          print("Use supercloud GPU script")
        # import ipdb; ipdb.set_trace()
    cmd = f'/bin/bash -c \"sbatch <<< \\"{script}\\" "'
    if verbose:
      print(cmd)
    os.system(cmd)

def screen(job_name, args, entry_script, verbose=False, **kwargs):
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"screen -S {job_name} -dm bash -c \"echo $STY; source setup.sh; {script}\""
    if verbose:
      print(cmd)
    os.system(cmd)

def docker(job_name, args, entry_script, verbose=False, **kwargs):
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"docker run --gpus=all --name={job_name} -d improbableailab/d3rlpy:latest bash -c \"{script}\""
    if verbose:
      print(cmd)
    os.system("docker pull improbableailab/d3rlpy:latest")
    os.system(cmd)

def bash(job_name, args, entry_script, verbose=False, **kwargs):
    script = ""
    for idx, task_args in enumerate(args):
      script += BASE_LOCAL_SCRIPT.format(args=task_args, entry_script=entry_script)
      if idx != len(args) - 1:
        script += "\n"

    cmd = f"{script}"
    if verbose:
      print(cmd)
    #os.system(cmd)

def local(job_name, args, entry_script, verbose=False, **kwargs):
    assert len(args) == 1 # Not support parallel screen jobs for now
    script = BASE_LOCAL_SCRIPT.format(args=args[0], entry_script=entry_script)
    cmd = f"{script}"

    if verbose:
      print(cmd)
    os.system(cmd)

def launch(job_name, args, mode, entry_script, verbose=False, **kwargs):
  if mode.startswith('sbatch'):
    if "-" in mode: # Select script
      slurm_script = mode.split("-")[1]
    else:
      slurm_script = None
    sbatch(job_name, args, entry_script, verbose=verbose, slurm_script=slurm_script, **kwargs)
  elif mode == 'screen':
    screen(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == "docker":
    docker(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == 'gcp':
    gcp(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == 'bash':
    return bash(job_name, args, entry_script, verbose=verbose, **kwargs)
  elif mode == 'local':
    local(job_name, args, entry_script, verbose=verbose, **kwargs)
  else:
    raise NotImplemented()

def parse_task_ids(task_ids):
  if task_ids == ":":
    start_task_id = 0
    end_task_id = None
  else:
    start_task_id, end_task_id = map(lambda x: int(x), task_ids.split(":"))

  return start_task_id, end_task_id

def launch_jobs(experiment, all_job_args, start_task_id, end_task_id, n_jobs, mode, script, n_parallel_task=1):
  import time, math
  if end_task_id is None:
    end_task_id = len(all_job_args) - 1
  job_launched = 0
  job_size = math.ceil((end_task_id + 1 - start_task_id) / n_jobs) if n_jobs >= 1 else 1
  expID = exp_id()
  for idx, i in enumerate(range(start_task_id, end_task_id + 1, job_size)):
    launch(f"{experiment}_{expID}_{idx}", all_job_args[i: min(i + job_size, end_task_id + 1)],
        mode=mode, entry_script=script,
        n_parallel_task=n_parallel_task,
        verbose=True)
    print(f"Run task {i}-{min(i + job_size, end_task_id + 1)}")
    job_launched += 1
    # time.sleep(3)
  print(f"Launched {job_launched} jobs. Each job runs {job_size} tasks.")


def get_launch_args(experiment):
  import argparse
  parser = argparse.ArgumentParser(description=f'{experiment}')
  parser.add_argument('--gpus', nargs="+", type=int, help="GPU Id lists (e.g., 0,1,2,3)", default=0)
  parser.add_argument('--mode', type=str, choices=[
    'sbatch',
    *[f'sbatch-{gpu_model}' for gpu_model in ["2080", "3090", "v100", "a100", "a6000", "free"]],
    'sbatch-xeonp8',
    'screen', 
    'bash', 
    'local', 
    "gcp", 
    "docker"], required=True)
  parser.add_argument('--n_parallel_task', type=int, default=1, help="Number of parallel jobs in on sbatch submission")
  parser.add_argument('--task_id', help="e.g., 5:10", type=str, required=False, default=":")
  parser.add_argument('--debug', action='store_true', default=False)
  parser.add_argument('--seq', action='store_true', default=False)
  parser.add_argument('--n_jobs', type=int, help="Number of jobs running in sequence.", default=-1)
  parser.add_argument('--n_task_per_gpu', type=int, help="Number of tasks running on the same gpu.", default=1)
  parser.add_argument('--tags', nargs="+", type=str, default=[])
  parser.add_argument('--run_name', type=str, default=None)
  args = parser.parse_args()
  args.gpus = [args.gpus] if isinstance(args.gpus, int) else args.gpus
  return args

def to_tuple_list(list_tuple):
  tuple_lists = [[] for i in range(len(list_tuple[0]))]
  for t in list_tuple:
    for i, e in enumerate(t):
      tuple_lists[i].append(e)
  return tuple_lists

def sweep(sweep_args, n_parallel_task=1, shuffle=False):
  buffer = [] # a list of tuple, each tuple is one arg combination
  if shuffle:
    sweep_args = list(sweep_args)
    random.shuffle(sweep_args)
  n_args = len(sweep_args)
  for args in sweep_args:
    buffer.append(args)
    if len(buffer) == n_parallel_task:
      yield (len(buffer), *to_tuple_list(buffer))
      buffer = []
  if len(buffer) > 0:
    yield (len(buffer), *to_tuple_list(buffer))
    buffer = []

def sweep_with_devices(sweep_args, devices, n_jobs, n_parallel_task=1, selected_task_indices=None, shuffle=False):
  buffer = [] # a list of tuple, each tuple is one arg combination
  devices_buffer = []
  sweep_args = list(sweep_args)
  
  if selected_task_indices is not None:
    sweep_args = [sweep_args[i] for i in selected_task_indices]

  if shuffle:    
    random.shuffle(sweep_args)
  
  n_args = len(sweep_args)
  n_tasks_per_device = math.ceil(n_args / (n_jobs * n_parallel_task))

  for idx, args in enumerate(sweep_args):
    buffer.append(args)
    devices_buffer.append(devices[(idx // n_tasks_per_device) % len(devices)])
    if len(buffer) == n_parallel_task:
      yield (len(buffer), devices_buffer, *to_tuple_list(buffer))
      buffer = []
      devices_buffer = []

  if len(buffer) > 0:
    yield (len(buffer), devices_buffer, *to_tuple_list(buffer))
    buffer = []
    devices_buffer = []
