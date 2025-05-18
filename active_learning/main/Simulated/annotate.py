import pandas as pd
import os
import numpy  as np
import torch
import wandb
import joblib

import hebo.acquisitions.acq
from hebo.models.model_factory import get_model
from hebo.acquisitions.acq import MACE, Mean, Sigma
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from main_hebo import HEATargetFunction

script_path = os.path.dirname(__file__)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--start_iter', type=int, required=True)
    parser.add_argument('--label', type=str, required=True)
    args = parser.parse_args()

    f = HEATargetFunction()
    space = DesignSpace().parse([
        *[{"name": dim_name, "type": "num", "lb": f.pbounds[dim_name][0], "ub": f.pbounds[dim_name][1]} for dim_name in f.dim_names]
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    if args.start_iter < 1:
        assert 0
    
    rec = pd.read_excel(f'{args.output_dir}/batch{args.start_iter}.xlsx')
    def obj(params : pd.DataFrame) -> np.ndarray:
        params = params.apply(lambda row: row / row.sum(), axis=1) # Normalize to sum to 1
        return np.array([f(**param) for param in params.to_dict(orient = 'records')]).reshape(-1, 1)
    rec[args.label] = obj(rec) # Store in positive

    rec.to_excel(f'{args.output_dir}/batch{args.start_iter}.xlsx', index=False)
    print(f'Best: {rec[args.label].max()}')