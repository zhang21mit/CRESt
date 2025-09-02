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

script_path = os.path.dirname(__file__)

class HEATargetFunction:

    def __init__(self, model_path = "HEA.joblib"):
        #scikit-learn version==1.2.2
        # Load the pre-trained regression model
        self.reg_model = joblib.load(model_path)

        # Define the bounds for the 8-dimensional input vector
        self.pbounds = {}
        # for i in range(8):
        #     self.pbounds['x%d' % (i)] = (0, 1)
        for dim_name in self.dim_names:
            self.pbounds[dim_name] = (0, 1)

    @property
    def input_dim(self):
        return 8

    @property
    def dim_names(self):
        # return [f"x{i}" for i in range(8)]
        return ["Pd", "Pt", "Cu", "Au", "Ir", "Ce", "Nb", "Cr"]

    @property
    def output_name(self):
        return "max_power"

    def __call__(self, **kwargs):
        element = []
        for x in self.dim_names:
            element.append(kwargs[x])
        element = np.array(element) + 1e-6  # Add a small value to avoid zero
        element /= element.sum()  # Normalize the vector to sum to 1
        return self.reg_model.predict(element.reshape(1, -1))[0]  # Predict the output using the regression model

class CustomHEBO(HEBO):

    def __init__(self, space, model_name = 'gp', rand_sample = None, acq_cls = MACE, es = 'nsga2', model_config = None, scramble_seed = None, 
            upsi = 0.5): 
        super().__init__(space, model_name, rand_sample, acq_cls, es, model_config, scramble_seed)
        self.upsi = upsi

    def suggest(self, n_suggestions=1, fix_input = None):
        if self.acq_cls != MACE and n_suggestions != 1:
            raise RuntimeError('Parallel optimization is supported only for MACE acquisition')
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            return sample
        else:
            X, Xe = self.space.transform(self.X)
            try:
                if self.y.min() <= 0:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                else:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'box-cox'))
                    if y.std() < 0.5:
                        y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                if y.std() < 0.5:
                    raise RuntimeError('Power transformation failed')
                model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                model.fit(X, Xe, y)
            except:
                y     = torch.FloatTensor(self.y).clone()
                model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                model.fit(X, Xe, y)

            best_id = self.get_best_id(fix_input)
            best_x  = self.X.iloc[[best_id]]
            best_y  = y.min()
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best.detach().numpy().squeeze()
            ps_best = ps2_best.sqrt().detach().numpy().squeeze()

            iter  = max(1, self.X.shape[0] // n_suggestions)
            upsi  = self.upsi
            delta = 0.01
            # kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))
            kappa = np.sqrt(upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi**2 / (3 * delta))))
            acq = self.acq_cls(model, best_y = py_best, kappa = kappa) # LCB < py_best
            mu  = Mean(model)
            sig = Sigma(model, linear_a = -1.)
            
            opt = EvolutionOpt(self.space, acq, pop = 100, iters = 100, verbose = False, es=self.es)
            rec = opt.optimize(initial_suggest = best_x, fix_input = fix_input).drop_duplicates()
            rec = rec[self.check_unique(rec)]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec      = pd.concat([rec, rand_rec], axis = 0, ignore_index = True)
                cnt +=  1
                if cnt > 3:
                    # sometimes the design space is so small that duplicated sampling is unavoidable
                    break 
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rec      = pd.concat([rec, rand_rec], axsi = 0, ignore_index = True)

            select_id = np.random.choice(rec.shape[0], n_suggestions, replace = False).tolist()
            x_guess   = []
            with torch.no_grad():
                py_all       = mu(*self.space.transform(rec)).squeeze().numpy()
                ps_all       = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                best_pred_id = np.argmin(py_all)
                best_unce_id = np.argmax(ps_all)
                if best_unce_id not in select_id and n_suggestions > 2:
                    select_id[0]= best_unce_id
                if best_pred_id not in select_id and n_suggestions > 2:
                    select_id[1]= best_pred_id
                rec_selected = rec.iloc[select_id].copy()
            return rec_selected

class MixHEBO(HEBO):

    def __init__(self, space, model_name = 'gp', rand_sample = None, acq_cls = MACE, es = 'nsga2', model_config = None, scramble_seed = None, 
            upsi_main = 2.0,
            upsi_ref = 0.5): 
        super().__init__(space, model_name, rand_sample, acq_cls, es, model_config, scramble_seed)
        self.upsi_main = upsi_main
        self.upsi_ref = upsi_ref

    def _sample(self, acq, mu, sig, n_suggestions, best_x, fix_input):
        opt = EvolutionOpt(self.space, acq, pop = 100, iters = 100, verbose = False, es=self.es)
        rec = opt.optimize(initial_suggest = best_x, fix_input = fix_input).drop_duplicates()
        rec = rec[self.check_unique(rec)]

        cnt = 0
        while rec.shape[0] < n_suggestions:
            rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
            rand_rec = rand_rec[self.check_unique(rand_rec)]
            rec      = pd.concat([rec, rand_rec], axis = 0, ignore_index = True)
            cnt +=  1
            if cnt > 3:
                # sometimes the design space is so small that duplicated sampling is unavoidable
                break 
        if rec.shape[0] < n_suggestions:
            rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
            rec      = pd.concat([rec, rand_rec], axsi = 0, ignore_index = True)

        select_id = np.random.choice(rec.shape[0], n_suggestions, replace = False).tolist()
        x_guess   = []
        with torch.no_grad():
            py_all       = mu(*self.space.transform(rec)).squeeze().numpy()
            ps_all       = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
            best_pred_id = np.argmin(py_all)
            best_unce_id = np.argmax(ps_all)
            if best_unce_id not in select_id and n_suggestions > 2:
                select_id[0]= best_unce_id
            if best_pred_id not in select_id and n_suggestions > 2:
                select_id[1]= best_pred_id
            rec_selected = rec.iloc[select_id].copy()

            return rec_selected, mu(*self.space.transform(rec_selected)).squeeze().numpy()

    def find_feasible_rec(self, costs_ref, acq_main, mu, sig, n_suggestions, best_x, fix_input):
        passed_recs = []
        n_passed = 0
        while n_passed < n_suggestions:
            rec_main, costs_main = self._sample(acq_main, mu, sig, (n_suggestions - n_passed), best_x, fix_input)

            if (n_suggestions - n_passed) == 1:
                costs_main = costs_main.reshape(-1)
            
            passed_recs.append(rec_main[costs_main <= costs_ref.mean()])
            n_passed += len(passed_recs[-1])

        return pd.concat(passed_recs, axis = 0, ignore_index = True)

    def suggest(self, n_suggestions=1, fix_input = None):
        if self.acq_cls != MACE and n_suggestions != 1:
            raise RuntimeError('Parallel optimization is supported only for MACE acquisition')
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            return sample
        else:
            X, Xe = self.space.transform(self.X)
            try:
                if self.y.min() <= 0:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                else:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'box-cox'))
                    if y.std() < 0.5:
                        y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                if y.std() < 0.5:
                    raise RuntimeError('Power transformation failed')
                model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                model.fit(X, Xe, y)
            except:
                y     = torch.FloatTensor(self.y).clone()
                model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                model.fit(X, Xe, y)

            best_id = self.get_best_id(fix_input)
            best_x  = self.X.iloc[[best_id]]
            best_y  = y.min()
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best.detach().numpy().squeeze()
            ps_best = ps2_best.sqrt().detach().numpy().squeeze()

            iter  = max(1, self.X.shape[0] // n_suggestions)
            upsi_main = self.upsi_main # 2.0 
            upsi_ref = self.upsi_ref # 0.1
            delta = 0.01

            # kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))
            kappa_main = np.sqrt(upsi_main * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi**2 / (3 * delta))))
            kappa_ref = np.sqrt(upsi_ref * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi**2 / (3 * delta))))
            acq_main = self.acq_cls(model, best_y = py_best, kappa = kappa_main) # LCB < py_best
            acq_ref = self.acq_cls(model, best_y = py_best, kappa = kappa_ref) # LCB < py_best
            mu  = Mean(model)
            sig = Sigma(model, linear_a = -1.)
            
            rec_ref, costs_ref = self._sample(acq_ref, mu, sig, n_suggestions, best_x, fix_input)

            return self.find_feasible_rec(costs_ref, acq_main, mu, sig, n_suggestions, best_x, fix_input)

class MaxMixHEBO(MixHEBO):

    def find_feasible_rec(self, costs_ref, acq_main, mu, sig, n_suggestions, best_x, fix_input):
        while True:
            rec_main, costs_main = self._sample(acq_main, mu, sig, n_suggestions, best_x, fix_input)
            if costs_main.min() <= costs_ref.mean():
                return rec_main

class RefMaxMixHEBO(MixHEBO):

    def find_feasible_rec(self, costs_ref, acq_main, mu, sig, n_suggestions, best_x, fix_input):
        
        n_attempts = 0
        max_attempts = 50
        best_cost = np.inf
        best_attempt = None
        
        while n_attempts < max_attempts:
            rec_main, costs_main = self._sample(acq_main, mu, sig, n_suggestions, best_x, fix_input)
            
            if costs_main.min() < best_cost:
                best_attempt = rec_main
                best_cost = costs_main.min()

            if costs_main.min() <= costs_ref.min():
                return rec_main

            n_attempts += 1
        
        return best_attempt

class AdaRefMaxMixHEBO(MixHEBO):

    def find_feasible_rec(self, costs_ref, acq_main, mu, sig, n_suggestions, best_x, fix_input):
        frac = 1.0
        delta = 0.999
        while True:
            rec_main, costs_main = self._sample(acq_main, mu, sig, n_suggestions, best_x, fix_input)
            if costs_main.min() <= costs_ref.min() * frac:
                return rec_main
            frac *= delta

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iters', type=int, default=5)
    parser.add_argument('--n_suggestions', type=int, default=20)
    parser.add_argument('--optim', type=str, default='hebo', 
        choices=[
            'hebo', 
            'mixhebo', 
            'max_mixhebo',
            'refmax_mixhebo',
            'adarefmax_mixhebo',
        ])
    parser.add_argument('--output', type=str, default='./output.csv')
    parser.add_argument('--upsi_main', type=float, default=0.5)
    parser.add_argument('--upsi_ref', type=float, default=0.1)
    parser.add_argument('--model_path', type=str, default='HEA.joblib')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()


    f = HEATargetFunction()
    def obj(params : pd.DataFrame) -> np.ndarray:
        params = params.apply(lambda row: row / row.sum(), axis=1) # Normalize to sum to 1
        return np.array([-f(**param) for param in params.to_dict(orient = 'records')]).reshape(-1, 1)

    space = DesignSpace().parse([
        *[{"name": dim_name, "type": "num", "lb": f.pbounds[dim_name][0], "ub": f.pbounds[dim_name][1]} for dim_name in f.dim_names]
    ])

    if args.optim == "hebo":
        # Same as the original HEBO. Customizing for tuning the upsi parameter.
        opt = CustomHEBO(space, upsi=args.upsi_main, scramble_seed=args.seed)
    elif args.optim == "max_mixhebo":
        opt = MaxMixHEBO(space, upsi_main=args.upsi_main, upsi_ref=args.upsi_ref, scramble_seed=args.seed)
    elif args.optim == "refmax_mixhebo":
        opt = RefMaxMixHEBO(space, upsi_main=args.upsi_main, upsi_ref=args.upsi_ref, scramble_seed=args.seed)
    elif args.optim == "adarefmax_mixhebo":
        opt = AdaRefMaxMixHEBO(space, upsi_main=args.upsi_main, upsi_ref=args.upsi_ref, scramble_seed=args.seed)
    else:
        opt = MixHEBO(space, upsi_main=args.upsi_main, upsi_ref=args.upsi_ref, scramble_seed=args.seed)

    wandb.init(
        project=f"hebo_hea_fix_{args.model_path.split('.')[0]}",
        entity="improbableai_zwh",
        config=vars(args),
    )

    # Log
    logs = []

    # Load priors
    priors = pd.read_excel('batch0.xlsx')
    rec0, obj0 = priors[f.dim_names], priors[f.output_name]
    opt.observe(rec0, -obj0.to_numpy().reshape(-1, 1))
        
    log = rec0.copy()
    log[f.output_name] = obj0
    log["iter"] = 0
    logs.append(log)

    for i in range(args.n_iters):
        rec = opt.suggest(n_suggestions=args.n_suggestions)
        obj_rec = obj(rec)
        opt.observe(rec, obj_rec)
        print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))

        log = rec.copy()
        log[f.output_name] = -obj_rec.reshape(-1)
        log["iter"] = i + 1
        logs.append(log)

        wandb.log({"best_obj": opt.y.min(), "iter": i})
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.concat(logs, axis=0).to_csv(args.output, index=False)