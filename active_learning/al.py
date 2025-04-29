from abc import ABC

import numpy as np
import pandas as pd
import sigfig
import torch
from ax import (
    Data, Metric, SearchSpace, ParameterType, RangeParameter, Objective, Experiment,
    OptimizationConfig,
    Runner, SumConstraint
)
from ax.core.arm import Arm
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.utils.report_utils import exp_to_df
from ax.storage.botorch_modular_registry import register_acquisition_function
from ax.storage.registry_bundle import RegistryBundle
from ax.storage.sqa_store.db import init_engine_and_session_factory, get_engine, create_all_tables
from ax.storage.sqa_store.delete import delete_experiment
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.utils.common.result import Ok
from botorch.acquisition import qUpperConfidenceBound, qExpectedImprovement, qProbabilityOfImprovement, \
    qKnowledgeGradient
from botorch.models.gp_regression import SingleTaskGP
from sqlalchemy import Text
from sqlalchemy.ext.compiler import compiles

from db_control.database import Database
from liquid_handling_robot.ot2_env import OT2_Env_Linux, OT2_Env_Win
from utils.utils import complement_value, config_loader, normalize_task


class AL:
    def __init__(self, exp_name, load_from_db=True, connect_ot2=True, ot2_env='linux'):

        self.connect_ot2 = connect_ot2
        self.exp_name = exp_name
        self.db_url = f'postgresql+psycopg2://li:nano@10.140.0.20:5432/{exp_name}'
        self.al_configs = config_loader(exp_name, 'al_configs')
        self.ot2_run_configs = config_loader(exp_name, 'ot2_run_configs')
        self.ot2_env = ot2_env

        self.db = Database(exp_name=self.exp_name)
        self.bo_model = None
        self.gs = None

        # exp setup
        self.elements = self.al_configs['elements']
        # how many times are repeated for each recipe vertically
        self.sample_repeats = self.ot2_run_configs['repeat']
        # name of the metric to be optimized
        self.metric_name = self.al_configs['metric_name']

        register_acquisition_function(qUpperConfidenceBound)

        # define the metric
        class AL_Metric(Metric):
            def fetch_trial_data(self, trial, **kwargs):
                records = []
                db = Database(trial.experiment.name)
                for arm_name, arm in trial.arms_by_name.items():
                    params = arm.parameters
                    records.append({
                        "arm_name": arm_name,
                        "metric_name": self.name,
                        "trial_index": trial.index,
                        "mean": db.get_arm_results(arm_name, next(iter(trial.experiment.metrics))),
                        "sem": None,
                    })
                return Ok(value=Data(df=pd.DataFrame.from_records(records)))

        # define the runner
        class LiquidHandler(Runner, ABC):
            def run(self, trial):
                trial_metadata = {"name": str(trial.index)}
                return trial_metadata

        # type id has to be specified, choose any int larger than 99, hash not working somehow
        bundle = RegistryBundle(
            metric_clss={AL_Metric: 666},
            runner_clss={LiquidHandler: 666}
        )
        self.sqa_config = SQAConfig(
            json_encoder_registry=bundle.encoder_registry,
            json_decoder_registry=bundle.decoder_registry,
            metric_registry=bundle.metric_registry,
            runner_registry=bundle.runner_registry,
        )

        if load_from_db:
            self.load_exp_from_db()
        else:
            # create search space
            self.search_space = SearchSpace(
                parameters=[
                    RangeParameter(
                        name=f"{self.elements[i]}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0, digits=3
                    )
                    # last element will be filled automatically by summation to 1
                    for i in range(len(self.elements) - 1)
                ]
            )

            # bound the sum by 1
            self.sum_constraint = SumConstraint(
                parameters=list(self.search_space.parameters.values()),
                is_upper_bound=True,
                bound=1.0,
            )
            self.search_space.add_parameter_constraints([self.sum_constraint])

            # initialize optimization config
            self.optimization_config = OptimizationConfig(
                objective=Objective(
                    metric=AL_Metric(name=self.metric_name),
                    minimize=self.al_configs['minimize'],
                )
            )

            # create experiment
            self.exp = Experiment(
                name=self.exp_name,
                search_space=self.search_space,
                optimization_config=self.optimization_config,
                runner=LiquidHandler(),
            )

        # create view if not exist
        if not self.db.check_view_exist_on_server('full_table'):
            self.db.create_view(self.elements, self.metric_name)

    def convert_trial_to_df(self, trial, add_pred=True, model=None):

        data_dicts = []

        for arm in trial.arms:
            data_dict = {
                'arm_name': arm.name,
                'trial_index': trial.index,
                'arm_index': arm.name.split('_')[-1],
                self.elements[-1]: complement_value(arm.parameters),
                **arm.parameters
            }
            if add_pred:
                model = model or self.get_bo_model()
                mean_dict, var_dict = model.predict([ObservationFeatures(arm.parameters)])
                data_dict['pred_mean'] = sigfig.round(mean_dict[self.metric_name][0], 4)
                data_dict['pred_std'] = sigfig.round(var_dict[self.metric_name][self.metric_name][0] ** 0.5, 4)
            data_dicts.append(data_dict)

        data_df = pd.DataFrame.from_records(data_dicts)

        return data_df

    def get_bo_model(self, acquisition='qUCB', cuda=False, **kwargs):
        """
        Args:
            acquisition: the acquisition function to be used
            cuda: whether to use cuda (GPU) or not
            **kwargs: kwargs for the botorch model
        """
        acquisition_dict = {
            'qUCB': qUpperConfidenceBound,
            'qEI': qExpectedImprovement,
            'qPI': qProbabilityOfImprovement,
            'qKG': qKnowledgeGradient
        }
        device = torch.device("cuda" if cuda else "cpu")
        bo_model = Models.BOTORCH_MODULAR(
            experiment=self.exp,
            data=self.exp.fetch_data(**kwargs),
            surrogate=Surrogate(SingleTaskGP),
            botorch_acqf_class=acquisition_dict[acquisition],
            acquisition_options={**kwargs},
            fit_out_of_design=True,
            torch_device=device,
            default_model_gen_options={**kwargs},
        )
        return bo_model

    def generate_sobol_trial(self, n=1):
        sobol = Models.SOBOL(search_space=self.exp.search_space)
        generator_run = sobol.gen(n=n)
        trial = self.exp.new_batch_trial(generator_run=generator_run)
        trial.run()
        print(f'SOBOL trial successfully added as trial {trial.index}')

    def generate_botorch_trial(self, n=1, **kwargs):
        bo_model = self.get_bo_model(**kwargs)
        generator_run = bo_model.gen(n=n)
        trial = self.exp.new_batch_trial(generator_run=generator_run)
        trial.run()
        print(f'BO trial successfully added as trial {trial.index}')

    def add_manual_trial(self, trial_df: pd.DataFrame):
        # normalize the trial
        trial_df_norm = normalize_task(trial_df)
        arm_list = []
        for i, arm in trial_df_norm.iterrows():
            arm_list.append(Arm({self.elements[i]: arm[self.elements[i]] for i in range(len(self.elements) - 1)}))
        trial = self.exp.new_batch_trial()
        trial.add_arms_and_weights(arm_list)
        trial.run()
        print(f'manual trial successfully added as trial {trial.index}')

    def make_sample(self, trial_index=None, benchmark: dict = None, operator_name=None):
        """
        Args:
            trial_index: the index of the trial to be made sample, if not specified, will use the latest trial
            benchmark: a dict of benchmark tasks, in the format of {stage_id: arm_name}. e.g. {6: '1_0'}. Note that stage_id here is a 1-d indexing of sample stage, starting from 1, in the direction of x. e.g. for strip sample stage of 5 col and 6 rows, stage_id = 6 means the first slot of the second row.
            operator_name: the name of the operator who is running the sample, if specified, will send hyperlapsed video to the operator

        """
        # if not specified, trial is the latest trial
        trial_index = max(k for k in self.exp.trials.keys()) if trial_index is None else trial_index
        trial = self.exp.trials[trial_index]
        # get sample id for this batch
        sample_starting_id = self.db.fetch_sample_starting_id()
        sample_ending_id = sample_starting_id + len(trial.arms) * self.sample_repeats - 1 + len(benchmark or [])
        print('*****IMPORTANT*****\n'
              f'The sample id for the current batch is from {sample_starting_id} to {sample_ending_id}\n'
              '*******************')
        # get tasks and start making sample
        trial_arm_name_list = list(trial.arms_by_name.keys())
        print('trial_arm_name_list', trial_arm_name_list)
        tasks = self.db.fetch_recipe_data_by_arm_name_list(arm_name_list=trial_arm_name_list, elements=self.elements)
        print('tasks:', tasks)
        benchmark_recipe = {stage_id: self.db.fetch_recipe_data_by_arm_name(arm_name=arm_name, elements=self.elements)
                            for stage_id, arm_name in benchmark.items()} if benchmark else None

        ot2_env = OT2_Env_Linux(self.exp_name, connect_ot2=self.connect_ot2) if self.ot2_env == 'linux' else OT2_Env_Win(self.exp_name, connect_ot2=self.connect_ot2)
        ot2_env.run(tasks=tasks, benchmark=benchmark_recipe, operator_name=operator_name)

    def mark_trial_complete(self, trial_index=None):
        trial_index = max(k for k in self.exp.trials.keys()) if trial_index is None else trial_index
        trial = self.exp.trials[trial_index]
        print(trial)
        trial.mark_completed()

    def abandon_arm(self, trial_index, arm_name, reason=None):
        self.exp.trials[trial_index].mark_arm_abandoned(arm_name, reason)

    def abandon_trial(self, trial_index=None, reason=None):
        trial_index = max(k for k in self.exp.trials.keys()) if trial_index is None else trial_index
        trial = self.exp.trials[trial_index]
        trial.mark_abandoned(reason)

    def update_exp_result(self, force_update_trial: list = None):
        if force_update_trial:
            for trial_index in force_update_trial:
                self.exp._data_by_trial.pop(trial_index)
        self.exp.fetch_data()

    def exp_monitor(self):
        """
        shows the current status of the experiment, note that the pred_mean and pred_std are calculated based on the
        current model, which should be different from the value stored in the recipe table on the database, which was
         logged when the trial was initially suggested and before the point was observed
        """
        df_1 = exp_to_df(self.exp)

        max_trial = max(k for k in self.exp.trials.keys())
        # do not add pred if no data is available
        try:
            df_2 = pd.concat(self.convert_trial_to_df(self.exp.trials[i]) for i in range(max_trial + 1))
        except DataRequiredError:
            df_2 = pd.concat(self.convert_trial_to_df(self.exp.trials[i], add_pred=False) for i in range(max_trial + 1))
        try:
            merged_df = pd.merge(df_2, df_1[['arm_name', self.metric_name, 'trial_status', 'generation_method']],
                                 on=['arm_name'], how='inner')
        # in the first round, there may not be metric name in df_1
        except KeyError:
            merged_df = pd.merge(df_2, df_1[['arm_name', 'trial_status']], on=['arm_name'], how='inner')
        return merged_df.sort_values(by=['trial_index', 'arm_index'], ascending=[True, True])

    @staticmethod
    def calculate_r_squared(cv_results) -> float:
        observed_values = [result.observed.data.means[0] for result in cv_results]
        predicted_values = [result.predicted.means[0] for result in cv_results]

        mean_observed = np.mean(observed_values)

        ss_res = np.sum((np.array(observed_values) - np.array(predicted_values)) ** 2)
        ss_tot = np.sum((np.array(observed_values) - mean_observed) ** 2)

        return sigfig.round(1 - ss_res / ss_tot, 4)

    # sql storage
    def save_exp_to_db(self):
        @compiles(Text, "postgresql")
        def postgresql_text(type_, compiler, **kwargs):
            return "TEXT"

        init_engine_and_session_factory(url=self.db_url)
        engine = get_engine()
        create_all_tables(engine)
        save_experiment(self.exp, config=self.sqa_config)
        engine.dispose()

    def load_exp_from_db(self):
        init_engine_and_session_factory(url=self.db_url)
        self.exp = load_experiment(experiment_name=self.exp_name, config=self.sqa_config)

    def delete_exp_on_db(self):
        init_engine_and_session_factory(url=self.db_url)
        delete_experiment(self.exp_name)

    def update_recipe_table(self, add_pred, trial_index=None):
        trial_index = max(k for k in self.exp.trials.keys()) if trial_index is None else trial_index
        trial = self.exp.trials[trial_index]
        trial_df = self.convert_trial_to_df(trial, add_pred)
        self.db.add_recipe_data(trial_df)

    def update_sample_table(self, trial_index=None, benchmark: dict = None):

        # add sample data to database if successfully made sample
        trial_index = max(k for k in self.exp.trials.keys()) if trial_index is None else trial_index
        trial = self.exp.trials[trial_index]
        recipe_list = list(trial.arms_by_name.keys())
        recipe_num = len(recipe_list)
        batch_total_sample_num = recipe_num * self.sample_repeats + len(benchmark or {})
        batch_recipe_list = []
        repeat_direction = self.ot2_run_configs['direction']

        if repeat_direction == 'y':
            batch_recipe_list += recipe_list * self.sample_repeats
        elif repeat_direction == 'x':
            for recipe in recipe_list:
                for i in range(self.sample_repeats):
                    batch_recipe_list.append(recipe)

        data_dict = {
            'trial_index': [trial_index] * batch_total_sample_num,
            'sample_batch_id': [self.db.fetch_next_sample_batch_id()] * batch_total_sample_num,
            'arm_name': batch_recipe_list + list(benchmark.values()) if benchmark else batch_recipe_list,
        }

        data_df = pd.DataFrame(data_dict)

        self.db.add_sample_data(data_df)
