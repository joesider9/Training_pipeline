import os
import sys
import traceback
if sys.platform == 'linux':
    path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
    path_pycharm = os.path.join(*path_pycharm[:-3])
    path_pycharm = '/' + path_pycharm
else:
    path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
    path_pycharm = '/'.join(path_pycharm[:-3])

print(path_pycharm)
sys.path.append(path_pycharm)
import shutil
import joblib
import pandas as pd

from eforecast.training.command_line.objective_abstract import ObjectiveAbstractClass
from eforecast.common_utils.dataset_utils import create_image_batches

try:
    import torch
    from eforecast.deep_models.pytorch_2x.network import DeepNetwork
    print('Backend is set pytorch')
except:
    try:
        from eforecast.deep_models.tf_2x.network import DeepNetwork
        from eforecast.deep_models.tf_2x.transformers.tranformer_network import TransformerNetwork
        print('Backend is set Tensorflow 2.10')
    except:
        raise ImportError('Cannot find backend')


method = 'TIMM'
cluster_name = 'global'
path_cluster =  ('/media/smartrue/HHD1/George/models/PPC/PPC_sat_ver2/pv/case_study/multi-output/model_ver0/global')
gpu_id = 0
refit = bool(0)
#
tl_path_weights = ''
tl_cluster_name = ''
tl_cluster_dir = ''


class ObjectiveProcess(ObjectiveAbstractClass):
    def __init__(self, static_data, runtime_file, cluster_name,
                 cluster_dir, method, refit=False):
        super(ObjectiveProcess, self).__init__(static_data, runtime_file, cluster_name,
                                               cluster_dir, method, refit)

    def fit_trial(self, trial_number, trials, gpu_i, fix_grid=False, tags=None, dates=None):
        print(f'Objective process starts for {self.cluster_name} to train trial'
              f' {trial_number} of model {self.method} ')
        print('\n')

        if fix_grid:
            file_grid = os.path.join(self.cluster_dir, self.method, 'trial_df.csv')
            if os.path.exists(file_grid):
                grid = pd.read_csv(file_grid, header=0).to_dict('records')
            else:
                grid = self.grid_()
                pd.DataFrame(grid).to_csv(file_grid, index=False)
            trial = grid[trial_number]
        else:
            trial = self.get_optim_trial(trials)

        experiment_tag = trial['experiment_tag'] if 'experiment_tag' in trial.keys() \
            else self.fix_params['experiment_tag']

        experiment_params = self.init_experiment_params(trial_number, trial, experiment_tag)

        trial_structure = self.get_optim_structure(experiment_tag, trials)

        experiment_params['experiment'] = self.select_structure(trial_structure, experiment_tag,
                                                                self.static_data['experiments'][experiment_tag])
        experiment_params['inputs'] = [key for key in experiment_params['experiment'].keys() if 'output' not in key]

        if experiment_params['image_in_data']:
            use_target = self.static_data['type'] == 'image2image'
            experiment_params['sat_image_type'] = 'rgb_snow:ir039:ir108:vis006'
            experiment_params = create_image_batches(self.static_data, experiment_params,
                                                     experiment_params['batch_size'], gpu_id, use_target,
                                                     tags=tags, api='eumetview', dates=dates)


if __name__ == '__main__':
    # static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    from case_study.short_term_image.configuration.config import config
    from eforecast.init.initialize import initializer

    static_data = initializer(config())
    path_trials = os.path.join(path_cluster, method, 'trials')
    if not os.path.exists(path_trials):
        os.makedirs(path_trials)
    trials = []
    for trial in sorted(os.listdir(path_trials)):
        trials.append(joblib.load(os.path.join(path_trials, trial)))
    objective = ObjectiveProcess(static_data, 'run_net_on_gpu.py', cluster_name, path_cluster, method,
                                 refit=refit)

    fix_grid = True
    dates = pd.date_range('2023-01-02', '2023-04-14', freq='15min')
    tags = ['val', 'test', 'eval']
    for gpu_id in range(1):
        for trial_number in range(1):
            objective.fit_trial(trial_number, trials, gpu_id, fix_grid=fix_grid, tags=None, dates=dates)


