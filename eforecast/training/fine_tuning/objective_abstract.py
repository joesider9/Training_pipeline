import os
import sys
path_pycharm = (os.path.normpath(os.path.normpath(os.path.dirname(__file__)))).split(os.sep)
path_pycharm = os.path.join(*path_pycharm[:-3])
if sys.platform == 'linux':
    path_pycharm = '/' + path_pycharm
print(path_pycharm)
sys.path.append(path_pycharm)

import time
import gc
import joblib
import copy

import numpy as np
import pandas as pd

from GPUtil import getGPUs
from eforecast.training.optimizers.optimizer_hyper import HyperoptOptimizer

from eforecast.common_utils.devops_utils import init_conda_in_exe
from eforecast.common_utils.devops_utils import run_exe
from eforecast.datasets.data_feeder import DataFeeder

class ObjectiveAbstractClass:

    def __init__(self, static_data, runtime_file, cluster_name, cluster_dir, method, refit=False):
        self.space_structure = None
        self.static_data = static_data
        self.method = method
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.refit = refit
        self.runtime_file = runtime_file
        self.warming = self.static_data[method]['warming_iterations']
        self.use_image = self.static_data['use_image']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_methods = self.static_data['scale_nwp_method']
        self.scale_row_methods = self.static_data['scale_row_method']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)


    def train_or_skip(self, trial_number, path_weights, model, gpu_id):
        params = dict()
        acc = np.inf
        start = time.time()

        while True:
            gpus = getGPUs()
            gpuUtil = gpus[gpu_id].load
            if gpuUtil < 0.9:
                break
            else:
                time.sleep(10)
        count_runs = 0
        while acc > self.static_data['max_performance'] and count_runs <= 0:
            print(f'Run test {trial_number} count_runs {count_runs}')
            try:
                self._fit(trial_number, gpu_id)
            except:
                pass
            count_runs += 1
            gpus = getGPUs()
            memory_util = gpus[gpu_id].memoryUtil
            try:
                model.load()
                acc = model.best_mae_test
            except:
                acc = np.inf
                pass
            if acc is None:
                continue
            if acc > self.static_data['max_performance']:
                continue
            elif not os.path.exists(os.path.join(path_weights, 'net_weights.pickle')):
                print('Trial aboard due to gpu memory utilization')
                print()
                model.best_weights = {}
                model.best_mae_test = np.inf
                model.best_mae_val = np.inf
                model.results = pd.DataFrame()
                model.is_trained = True
                model.save()
                print('deep model failed')
                break

            else:
                break
        print(acc)
        params['value'] = acc
        params['mae_test'] = model.best_mae_test
        params['mae_val'] = model.best_mae_val
        params['sse_test'] = model.best_sse_test
        params['sse_val'] = model.best_sse_val
        params['duration'] = time.time() - start

        columns = ['trial_number', 'duration', 'value', 'mae_test', 'mae_val', 'sse_val', 'sse_test']
        trial = {key: value for key, value in params.items() if key in columns}

        if not np.isinf(acc):
            path_trials = os.path.join(self.cluster_dir, self.method, 'trials')
            os.makedirs(path_trials, exist_ok=True)
            file_trial = f'trial{trial_number}.pickle'
            joblib.dump(trial, os.path.join(path_trials, file_trial))
        del model
        gc.collect()

    def _fit(self, test_id, gpu_id):
        if not os.path.exists('/media/sider/data'):
            python_file = f"~/{self.static_data['env_name']}/bin/python"
        else:
            python_file = 'python'
        method = self.method
        path_weights = os.path.join(self.cluster_dir, method, f'test_{test_id}')
        exe_file_name = f"exe_{self.cluster_name}_{method}_gpu_{gpu_id}_test{test_id}.sh" if sys.platform == 'linux' \
            else f"exe_{self.cluster_name}_{method}_gpu_{gpu_id}_test{test_id}.bat"
        exe_file = os.path.join(path_weights, exe_file_name)
        if sys.platform == 'linux':
            path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
            path_pycharm = os.path.join(*path_pycharm[:-2])
            path_pycharm = '/' + path_pycharm
        else:
            path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
            path_pycharm = '/'.join(path_pycharm[:-2])
        init_conda_in_exe(self.static_data, path_pycharm, [exe_file])

        with open(exe_file, mode='a') as fp:
            fp.write(f"{python_file} {os.path.dirname(__file__)}/{self.runtime_file} "
                     f"{path_weights}  "
                     f"{gpu_id} "
                     f"{1}\n")
            file_weights = os.path.join(path_weights, 'net_weights.pickle')
            fp.write(f'if [ -f {file_weights} ]; then\n')
            fp.write(f'\techo "succeed"\n')
            fp.write('else\n')
            fp.write(f'\techo "failed" > {exe_file.replace(".sh", ".txt").replace(".bat", ".txt")}\n')
            fp.write('\texit\n')
            fp.write('fi\n')
        run_exe([exe_file])

    def grid_(self,):
        from sklearn.model_selection import ParameterGrid
        space_temp = dict()
        for key, value in self.space.items():
            if value['values'] is not None:
                space_temp[key] = value['values']
            elif value['range'] is not None:
                space_temp[key] = value['range']
            elif value['type'] == 'bool':
                space_temp[key] = [True, False]
            else:
                print('space param is not defined correctly')
                space_temp[key] = None
        grid = ParameterGrid(space_temp)
        return list(grid)
