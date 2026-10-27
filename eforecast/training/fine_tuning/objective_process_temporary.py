import os
import sys
import copy
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

from eforecast.training.fine_tuning.objective_abstract import ObjectiveAbstractClass

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

#
# if not sys.argv[1].startswith('tl'):
#     trial_number = int(sys.argv[1])
#     method = sys.argv[2]
#     cluster_name = sys.argv[3]
#     path_cluster = sys.argv[4]
#     gpu_id = int(sys.argv[5])
#     refit = bool(int(sys.argv[6]))
#     print(f'Get trial_number: {trial_number}, method: {method}, cluster_name: {cluster_name},\n '
#           f'path_cluster: {path_cluster},\n gpu_id: {gpu_id}, refit: {refit}')
# else:
#     trial_number = sys.argv[1]
#     method = sys.argv[2]
#     cluster_name = sys.argv[3]
#     path_cluster = sys.argv[4]
#     tl_path_weights = sys.argv[5]
#     tl_cluster_name = sys.argv[6]
#     tl_cluster_dir = sys.argv[7]
#     gpu_id = int(sys.argv[8])
#     refit = bool(int(sys.argv[9]))
#     print(f'Get trial_number: {trial_number}, method: {method}, cluster_name: {cluster_name}\n, '
#           f'path_cluster: {path_cluster},\n '
#           f'tl_path_weights: {tl_path_weights},\n '
#           f'tl_cluster_name: {tl_cluster_name},\n '
#           f'tl_cluster_dir: {tl_cluster_dir},\n '
#           f'gpu_id: {gpu_id}, refit: {refit}')

trial_number = 8
method = 'MLP'
cluster_name = 'RBF_rule_0'
path_cluster = '/media/sider/data/Dropbox/data_transfer/models/ADMIE/RejectedPower_ver0/wind/ΑΓΓΕΛΟΚΑ_ΑΓΓΕΛΟΚΑ/intra-output/model_ver2/cluster_organizer/RBF/rule_0'
gpu_id = 0
refit = bool(1)

class ObjectiveProcess(ObjectiveAbstractClass):
    def __init__(self, static_data, runtime_file, cluster_name,
                 cluster_dir, method, refit=False):
        super(ObjectiveProcess, self).__init__(static_data, runtime_file, cluster_name,
                                               cluster_dir, method, refit)

    def fit_trial(self, trial_number):
        print(f'Objective process starts for {self.cluster_name} to train trial'
              f' {trial_number} of model {self.method} ')
        print('\n')

        path_weights = os.path.join(self.cluster_dir,
                                    self.method,
                                    f'test_{trial_number}')
        params = joblib.load(os.path.join(path_weights, 'parameters.pickle'))
        params['cluster_dir'] = self.cluster_dir
        params['cluster'] = {'cluster_name': self.cluster_name, 'cluster_path': self.cluster_dir}
        params['tl_path_weights'] = path_weights.replace('ver2', 'ver1')
        joblib.dump(params, os.path.join(path_weights, 'parameters.pickle'))
        model = DeepNetwork(self.static_data, path_weights)
        model.params = copy.deepcopy(params)
        model.cluster = copy.deepcopy(params['cluster'])
        model.save()

        self.train_or_skip(trial_number, path_weights, model, gpu_id)



if __name__ == '__main__':
    static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    path_trials = os.path.join(path_cluster, method, 'trials')
    if not os.path.exists(path_trials):
        os.makedirs(path_trials)
    trials = []
    for trial in sorted(os.listdir(path_trials)):
        trials.append(joblib.load(os.path.join(path_trials, trial)))
    objective = ObjectiveProcess(static_data, 'run_net_on_gpu.py', cluster_name, path_cluster, method,
                                 refit=refit)
    objective.fit_trial(trial_number,)



