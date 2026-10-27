import sys
import os
import joblib
import copy
import importlib
import time
import pandas as pd
import numpy as np

print(os.getcwd())
sys.path.append(os.getcwd())
from eforecast.init.initialize import initializer
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.common_utils.logger import create_logger


if sys.platform == 'linux':
    BACKEND = 'command_line'  # Available options: command_line, threads
else:
    BACKEND = 'threads'  # Available options: command_line, threads

# Import the appropriate training manager based on the selected backend
if BACKEND == 'command_line':
    from eforecast.training.command_line.train_manager_for_gpu import train_on_gpus
    from eforecast.training.command_line.train_transfer_learning_for_gpu import train_transfer_learning_on_gpus
elif BACKEND == 'threads':
    from eforecast.training.threading.train_manager_for_gpu import train_on_gpus
    from eforecast.training.threading.train_transfer_learning_for_gpu import train_transfer_learning_on_gpus
else:
    raise ValueError('Unknown backend')

path_sys = '/home/smartrue' if not os.path.exists('/media/sider/data') else '/media/sider/data'
path_data =  os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')
path_log = os.path.join(path_sys,'Dropbox/data_transfer')
logger_name = 'fit_gpu2_models_logger' if not os.path.exists('/media/sider/data') else 'fit_gpu2_models_logger_win'
logger = create_logger(logger_name, path_log, f'{logger_name}.log', 'w')

def fit_on_gpus(static_data, cluster=None, global_method=None,
                cluster_method=None, refit=False, gpu_id=None, n_jobs=None):
    if static_data['transfer_learning']:
        train_transfer_learning_on_gpus(static_data, static_data['static_data_base'], refit=refit, gpu_id=gpu_id, n_jobs=n_jobs)
    else:
        train_on_gpus(static_data, cluster=cluster, global_method=global_method,
                  cluster_method=cluster_method, refit=refit, gpu_id=gpu_id, n_jobs=n_jobs)
    return 'Done'

def get_variables(config_temp, features):
    variables = config_temp['variables']
    new_variables = []
    for feature in features:
        tag = feature.split('_')[-1]
        for var in variables:
            if tag in var['name']:
                new_var = copy.deepcopy(var)
                new_var['name'] = var['name'].replace(tag, feature)
                new_variables.append(new_var)
    return new_variables

def get_fuzzy_variables(config_temp, park_name):
    variables = config_temp['clustering']['rbf_var_imp']
    new_variables = []
    for case in variables:
        new_vars = []
        for var in case:
            var = '_'.join([park_name, var])
            new_vars.append(var)
        new_variables.append(new_vars)
    return new_variables

def init_park(park):
    #### KEEP THE FOLLOWING LINES TILL THE EVALUATION IS DONE####
    project_name_list = ['ADMIE']
    name = '.'.join(project_name_list)
    module = importlib.import_module(f'{name}.configuration.config')
    config = getattr(module, 'config')
    configuration = config()
    config_temp = copy.deepcopy(configuration)
    df = park['info']
    #### KEEP THE FOLLOWING LINES TILL THE EVALUATION IS DONE####
    park.update(joblib.load(os.path.join(path_data, df["tag"].values[0], f'park_info.pickle')))
    config_temp['project_name'] = park['name'].replace(' ', '_')
    config_temp['filename'] = os.path.join(path_data, df["tag"].values[0], f'wind_ts.csv')
    rated = park['new_rated'] if park['rated'] < park['new_rated'] else park['rated']
    config_temp['rated'] = rated
    config_temp['variables'] = get_variables(config_temp, park['variables'])
    if any(['_'.join([park['name'], 'SWIN']) == v for v in park['variables']]):
        config_temp['clustering']['rbf_var_imp'] = get_fuzzy_variables(config_temp, park['name'])
    else:
        if 'DWIN' in park['variables'][0]:
            park_variable = ['_'.join(v.split('_')[:-1]) for v in park['variables'] if 'SWIN' in v][0]
        else:
            park_variable = '_'.join(park['variables'][0].split('_')[:-1])
        config_temp['clustering']['rbf_var_imp'] = get_fuzzy_variables(config_temp, park_variable)
    #### KEEP THE FOLLOWING LINES TILL THE EVALUATION IS DONE####
    config_temp['Evaluation_start'] = park['Evaluation_start']
    #### KEEP THE FOLLOWING LINES TILL THE EVALUATION IS DONE####
    configuration.update(config_temp)
    static_data = initializer(configuration)
    return static_data

def process_pipeline(park, safe=True, gpu_id=None, n_jobs=None):
    if safe:
        try:
            static_data = init_park(park)
            if static_data['transfer_learning']:
                static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
                static_data['static_data_base'] = static_data_base
            cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
            cluster_organizer.update_cluster_folders()
            fit_on_gpus(static_data, gpu_id=gpu_id, n_jobs=n_jobs)
            del static_data
        except Exception as e:
            print(e)
            return (park['name'], False)
    else:
        static_data = init_park(park)
        if static_data['transfer_learning']:
            static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
            static_data['static_data_base'] = static_data_base
        cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
        cluster_organizer.update_cluster_folders()
        fit_on_gpus(static_data, gpu_id=gpu_id, n_jobs=n_jobs)
        del static_data
    return (park['name'], True)

def check_if_is_done(park):
    static_data = init_park(park)
    if static_data['transfer_learning']:
        static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
        static_data['static_data_base'] = static_data_base
    cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
    cluster_organizer.update_cluster_folders()
    path_last0 = os.path.join(static_data['path_model'], 'cluster_organizer/RBF/rule_0/results_RBF_rule_0_MLP.csv')
    path_last1 = os.path.join(static_data['path_model'], 'cluster_organizer/RBF/rule_1/results_RBF_rule_1_MLP.csv')
    path_last2 = os.path.join(static_data['path_model'], 'cluster_organizer/RBF/rule_3/results_RBF_rule_2_MLP.csv')
    path_last3 = os.path.join(static_data['path_model'], 'cluster_organizer/RBF/rule_3/results_RBF_rule_3_MLP.csv')
    path_last4 = os.path.join(static_data['path_model'], 'cluster_organizer/RBF/rule_4/results_RBF_rule_4_MLP.csv')
    if all([os.path.exists(path_last0), os.path.exists(path_last1),
            os.path.exists(path_last2), os.path.exists(path_last3),
            os.path.exists(path_last4)]):
        return True
    else:
        return False

if __name__ == "__main__":

    parks = joblib.load(os.path.join(path_data, 'parks_static_info.pickle'))
    unable = [5, 61, 29, 104]
    # [(11, 'ΑΙΤΩΛΙΚΟ_ΧΙΤΣΑ'),
    #  (35, 'ΒΕΥΗ_ΑΓΝΑΝΤΙ'),
    #  (53, 'ΔΙΣΤΟΜΟ_WINDBEAM'),
    #  (65, 'ΘΗΣ_ΘΙΣΒ_ΜΕΓ.ΡΑΧΗ')]
    special_cases = [11, 35, 53, 65]
    parks = [parks[i] for i in range(len(parks)) if i not in unable + special_cases]
    if not os.path.exists('/media/sider/data'):
        parks_temp = [p for p in parks][:70]
    else:
        parks_temp = [p for p in parks][70:][::-1]
    parks_temp = [p for p in parks_temp if not check_if_is_done(p)]
    debug = False
    if debug:
        results = process_pipeline(parks_temp[0], safe=False)
    else:
        results = []
        for i, p in enumerate(parks_temp):
            if i % 2 == 0:
                continue
            start = time.time()
            print('New Park processing')
            print(p['name'])
            logger.info(f"New Park processing {p['name']}")
            print('\n')
            # if p['name'] == 'ΣΤΟΥΠΠΑΙ_ΚΑΡΑΜΠΙΛ':
            if not os.path.exists('/media/sider/data'):
                gpu_id = 1
                n_jobs = 5
            else:
                gpu_id = 0
                n_jobs = 5
            results.append(process_pipeline(p, safe=False, gpu_id=gpu_id, n_jobs=n_jobs))
            end = time.time()
            logger.info(f'It took {(end - start) / 3600} hours')
            logger.info('\n')
