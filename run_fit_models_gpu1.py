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
print(BACKEND)
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
print(path_sys)
path_data =  os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')
path_log = os.path.join(path_sys,'Dropbox/data_transfer')
logger_name = 'fit_gpu1_models_logger' if not os.path.exists('/media/sider/data') else 'fit_gpu1_models_logger_win'
logger = create_logger(logger_name, path_log, f'{logger_name}.log', 'w')

def fit_on_gpus(static_data, cluster=None, global_method=None,
                cluster_method=None, refit=False, gpu_id=None, n_jobs=None):
    if static_data['transfer_learning']:
        train_transfer_learning_on_gpus(static_data, static_data['static_data_base'], refit=refit, gpu_id=gpu_id, n_jobs=n_jobs)
    else:
        train_on_gpus(static_data, cluster=cluster, global_method=global_method,
                  cluster_method=cluster_method, refit=refit, gpu_id=gpu_id, n_jobs=n_jobs)
    return 'Done'

def process_pipeline(static_data, safe=True, gpu_id=None, n_jobs=None):
    if safe:
        try:
            if static_data['transfer_learning']:
                static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
                static_data['static_data_base'] = static_data_base
            cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
            cluster_organizer.update_cluster_folders()
            fit_on_gpus(static_data, gpu_id=gpu_id, n_jobs=n_jobs)
            del static_data
        except Exception as e:
            print(e)
    else:
        if static_data['transfer_learning']:
            static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
            static_data['static_data_base'] = static_data_base
        cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
        cluster_organizer.update_cluster_folders()
        fit_on_gpus(static_data, gpu_id=gpu_id, n_jobs=n_jobs)
        del static_data

if __name__ == "__main__":
    from case_study.short_term_image.configuration.config import config

    # Initialize the system with configuration settings
    static_data = initializer(config())
    process_pipeline(static_data, safe=False, gpu_id=0, n_jobs=1)
