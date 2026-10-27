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
from eforecast.datasets.dataset_creator import DatasetCreator
from run_combine_models import *

path_sys = '/home/smartrue' if not os.path.exists('/media/sider/data') else '/media/sider/data'
print(path_sys)
path_data =  os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')

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
    rated = park['new_rated']
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

def process_pipeline(park, safe=True, refit=False):
    if safe:
        try:
            static_data = init_park(park)
            if static_data['transfer_learning']:
                static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
                static_data['static_data_base'] = static_data_base
            # cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
            # cluster_organizer.update_cluster_folders()
            # predict_methods(static_data, train=True, average=False, parallel=False)
            # predict_methods(static_data, train=False, average=False, parallel=False)
            evaluate_methods(static_data)
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
        # dataset = DatasetCreator(static_data, recreate=False, train=True)
        # dataset.create_lstm_dataset_custom()
        # predict_methods(static_data, train=True, average=False, parallel=True)
        # predict_methods(static_data, train=False, average=False, parallel=True)
        # evaluate_methods(static_data)
        # predict_methods(static_data, train=True, average=True, parallel=True)
        # predict_methods(static_data, train=False, average=True, parallel=True)
        # combine_methods(static_data)
        # predict_combine_methods(static_data, train=True)
        # compute_averages_methods(static_data, train=True, only_methods=False, only_combine_methods=False)
        # predict_combine_methods(static_data, train=False)
        # compute_averages_methods(static_data, train=False, only_methods=False, only_combine_methods=False)
        evaluate_averages(static_data)
        del static_data
    return (park['name'], True)

def check_if_is_done(park):
    static_data = init_park(park)
    if static_data['transfer_learning']:
        static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
        static_data['static_data_base'] = static_data_base
    cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
    cluster_organizer.update_cluster_folders()
    path_last0 = os.path.join(static_data['path_model'], 'DATA/predictions_regressors_train.pickle')
    path_last1 = os.path.join(static_data['path_model'], 'DATA/predictions_regressors_eval.pickle')

    if os.path.exists(path_last0) and os.path.exists(path_last1):
        return True
    else:
        return False
if __name__ == "__main__":
    ##TODO 32.ΣΑΝΤΑ_ΚΥΡΤΟΝ -> FIX THE CLUSTERER
    ## Stopped at ΚΟΡΩΝΕΙΑ_ΣΤΟΑ - Full disk
    parks = joblib.load(os.path.join(path_data, 'parks_static_info.pickle'))
    unable = [5, 61, 29, 104]
    # [(11, 'ΑΙΤΩΛΙΚΟ_ΧΙΤΣΑ'),
    #  (35, 'ΒΕΥΗ_ΑΓΝΑΝΤΙ'),
    #  (53, 'ΔΙΣΤΟΜΟ_WINDBEAM'),
    #  (65, 'ΘΗΣ_ΘΙΣΒ_ΜΕΓ.ΡΑΧΗ')]
    special_cases = [11, 35, 53, 65]
    parks = [parks[i] for i in range(len(parks)) if i not in unable + special_cases]
    # if not os.path.exists('/media/sider/data'):
    #     parks_temp = [p for p in parks][:69]
    # else:
    #     parks_temp = [p for p in parks][69:][::-1]
    # parks_temp = [p for p in parks if not check_if_is_done(p)]
    parks_temp = [p for p in parks if check_if_is_done(p)]
    failed = []
    debug = False
    if debug:
        results = process_pipeline(parks_temp[0], safe=False)
    else:
        results = []
        for p in parks_temp:
            print('New Park processing')
            print(p['name'])
            print('\n')
            if p['name'] == 'ΣΙΑΤΙΣΤΑ_ΚΟΥΚΟΥΛΙ':
            # try:
                results.append(process_pipeline(p, safe=False, refit=False))
            # except:
            #     failed.append(p['name'])
            #     continue
    print(failed)

