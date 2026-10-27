import numpy as np
import pandas as pd
import os

import timm

from case_study.short_term_image.configuration.config_utils import define_n_jobs
from case_study.short_term_image.configuration.config_utils import find_pycharm_path
from case_study.short_term_image.configuration.config_utils import define_enviroment
from case_study.short_term_image.configuration.config_project import TYPE
from case_study.short_term_image.configuration.config_project import HORIZON_TYPE
from case_study.short_term_image.configuration.config_input_data import NWP_DATA_MERGE
from case_study.short_term_image.configuration.config_input_data import DATA_COMPRESS
from case_study.short_term_image.configuration.config_input_data import DATA_ROW_SCALE
from case_study.short_term_image.configuration.config_input_data import DATA_NWP_SCALE
from case_study.short_term_image.configuration.config_input_data import DATA_IMG_SCALE
from case_study.short_term_image.configuration.config_input_data import DATA_IMG
from case_study.short_term_image.configuration.experiments import experiments

RUNTIME_BACKEND = 'TORCH'  # TF_2, TORCH

from case_study.short_term_image.configuration.timm_model_list import timm_models
from case_study.short_term_image.configuration.huggingface_vit_list import huggingface_models

vit_models = huggingface_models.union(timm_models)

Transformer_models = {'Autoformer', 'Crossformer', 'DLinear', 'FEDformer', 'FiLM', 'Informer', 'LightTS', 'PatchTST',
                      'Reformer'}
ENV_NAME, PATH_ENV = define_enviroment(RUNTIME_BACKEND)

METHODS_CPU = {'RF': False,
               'CatBoost': False,
               'lightgbm': False,
               'lasso': False,
               'RBFols': False,
               'GA_RBFols': False,
               }

CLUSTER_METHODS = {'CNN': False,
                   'TIMM': True,
                   'LSTM': False,
                   'TRANSFORMER': False,
                   'MLP': True,
                   }

GLOBAL_METHODS = {'CrossViVit': False,
                  'CNN': False,
                  'TIMM': True,
                  'TRANSFORMER': False,
                  'Distributed': False,
                  'LSTM': False,
                  }

TRANSFER_LEARNING = False
TRANSFER_LEARNING_FROM = {'configuration': '/path/to/model'}
TRANSFER_LEARNING_MERGE_DATA = True


FEATURE_SELECTION_METHODS = [None, 'ShapValues']
                                             # Could be None, 'Leshy_native','Leshy_pimp','Leshy_shap',
                                             # 'BoostAGroota_native','BoostAGroota_pimp','BoostAGroota_shap',
                                             #, 'lasso', 'boruta', 'ShapValues'

combine_methods = ['kmeans']  # 'bcp', 'elastic_net', 'kmeans', 'CatBoost'

ACCEPTABLE_PERFORMANCE = 2

DATA_TYPE_TO_CLUSTER_COMBINE = {'scale_row_method': DATA_ROW_SCALE[0],
                                'scale_nwp_method': DATA_NWP_SCALE[0],
                                'scale_img_method': DATA_IMG_SCALE[0],
                                'merge': NWP_DATA_MERGE[0],
                                'compress': DATA_COMPRESS[0],
                                'image_data_type': DATA_IMG[0]
                                }


def rbf_variables():
    if TYPE == 'pv':
        var_imp = [['hour', 'q50_rgb_snow_250_-0.5', 'q50_ir039_250_-0.5']]
    elif TYPE == 'wind':
        var_imp = [['Uwind_0_ecmwf_center', 'Vwind_0_ecmwf_center','Uwind10_0_ecmwf_center', 'Vwind10_0_ecmwf_center']]
    elif TYPE == 'load':
        var_imp = [['sp_index', 'hour', 'month', 'Temp']]
    elif TYPE == 'FA':
        var_imp = [['sp_index', 'month', 'temp_max', 'dayweek']]
    else:
        var_imp = [[]]
    return var_imp


def config_methods():
    static_data = dict()
    static_data['env_name'] = ENV_NAME
    static_data['path_env'] = PATH_ENV
    static_data['backend'] = RUNTIME_BACKEND
    static_data['methods_cpu'] = METHODS_CPU
    static_data['cluster_methods'] = CLUSTER_METHODS
    static_data['global_methods'] = GLOBAL_METHODS
    static_data['transfer_learning'] = TRANSFER_LEARNING
    static_data['transfer_learning_from'] = TRANSFER_LEARNING_FROM
    static_data['transfer_learning_merge_data'] = TRANSFER_LEARNING_MERGE_DATA

    
    static_data['base_data_types'] = DATA_TYPE_TO_CLUSTER_COMBINE
    static_data['feature_selection_methods'] = FEATURE_SELECTION_METHODS
    static_data['experiments'] = experiments
    static_data['max_performance'] = ACCEPTABLE_PERFORMANCE
    static_data['val_test_ratio'] = 0.125
    static_data['keep_best_models'] = 4
    static_data['CVs'] = 5

    n_jobs = define_n_jobs()
    static_data['n_jobs'] = n_jobs['n_jobs']
    static_data['n_gpus'] = n_jobs['n_gpus']
    static_data['intra_op'] = n_jobs['intra_op']
    static_data['kernels'] = {'nwp': [[2, 2, 1], [2, 1, 2], [2, 2, 2], [1, 1, 2], [1, 2, 1], [1, 2, 2]],
                              'images': [[2, 5, 5], [2, 10, 10], [2, 40, 40], [4, 30, 40], [4, 30, 30], [2, 30, 30]]}
    static_data['CNN'] = {'experiment_tag': {'cnn1'},
                          'use_nwp': False,
                          'n_trials': 3,
                          'filters': 16,
                          'stride': 1,
                          'conv_dim': 2,
                          'batch_norm': False,
                          'use_norm': True,
                          'attention': {'cbam', 'att', 'geodatt'},
                          # 'clip_loss_mode': {None, 'decoupled', 'barlow'},
                          'area_adjust': None,
                          'final_image_size': None,
                          'act_func': 'gelu',
                          'sat_image_type': 'rgb_snow',
                          'create_image_dataset_real_time': True,
                          'merge': set(NWP_DATA_MERGE),
                          'compress': None,
                          'scale_row_method': set(DATA_ROW_SCALE),
                          'scale_nwp_method': set(DATA_NWP_SCALE),
                          'scale_img_method': None,
                          'image_data_type': None,
                          'feature_selection_method': None,
                          'optimizer': 'prodigy',
                          'scheduler': 'ReduceLROnPlateau',
                          'transfer_learning_from_global': True,
                          'learning_rate': 1e-4,
                          'batch_size': 128,
                          'max_iterations': 100,
                          'warming_iterations': 0,
                          'n_jobs': 3
                          }

    static_data['TRANSFORMER'] = {'experiment_tag': {'transformer1', 'transformer2', 'transformer3'},
                                  'n_trials': 3,
                                  'model_name': {},
                                  'transfer_learning_from_global': True,
                                  # 'clip_loss_mode': {None}, # 'decoupled', 'barlow'
                                  'use_norm': True,
                                  'use_nwp': True,
                                  'filters': 16,
                                  'stride': 1,
                                  'conv_dim': 2,
                                  'merge': set(NWP_DATA_MERGE),
                                  'attention': 'cbam',
                                  'batch_norm': False,
                                  'compress': None,
                                  'scale_row_method': set(DATA_ROW_SCALE),
                                  'scale_nwp_method': set(DATA_NWP_SCALE),
                                  'scale_img_method': None,
                                  'image_data_type': None,
                                  'feature_selection_method': None,
                                  'act_func': 'gelu',
                                  'optimizer': 'prodigy',
                                  'scheduler': {'ReduceLROnPlateau'},
                                  'batch_size': 128,
                                  'max_iterations': 96,
                                  'warming_iterations': 0,
                                  'learning_rate': 1e-4,
                                  'n_jobs': 2
                                  }
    static_data['TIMM'] = {'experiment_tag': {'timm_net'},
                           'n_trials': 1,
                           'model_name': 'visformer_small',
                           'timm_models': ' '.join(timm_models),
                           'pretrained': {'tune'},  # 'initial', 'tune', 'finetune' better 'tune
                           'sat_image_type': None,
                           'clip_value':False,
                           'final_image_size': 160,
                           'transfer_learning_from_global': True,
                           'optimizer': 'prodigy',
                           'scheduler': 'ReduceLROnPlateau',
                           'fix_grid_search': False,
                           'merge': set(NWP_DATA_MERGE),
                           'compress': None,
                           'scale_row_method': set(DATA_ROW_SCALE),
                           'scale_nwp_method': set(DATA_NWP_SCALE),
                           'feature_selection_method': None,
                           'scale_img_method': None,
                           'image_data_type': None,
                           'create_image_dataset_real_time': False,
                           'spatal_coord': True,
                           'area_adjust': {'eumetdac': 50, 'eumetview': 80},
                           'use_classes': False,  # better false
                           'filters': 12,
                           'conv_dim': 2,
                           'batch_size': 128,
                           'max_iterations': 72,
                           'warming_iterations': 0,
                           'learning_rate': [0.9e-4, 1.1e-4],
                           'use_data': None,
                           # 1. all, 2. image and calendar 3. calendar and meas, 4. image
                           'act_func': 'gelu',
                           'n_jobs': 1
                           }
    static_data['LSTM'] = {'units': 24,
                           'n_trials': 3,
                           'experiment_tag': {'lstm1'},
                           # 'clip_loss_mode': {None, 'decoupled', 'barlow'},
                           # 'use_norm': True,
                           'attention': {'cbam', 'att', 'geodatt'},
                           'filters': 16,
                           'stride': 1,
                           'conv_dim': 2,
                           'use_nwp': True,
                           'batch_norm': False,
                           'transfer_learning_from_global': True,
                           'use_embedding': False,
                           'merge': set(NWP_DATA_MERGE),
                           'compress': None,
                           'scale_row_method': set(DATA_ROW_SCALE),
                           'scale_nwp_method': set(DATA_NWP_SCALE),
                           'feature_selection_method': set(FEATURE_SELECTION_METHODS),
                           'act_func': 'gelu',
                           'batch_size': 128,
                           'max_iterations': 96,
                           'warming_iterations': 5,
                           'optimizer': 'prodigy',
                           'scheduler': {'ReduceLROnPlateau'},
                           'learning_rate': 1e-4,
                           'n_jobs': 2
                           }
    static_data['MLP'] = {'experiment_tag': {'mlp1', 'mlp2', 'mlp3', 'mlp4'}, #'mlp1', 'mlp2', 'mlp3', 'mlp4',
                          'n_trials': 4,
                          'hold_prob': 1,
                          'fix_grid_search': False,
                          'use_nwp': False,
                          'batch_norm': False,
                          'transfer_learning_from_global': True,
                          'use_embedding': False,
                          'filters': 16,
                          'stride': 1,
                          'conv_dim': 2,
                          'merge': set(NWP_DATA_MERGE),
                          'compress': set(DATA_COMPRESS),
                          'scale_row_method': set(DATA_ROW_SCALE),
                          'scale_nwp_method': set(DATA_NWP_SCALE),
                          'scale_img_method': None,
                          'image_data_type': None,
                          'feature_selection_method': set(FEATURE_SELECTION_METHODS),
                          'act_func': 'gelu',
                          'batch_size': 512,
                          'max_iterations': 100,
                          'warming_iterations': 5,
                          'optimizer': 'prodigy',
                          'scheduler': 'ReduceLROnPlateau',
                          'learning_rate': 1e-4,
                          'n_jobs': 3,
                          }
    static_data['CrossViVit'] = {'experiment_tag': {'CrossViVit_net'},
                                 'n_trials': 2,
                                 'model_name': 'visformer_small',
                                 'timm_models': ' '.join(timm_models),
                                 'transformer_name': None,
                                 'sat_image_type': {#'RBG:Infrared1:Infrared2:coord',
                                                    'RBG:Infrared1:RBG_opt',
                                                    #'RBG_grey:Infrared1:Infrared2:RBG_opt'
                                 },
                                 'pretrained': {'tune'},
                                 'transfer_learning_from_global': False,
                                 'area_adjust': 24,
                                 'final_image_size': {128, 224},
                                 'create_image_dataset_real_time': True,
                                 'stride': 1,
                                 'filters': 12,
                                 'conv_dim': 3,
                                 'optimizer': 'adamw',
                                 'scheduler': {'ReduceLROnPlateau'},
                                 'merge': NWP_DATA_MERGE[0],
                                 'compress': DATA_COMPRESS[0],
                                 'scale_row_method': DATA_ROW_SCALE[0],
                                 'scale_nwp_method': DATA_NWP_SCALE[0],
                                 'scale_img_method': set(DATA_IMG_SCALE),
                                 'image_data_type': set(DATA_IMG),
                                 'feature_selection_method': None,
                                 'use_data': 3,
                                 # 1. all, 2. image and calendar 3. calendar and meas, 4. image
                                 'act_func': 'gelu',
                                 'batch_size': 128,
                                 'max_iterations': 100,
                                 'warming_iterations': 0,
                                 'learning_rate': {1e-3, 1e-5},
                                 'n_jobs': 1
                                 }
    static_data['Distributed'] = {'experiment_tag': {'distributed_cnn1', 'distributed_mlp2'},
                                  'keep_n_models': 2,
                                  'n_trials': 10,
                                  'hold_prob': 1,
                                  'batch_size': {32, 64, 128},
                                  'filters': 6,
                                  'conv_dim': 2,
                                  'transfer_learning_from_global': True,
                                  'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                                  'merge': set(NWP_DATA_MERGE),
                                  'compress': set(DATA_COMPRESS),
                                  'scale_row_method': set(DATA_ROW_SCALE),
                                  'scale_nwp_method': set(DATA_NWP_SCALE),
                                  'feature_selection_method': set(FEATURE_SELECTION_METHODS),
                                  'act_func': {'sigmoid'},
                                  'max_iterations': 800,
                                  'optimizer': {'prodigy', 'adamw'},
                                  'scheduler': {'CosineAnnealing'},
                                  'warming_iterations': 4,
                                  'learning_rate': {1e-3, 1e-3},
                                  'thres_act': 0.001,
                                  'min_samples': 200,
                                  'max_samples_ratio': 0.8,
                                  'train_schedule': 'simple',  # simple or complex
                                  'n_rules': {9},
                                  'is_fuzzy': {False},  # If True creates its own activations
                                  'clustering_method': {'RBF', None},  # None RBF
                                  'rbf_var_imp': rbf_variables()[0],
                                  # Data type used for clustering when it is Fuzzy
                                  'n_jobs': n_jobs['n_jobs_cnn_3d'],
                                  }
    CLUSTERING_METHOD = ['RBF']  # RBF
    static_data['clustering'] = {'n_jobs': n_jobs['n_jobs'],
                                 'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                                 'methods': CLUSTERING_METHOD,
                                 'clusters_for_method': 'RBF',  # RBF
                                 'prediction_for_method': ['RBF'],  # RBF
                                 'thres_act': 0.001,
                                 'rbf_var_imp': rbf_variables(),
                                 'explode_clusters': True,
                                 'n_var_lin': 24,
                                 'min_samples': 8000,
                                 'max_samples_ratio': 0.45,
                                 'warming_iterations': 2,
                                 'n_rules': 7,
                                 'params': {'experiment_tag': 'exp_fuzzy1',
                                            'n_trials': 1,
                                            'hold_prob': 1,
                                            'batch_size': 2048,
                                            'filters': 12,
                                            'conv_dim': 2,
                                            'merge': NWP_DATA_MERGE[0],
                                            'compress': DATA_COMPRESS[0],
                                            'scale_row_method': DATA_ROW_SCALE[0],
                                            'scale_nwp_method': DATA_NWP_SCALE[0],
                                            'feature_selection_method': None,
                                            'train_schedule': 'simple',  # Simple or complex
                                            'optimizer': 'adamw',
                                            'scheduler': 'CosineAnnealing',
                                            'act_func': None,
                                            'max_iterations': 1000,
                                            'warming_iterations': 2,
                                            'learning_rate': 0.25e-3,
                                            'n_jobs': 1,
                                            }
                                 }
    static_data['RF'] = {'n_trials': 40,
                         'max_depth': {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 42, 48},
                         'max_features': {'sqrt', 'log2', 0.8, 0.7, 0.5, 0.4, 0.3, 0.2},
                         'min_samples_leaf': [2, 200],
                         'min_samples_split': [2, 100],
                         'max_samples': [0.3, 1],
                         'oob_score': True
                         }
    static_data['CatBoost'] = {'n_trials': 15,
                               'iterations': 1000,
                               'learning_rate': {0.01, 0.05, 0.1},
                               'l2_leaf_reg': [5, 6],
                               "objective": {"RMSE", "MAE"} if HORIZON_TYPE != 'multi-output' else {'MultiRMSE'},
                               'min_data_in_leaf': [50, 100],
                               "colsample_bylevel": [0.8, 1],
                               "depth": [5, 9],
                               "boosting_type": "Ordered",
                               "bootstrap_type": {"Bayesian", "Bernoulli", "MVS"},
                               "eval_metric": "MAE" if HORIZON_TYPE != 'multi-output' else 'MultiRMSE',
                               }
    static_data['lightgbm'] = {'n_trials': 20,
                               'num_iterations': 1000,
                               'boosting_type':'gbdt',
                               'learning_rate': {0.01, 0.05, 0.1},
                               'reg_lambda': 0,
                               'num_leaves': [10, 100],
                               'min_child_samples': [10, 100],
                               "colsample_bytree": [0.8, 1],
                               "max_depth": {-1,5, 9, 15, 21},
                               }
    static_data['lasso'] = {'n_trials': 100,
                            'max_iter': 150000,
                            'eps': {0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2},
                            'fit_intercept': {True, False},
                            "selection": {"cyclic", "random"},
                            }
    static_data['RBFols'] = {'n_trials': 10,
                             'warming_iterations': 4,
                             'width': [0.01, 15],
                             'keep': [2, 6]
                             }
    static_data['GA_RBFols'] = {'n_trials': 1000,
                                'warming_iterations': 4,
                                'width': [0.01, 15],
                                'keep': [2, 6]
                                }
    static_data['combining'] = {'methods': combine_methods,
                                'data_type': DATA_TYPE_TO_CLUSTER_COMBINE,
                                }

    return static_data
