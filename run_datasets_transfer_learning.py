import sys
import os
import joblib
import importlib
import numpy as np
import pandas as pd
from sqlalchemy.testing.suite.test_reflection import metadata

print(os.getcwd())
sys.path.append(os.getcwd())
from eforecast.init.initialize import initializer
from eforecast.nwp_extraction.nwp_extractor import NwpExtractor

from eforecast.datasets.dataset_creator import DatasetCreator
from eforecast.datasets.data_preprocessing.data_pipeline import DataPipeline
from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.datasets.data_preprocessing.data_split import Splitter
from eforecast.feature_selection.feature_selection_fit import FeatureSelector

import traceback
from eforecast.common_utils.train_utils import send_predictions

RECREATE_DATASETS = False

def nwp_extraction(static_data, dataset):
    nwp_extractor = NwpExtractor(static_data, recreate=RECREATE_DATASETS, is_online=False,
                                 dates=dataset.dates)
    nwp_extractor.extract()

#
# def add_lag_dates(dates_base, dates):
#     if dates_base.max() > dates.min():
#         diff_date = (dates_base.max() - dates.min()) - (dates[1] - dates[0])
#         dates_base = dates_base - diff_date
#     return dates_base
#
# def merge_datasets(static_data, dataset, static_data_base):
#     if static_data['transfer_learning_merge_data']:
#         dataset_base = DatasetCreator(static_data_base, train=True)
#         if static_data['NWP'] is not None:
#             nwp_data_base = dataset_base.files_manager.check_if_exists_nwp_data(get_all=True)
#             nwp_data = dataset.files_manager.check_if_exists_nwp_data(get_all=True)
#             area_base = list(nwp_data_base.keys())[0]
#             area = list(nwp_data.keys())[0]
#             for variable, var_data in nwp_data[area].items():
#                 for vendor, nwp_provide_data in var_data.items():
#                     nwp_data[area][variable][vendor]['data'] = np.concatenate(
#                         [nwp_data_base[area_base][variable][vendor]['data'], nwp_provide_data['data']], axis=0)
#                     dates_base = nwp_data_base[area_base][variable][vendor]['dates']
#                     dates = nwp_provide_data['dates']
#                     dates_base = add_lag_dates(dates_base, dates)
#                     nwp_data[area][variable][vendor]['dates'] = dates_base.append(dates)
#             dataset.files_manager.save_nwps(nwp_data)
#         if static_data['use_image']:
#             image_data_base = dataset_base.files_manager.check_if_exists_image_data()
#             image_data = dataset.files_manager.check_if_exists_image_data()
#             image_data_base = add_lag_dates(image_data_base, image_data)
#             dataset.files_manager.save_images(image_data_base)
#         row_data_base = dataset_base.files_manager.check_if_exists_row_data()
#         row_data = dataset.files_manager.check_if_exists_row_data()
#         row_obs_base = row_data_base['row_obs']
#         row_obs = row_data['row_obs']
#         if row_obs_base is not None and row_obs is not None:
#             row_obs_base.index = add_lag_dates(row_obs_base.index, row_obs.index)
#             row_data['row_obs'] = pd.concat([row_obs_base, row_obs], axis=0)
#         calendar_base = row_data_base['calendar']
#         calendar = row_data['calendar']
#         if calendar_base is not None and calendar is not None:
#             calendar_base.index = add_lag_dates(calendar_base.index, calendar.index)
#             row_data['calendar'] = pd.concat([calendar_base, calendar], axis=0)
#         dataset.files_manager.save_row_data(row_data)
#
#         lstm_data_base = dataset_base.files_manager.check_if_exists_lstm_data()
#         lstm_data = dataset.files_manager.check_if_exists_lstm_data()
#         data_base = lstm_data_base['data']
#         metadata_base = lstm_data_base['metadata']
#         data = lstm_data['data']
#         metadata = lstm_data['metadata']
#         metadata_base['dates'] = add_lag_dates(metadata_base['dates'], metadata['dates'])
#         metadata['dates'] = metadata_base['dates'].append(metadata['dates'])
#         for key, value in data.items():
#             for key1, value1 in value.items():
#                 value = data_base[key][key1]
#                 value.index = add_lag_dates(value.index, value1.index)
#                 data[key][key1] = pd.concat([value, value1], axis=0)
#         dataset.files_manager.save_lstm_data(data, metadata)
#
#         target_base = dataset_base.files_manager.check_if_exists_target_data()
#         target = dataset.files_manager.check_if_exists_target_data()
#         target_base.index = add_lag_dates(target_base.index, target.index)
#         target = pd.concat([target_base, target], axis=0)
#         dataset.files_manager.save_target(target)


def create_datasets(static_data, dataset, static_data_base):
    if static_data['NWP'] is not None:
        dataset.create_nwp_dataset(parallel=True)
    if static_data['use_image']:
        dataset.create_image_dataset()
    dataset.create_row_datasets()
    dataset.create_lstm_dataset()
    dataset.create_target()
    # TODO merge_datasets(static_data, dataset, static_data_base) IT MIGHT NOT NECESSARY

def preprocess_data(static_data):
    pipeline = DataPipeline(static_data, recreate=True, online=False, train=True)
    pipeline.fit_pipe()
    splitter = Splitter(static_data, is_online=False, train=True)
    splitter.split(refit=RECREATE_DATASETS)

def clustering_fit(static_data, static_data_base):
    cluster_organizer_base = ClusterOrganizer(static_data_base, is_online=False, train=True, refit=False)
    paths, fuzzy_file = cluster_organizer_base.get_fuzzy_models()
    cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
    cluster_organizer.copy_fuzzy_models(paths, fuzzy_file)
    cluster_organizer.create_clusters_and_cvs()


def feature_selection(static_data, static_data_base):
    feature_selector = FeatureSelector(static_data, recreate=RECREATE_DATASETS, online=False, train=True)
    feature_selector.transfer_learning(static_data_base)


if __name__ == '__main__':
    for park in ['pv_16005', 'pv_24792', 'pv_27533']:
        for hor in ['day_ahead', 'intra_day']:

            module = importlib.import_module(f'{park}.{hor}.configuration.config')
            config = getattr(module, 'config')

            # Now you can use the config function for each park
            static_data = initializer(config())
            static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
            dataset = DatasetCreator(static_data, recreate=False, train=True)
            nwp_extraction(static_data, dataset)
            create_datasets(static_data, dataset, static_data_base)
            preprocess_data(static_data)
            clustering_fit(static_data, static_data_base)
            feature_selection(static_data, static_data_base)
