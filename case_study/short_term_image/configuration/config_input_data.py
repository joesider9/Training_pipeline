import os
import numpy as np
from case_study.short_term_image.configuration.config_project import config_project
from case_study.short_term_image.configuration.config_utils import *

static_data = config_project()
path_owner = os.path.join(static_data['sys_folder'], static_data['project_owner'])
path_data = os.path.join(path_owner, f"{static_data['projects_group']}_ver{static_data['version_group']}", 'DATA')

NWP_MODELS = static_data['NWP']
NWP = NWP_MODELS is None

TYPE = static_data['type']
ts_resolution = 0.25 if static_data['ts_resolution'] == '15min' else 1

NWP_DATA_MERGE = [None]  # 'all', 'by_area', 'by_area_variable',#! THIS IS NOT EMPTY. SET [None] INSTEAD
# 'by_variable',#! THIS IS NOT EMPTY. SET [None] INSTEAD
# by_nwp_provider
#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_COMPRESS = [None]  # dense or semi_full or full or load#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_IMG = ['eumetview_ir039_ir108_vis006:eumetview_rgb_snow']#! THIS IS NOT EMPTY. SET [None] INSTEAD

DATA_IMG_SCALE = ['minmax'] #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_NWP_SCALE = [None] #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_ROW_SCALE = ['minmax'] #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
#! THIS IS NOT EMPTY. SET [None] INSTEAD
DATA_TARGET_SCALE = 'maxabs' #'minmax', 'standard', 'maxabs'#! THIS IS NOT EMPTY. SET [None] INSTEAD
#! THIS IS NOT EMPTY. SET [None] INSTEAD
USE_DATA_BEFORE_AND_AFTER_TARGET = True

REMOVE_NIGHT_HOURS = True

USE_POLARS = False

HORIZON = static_data['horizon']
HORIZON_TYPE = static_data['horizon_type']

## TRANSFORMER FEATURES

GLOBAL_PAST_LAGS = [-int(i) for i in range(1, int(3 / ts_resolution))]
GLOBAL_FUTURE_LAGS = [int(i) for i in range(int(HORIZON / ts_resolution))]

TIME_MERGE_VARIABLES = {}
if HORIZON_TYPE == 'multi-output':
    targ_lags = [int(i) for i in range(int(HORIZON / ts_resolution))]
else:
    targ_lags = [0]
targ_tag = 'Step' if ts_resolution == 0.25 else 'Hour'

TARGET_VARIABLE = {'name': 'case_study',
                   'source': 'case_study',
                   'lags': targ_lags if HORIZON_TYPE == 'multi-output' else [0],
                   'columns': [f'{targ_tag}_{i}' for i in targ_lags]
                                                       if HORIZON_TYPE == 'multi-output'else ['target'],
                   'transformer': ['pv_clear_sky'],
                   'transformer_params': {'pv_clear_sky': 0.33}
}
## LAGs for NWP and Images are hourly steps

def variables():
    if TYPE == 'pv':
        # Labels for NWP variables: Flux, Cloud, Temperature
        sat_inputs = [
            variable_wrapper('ir039_ir108_vis006', bands=[],
                             input_type='image', source='eumetview',
                             lags=[-0.5, -1, -1.5, -2, -3.5],
                             transformer=[ 'area_adjust', 'resize'],
                             transformer_params={'resize': [224, 224],
                                                 'area_adjust':[250]
                                                 }
                             ),
            variable_wrapper('rgb_snow', bands=[],
                             input_type='image', source='eumetview',
                             lags=[-0.5, -1, -1.5, -2, -3.5],
                             transformer=['area_adjust', 'resize'],
                             transformer_params={'resize': [224, 224],
                                                 'area_adjust': [250]
                                                 }
                             ),
            variable_wrapper('RBG', bands=[f'channel_{i}' for i in [3, 2, 1]],
                             input_type='image', source='eumetdac',
                             lags=[-0.5, -1, -1.5, -2, -3.5],
                             transformer=['fillnan', 'alignment', 'resize', 'inverse', 'crop', 'normalize', 'img_clear_sky'],
                             transformer_params={'resize': [211, 310], 'fillnan': -1,
                                                 'alignment': [(62, 11), (0, 211), (220, 200), (310, 0)],
                                                 'crop':[[11, 121], [49, 199]],
                                                 'normalize': [13, 22, 18]
                                                 }
                             ),
            # variable_wrapper('Infrared', bands=[f'channel_{i}' for i in [4, 7, 8]],
            #                  input_type='image', source='eumetdac',
            #                  lags=[-0.5, -2, -3.5],
            #                  transformer=['fillnan', 'alignment', 'resize', 'inverse', 'crop', 'sum', 'normalize'],
            #                  transformer_params={'resize': [211, 310], 'fillnan': -1,
            #                                      'alignment': [(62, 11), (0, 211), (220, 200), (310, 0)],
            #                                      'crop':[[11, 121], [49, 199]]
            #                                      }
            #                  ),
        ]
        variable_list = sat_inputs + [
            # variable_wrapper('Flux', nwp_provider='ALL', transformer='clear_sky'),
            # variable_wrapper('Cloud', nwp_provider='ecmwf'),
            variable_wrapper('hour', input_type='calendar', source='index', lags=[0]),
            variable_wrapper('month', input_type='calendar', source='index', lags=[0])
        ]
        if HORIZON > 0:
            var_obs = variable_wrapper('Obs', input_type='timeseries', source='target',
                                       lags=[-1, -2, -3, -4, -8, -10, -12],
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
            var_obs = variable_wrapper('Obs_mean-1', input_type='timeseries', source='target',
                                       transformer='mean',
                                       lags=[-5, -6, -7, -8],
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
            var_obs = variable_wrapper('Obs_mean-2', input_type='timeseries', source='target',
                                       transformer='mean',
                                       lags=[-9, -10, -12, -11],
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
            var_obs = variable_wrapper('Obs_mean-3', input_type='timeseries', source='target',
                                       transformer='mean',
                                       lags=[-12, -13, -14, -15],
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
            var_obs = variable_wrapper('Obs_mean-4', input_type='timeseries', source='target',
                                       transformer='mean',
                                       lags=[-16, -17, -18, -19],
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
    elif TYPE == 'wind':
        # Labels for NWP variables: Uwind, Vwind, WS, WD, Temperature
        variable_list = [
            variable_wrapper('WS', nwp_provider='ALL'),
            variable_wrapper('WD', nwp_provider='ecmwf'),
            variable_wrapper('hour', input_type='calendar', source='index'),
            variable_wrapper('month', input_type='calendar', source='index')
        ]
        if HORIZON > 0:
            var_obs = variable_wrapper('Obs', input_type='timeseries', source='target', lags=3,
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
    elif TYPE == 'FA':
        lags1 = [-int(i) for i in range(1, 7)]
        lags2 = [-int(i) for i in range(7, 11)] + [-int(i) for i in range(14, 16)] + [-int(i) for i in range(21, 23)]

        lags_pred = [int(i) for i in range(2)]
        variable_list = [
            variable_wrapper('Final/Ζητούμενο', input_type='timeseries', source='target', lags=lags2,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Athens_24', input_type='timeseries', source='target', lags=lags1,
                             timezone=static_data['local_timezone'], use_diff_between_lags=True),
            # variable_wrapper('Athens_6', input_type='timeseries', source='target',
            #                  timezone=static_data['local_timezone']),
            variable_wrapper('temp_max', input_type='timeseries', source='target', lags=lags_pred + lags1,
                             timezone=static_data['local_timezone']),
            variable_wrapper('temp_min', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('temp_mean', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('rh', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('precip', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('hdd_h', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('hdd_h2', input_type='timeseries', source='target', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp_max', input_type='timeseries', source='nwp_dataset',
                             lags=[1, 0, -1, -2, -3],
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp_min', input_type='timeseries', source='nwp_dataset',
                             lags=[1, 0, -1, -2, -3],
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temperature', nwp_provider='ALL'),
            variable_wrapper('Cloud', nwp_provider='ALL'),
            variable_wrapper('WS', nwp_provider='ALL'),
            variable_wrapper('WD', nwp_provider='ALL'),
            variable_wrapper('dayweek', input_type='calendar', source='index', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('sp_index', input_type='calendar', source='index', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            variable_wrapper('month', input_type='calendar', source='index', lags=lags_pred,
                             timezone=static_data['local_timezone']),
            # variable_wrapper('dayofyear', input_type='calendar', source='index', lags=lags_pred,
            #                  timezone=static_data['local_timezone']),
        ]
    elif TYPE == 'load':
        if HORIZON > 0:
            lags = [-int(i) for i in range(1, 13)] + [-int(i) for i in range(22, 28)] + [-int(i) for i in range(47, 53)] + \
                   [-int(i) for i in range(166, 176)] + [-192]

            lags_days = [-int(24 * i)  for i in range(0, 8)]
        else:
            if HORIZON_TYPE == 'day-ahead':
                lags = [-int(i) for i in range(48, 60)] + [-int(i) for i in range(72, 77)] + [-int(i) for i in range(96, 100)] + \
                       [-int(i) for i in range(120, 122)] + [-int(i) for i in range(144, 146)] + [-int(i) for i in range(166, 176)] + \
                       [-int(i) for i in range(190, 192)] + [-216]  # + ['last_year_lags']
            else:
                lags = [-int(i) for i in range(24, 36)] + [-int(i) for i in range(48, 54)] + [-int(i) for i in range(72, 77)] + \
                       [-int(i) for i in range(96, 100)] + \
                       [-int(i) for i in range(120, 122)] + [-int(i) for i in range(144, 146)] + [-int(i) for i in range(166, 176)] + \
                       [-int(i) for i in range(190, 192)] + [-216]  # + ['last_year_lags']

            lags_days = [-int(24 * i)  for i in range(0, 8)]

        variable_list = [
            variable_wrapper('load', input_type='timeseries', source='target', lags=lags,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp_max', input_type='timeseries', source='nwp_dataset', lags=lags_days,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp', input_type='timeseries', source='nwp_dataset',
                             lags=[0, -1, -2, -3, -24, -48, -168],
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temperature', nwp_provider='ALL'),
            variable_wrapper('Cloud', nwp_provider='ALL'),
            variable_wrapper('WS', nwp_provider='ALL'),
            variable_wrapper('WD', nwp_provider='ALL'),
            variable_wrapper('dayweek', input_type='calendar', source='index',
                             timezone=static_data['local_timezone']),
            variable_wrapper('sp_index', input_type='calendar', source='index', lags=lags_days,
                             timezone=static_data['local_timezone']),
            variable_wrapper('hour', input_type='calendar', source='index',
                             timezone=static_data['local_timezone']),
            variable_wrapper('month', input_type='calendar', source='index',
                             timezone=static_data['local_timezone'])
        ]
    else:
        raise NotImplementedError(f'Define variables for type {TYPE}')
    return variable_list


def variable_wrapper(name, input_type='nwp', source='grib', lags=None, timezone='UTC', nwp_provider=None,
                     transformer=None, transformer_params=None, bands=None, use_diff_between_lags=False):
    if nwp_provider is not None:
        if nwp_provider == 'ALL':
            providers = [nwp_model['model'] for nwp_model in NWP_MODELS]
        else:
            providers = [nwp_model['model'] for nwp_model in NWP_MODELS if nwp_model['model'] == nwp_provider]
    else:
        providers = None

    return {'name': name,
            'type': input_type,  # nwp or timeseries or calendar
            'source': source,  # use 'target' for the main timeseries otherwise 'grib', 'database' for nwps,
            # 'nwp_dataset' to get data from created nwp dataset,
            # a column label of input file csv or a csv file extra, 'index' for calendar variables,
            # 'astral' for zenith, azimuth
            'lags': define_variable_lags(name, input_type, lags),
            'timezone': timezone,
            'transformer': transformer,
            'transformer_params': transformer_params,
            'bands': bands,
            'nwp_provider': providers,
            'use_diff_between_lags': use_diff_between_lags
            }


def define_variable_lags(name, input_type, lags):
    if lags is None or lags == 0:
        lags = [0] if HORIZON_TYPE != 'multi-output' else [int(i) for i in range(int(HORIZON / ts_resolution))]
    elif isinstance(lags, int):
        lags = [-int(i) for i in range(int(lags / ts_resolution))]
    elif isinstance(lags, list):
        pass
    else:
        raise ValueError(f'lags should be None or int or list')
    if name in {'Flux', 'wind'}:
        if USE_DATA_BEFORE_AND_AFTER_TARGET:
            if HORIZON == 0:
                max_lag = np.max(lags)
                min_lag = np.min(lags)
                lags = [min_lag - 1] + lags + [max_lag + 1]
    return lags


def config_data():
    static_input_data = {'nwp_data_merge': NWP_DATA_MERGE,
                         'compress_data': DATA_COMPRESS,
                         'img_data': DATA_IMG,
                         'use_data_before_and_after_target': USE_DATA_BEFORE_AND_AFTER_TARGET,
                         'remove_night_hours': REMOVE_NIGHT_HOURS,
                         'variables': variables(),
                         'target_variable': TARGET_VARIABLE,
                         'time_merge_variables': TIME_MERGE_VARIABLES,
                         'global_past_lags': GLOBAL_PAST_LAGS,
                         'global_future_lags': GLOBAL_FUTURE_LAGS,
                         'scale_row_method': DATA_ROW_SCALE,
                         'scale_img_method': DATA_IMG_SCALE,
                         'scale_nwp_method': DATA_NWP_SCALE,
                         'scale_target_method': DATA_TARGET_SCALE,
                         'use_polars': USE_POLARS
                         }
    return static_input_data
