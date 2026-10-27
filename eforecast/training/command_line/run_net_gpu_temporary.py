import copy
import os
import sys
import pandas as pd
import numpy as np
from collections import OrderedDict
import random
import torch
path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
path_pycharm = os.path.join(*path_pycharm[:-3])
if sys.platform == 'linux':
    path_pycharm = '/' + path_pycharm
print(path_pycharm)
sys.path.append(path_pycharm)

import joblib
def seed_everything(seed=1029):
    '''
    Set the seed for reproducibility.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


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

# path_weights = sys.argv[1]
# gpu_id = int(sys.argv[2])
# refit = bool(sys.argv[3])

if not os.path.exists('/media/sider/data'):
    path_weights = ('/media/smartrue/HHD1/George/models/METLEN/Forecasting_ver0/wind/PETALAS/intra-ahead/model_ver0/global/CNN/exp')
    gpu_id = 1
else:
    path_weights = ('/media/sider/data/models/METLEN/Forecasting_ver0/wind/PETALAS/day-ahead/model_ver0/global/CNN/exp')
    gpu_id = 0
refit = bool(1)


class Trainer:
    def __init__(self, static_data, params, path_weights, refit):
        seed_everything(1048)
        self.params = params
        self.static_data = static_data
        self.method = self.params['method']
        self.path_weights = path_weights
        self.refit = refit
        self.cluster_name = self.params['name']
        self.cluster_dir = self.params['cluster_dir']
        self.use_image = self.static_data['use_image']


    def train(self, gpu_id):
        cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))
        if self.refit:
            for f in os.listdir(self.path_weights):
                if 'checkpoint' in f or 'net_weights' in f:
                    os.remove(os.path.join(self.path_weights, f))
        self.model = DeepNetwork(self.static_data, self.path_weights, self.params, train=True, refit=self.refit)
        self.model.fit(cv_masks, gpu_id=gpu_id)

    def evaluate(self, id):
        from eforecast.datasets.data_feeder import DataFeeder
        from eforecast.common_utils.eval_utils import compute_metrics
        res = []
        self.model = DeepNetwork(self.static_data, self.path_weights, train=False)
        pred = self.model.predict(cuda=None)
        data_feeder = DataFeeder(self.static_data, train=False)
        y = data_feeder.feed_target(inverse=False)
        dates = pred.index.intersection(y.index)
        res_eval = compute_metrics(pred.loc[dates], y.loc[dates], 1,
                                   'main')['mae'].to_frame()
        res_train =pd.DataFrame({'mae_val': self.model.best_mae_val, 'mae_test': self.model.best_mae_test}, index=['main'])
        res_eval = pd.concat([res_eval, 100*res_train], axis=1)
        res.append(res_eval)
        print(res_eval)
        for file_weights in os.listdir(self.path_weights):
            if 'checkpoint' in file_weights and not 'mae' in file_weights:
                best_weights = torch.load(os.path.join(self.path_weights, file_weights), weights_only=False,
                           map_location=torch.device('cpu'))
                self.model.best_weights = best_weights
                pred = self.model.predict(cuda=None)
                res_eval = compute_metrics(pred.loc[dates], y.loc[dates], 1,
                                           file_weights)['mae'].to_frame()
                res_train = pd.DataFrame(joblib.load(os.path.join(self.path_weights,
                                                                  file_weights.replace('checkpoint',
                                                                                       'checkpoint_mae'))), index=[file_weights])
                res.append(pd.concat([res_eval, 100*res_train], axis=1))
                print(res_eval)
        res = pd.concat(res)
        res['average'] = res.mean(axis=1)
        best_weights_file = res['average'].idxmin()
        for file_weights in os.listdir(self.path_weights):
            if 'checkpoint' in file_weights and file_weights != best_weights_file:
                os.remove(os.path.join(self.path_weights, file_weights))
        if best_weights_file == 'main':
            pass
        else:
            self.model.best_weights = torch.load(os.path.join(self.path_weights, best_weights_file),
                                                 weights_only=False,
                                                map_location=torch.device('cpu'))
            os.remove(os.path.join(self.path_weights, best_weights_file))
            self.model.save()
        return res.loc[best_weights_file]


if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    path_cluster = os.path.dirname(os.path.dirname(path_weights))
    static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    # for k, v in static_data.items():
    #     if isinstance(v, str):
    #         if '/home/smartrue/Dropbox' in v:
    #             static_data[k] = v.replace('/home/smartrue/Dropbox', '/media/sider/data/Dropbox')
    #         elif '/media/smartrue/HHD1/George/models/' in v:
    #             static_data[k] = v.replace('/media/smartrue/HHD1/George/models/', '/media/sider/data/Dropbox/data_transfer/models/')

    params = joblib.load(os.path.join(path_weights, 'parameters.pickle'))
    # trainer = Trainer(static_data, params, path_weights, True)
    # trainer.train(gpu_id)
    # res = trainer.evaluate()
    # joblib.dump(res, os.path.join(path_weights, f'results_eval.pickle'))


    id = 2
    params['data_types'] = {
                  'nwp': {
                             'scale_nwp_method': 'minmax',
                             'merge': 'all',
                             },
                    'row_nwp': {
                        'scale_nwp_method': 'minmax',
                        'scale_row_method': 'minmax',
                        'compress': 'full',
                        'merge': 'all',
                    },
                    'row_calendar': {
                        'scale_nwp_method': 'minmax',
                        'scale_row_method': 'minmax',
                        'compress': 'full',
                        'merge': 'all',
                    },
                        }

    experiments = [
                        {
                            'nwp': [('cnn_att', 2),
                                    ('conv', 2),
                                    ('conv', 2),
                                     ('Flatten', []),
                                    ('dense', 128),
                                     ],
                            'row_nwp': [('dense', 512),
                                        ('dense', 128)
                                        ],
                            'row_calendar': [('dense', 64),
                                             ('dense', 128),
                                             ],
                            'output': [('cross_attention_both', 1),
                                       ('dense', 256),
                                       ('dense', 64)
                                       ]
                        },
        {

            'nwp': [('conv', 2),
                    ('Flatten', []),
                    ('dense', 512),
                    ('dense', 64),
                    ],
            'row_nwp': [('dense', 256),
                        ('dense', 64)
                        ],
            'row_calendar': [('dense', 64),
                             ],
            'output': [('cross_attention_both', 1),
                       ('dense', 32)
                       ],
        }

    ]
    params['inputs'] = [i for i in params['experiment'].keys() if 'output' not in i]

    params['max_iterations'] = 500
    params['cluster_dir'] = path_cluster
    params['cluster']['cluster_path'] = path_cluster
    params['feature_selection_method'] = None

    # ── Select which variant to use here ────────────────────────────────────
    # Change the string to switch: 'soft_margin' | 'decoupled' | 'barlow'
    for clip in {True}:
        for id, exp in enumerate(experiments):
            for clip_loss_mode in {'soft_margin', 'decoupled', 'barlow'}:
                params['clip_loss_mode'] = clip_loss_mode
                for optimizer in {'adamw', 'prodigy'}:
                    params['optimizer'] = optimizer
                    params['clip'] = clip
                    if id == 1:
                        tag = f'{optimizer}_{clip_loss_mode}_cnn_simple6_clip' if clip else f'{optimizer}_{clip_loss_mode}_cnn_simple6'
                        if os.path.exists(os.path.join(path_weights, f'results_{tag}.csv')):
                            continue
                        params['experiment'] = exp
                        trainer = Trainer(static_data, params, path_weights, True)
                        trainer.train(gpu_id)
                        res = trainer.evaluate(id)

                        res.to_csv(os.path.join(path_weights, f'results_{tag}.csv'))
                    else:
                        params['experiment'] = exp
                        att_methods = {'cbam', 'att'}
                        for id_att, att in enumerate(att_methods):
                            tag = f'{optimizer}_{clip_loss_mode}_{att}6_clip' if clip else f'{optimizer}_{clip_loss_mode}_{att}6'
                            if os.path.exists(os.path.join(path_weights, f'results_{tag}.csv')):
                                continue
                            params['attention'] = att
                            trainer = Trainer(static_data, params, path_weights, True)
                            trainer.train(gpu_id)
                            res = trainer.evaluate(id)

                            res.to_csv(os.path.join(path_weights, f'results_{tag}.csv'))

