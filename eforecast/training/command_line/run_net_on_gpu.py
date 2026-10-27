import pandas as pd
import time
import os
import sys

path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
path_pycharm = os.path.join(*path_pycharm[:-3])
if sys.platform == 'linux':
    path_pycharm = '/' + path_pycharm
print(path_pycharm)
sys.path.append(path_pycharm)

import joblib
from eforecast.common_utils.devops_utils import is_cuda_oom

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

path_weights = sys.argv[1]
gpu_id = int(sys.argv[2])
refit = bool(sys.argv[3])


# path_weights = ('/media/smartrue/HHD1/George/models/PPC/PPC_sat_ver1/pv/Bitakos/day-ahead/model_ver0/'
#                 'cluster_organizer/RBF/rule_0/LSTM/test_tl0')
# gpu_id = 1
# refit = bool(1)


class Trainer:
    def __init__(self, static_data, params, path_weights, refit):
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
        self.model = DeepNetwork(self.static_data, self.path_weights, self.params, train=True, refit=self.refit)
        self.model.fit(cv_masks, gpu_id=gpu_id)

    def evaluate(self):
        from eforecast.datasets.data_feeder import DataFeeder
        from eforecast.common_utils.eval_utils import compute_metrics
        res = []
        self.model = DeepNetwork(self.static_data, self.path_weights, train=False)
        pred = self.model.predict()
        data_feeder = DataFeeder(self.static_data, train=False)
        y = data_feeder.feed_target(inverse=False)
        dates = pred.index.intersection(y.index)
        res_eval = compute_metrics(pred.loc[dates], y.loc[dates], 1,
                                   'main')['mae'].to_frame()
        res_train = pd.DataFrame({'mae_val': self.model.best_mae_val, 'mae_test': self.model.best_mae_test},
                                 index=['main'])
        res_eval = pd.concat([res_eval, 100 * res_train], axis=1)
        res.append(res_eval)
        print(res_eval)
        for file_weights in os.listdir(self.path_weights):
            if 'checkpoint' in file_weights and not 'mae' in file_weights:
                best_weights = torch.load(os.path.join(self.path_weights, file_weights), weights_only=False,
                                          map_location=torch.device('cpu'))
                self.model.best_weights = best_weights
                pred = self.model.predict()
                res_eval = compute_metrics(pred.loc[dates], y.loc[dates], 1,
                                           file_weights)['mae'].to_frame()
                res_train = pd.DataFrame(joblib.load(os.path.join(self.path_weights,
                                                                  file_weights.replace('checkpoint',
                                                                                       'checkpoint_mae'))),
                                         index=[file_weights])
                res.append(pd.concat([res_eval, 100 * res_train], axis=1))
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
        return {'exp': self.params['experiment'], 'mae': res.loc[best_weights_file]}


if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    path_cluster = os.path.dirname(os.path.dirname(path_weights))
    static_data = joblib.load(os.path.join(path_cluster, 'static_data.pickle'))
    # from case_study.short_term_image.configuration.config import config
    #
    # from eforecast.init.initialize import initializer
    # static_data = initializer(config())
    params = joblib.load(os.path.join(path_weights, 'parameters.pickle'))

    i = 0
    while True:
        try:
            trainer = Trainer(static_data, params, path_weights, refit)
            trainer.train(gpu_id)
            break
        except RuntimeError as e:
            if is_cuda_oom(str(e)):
                print(f'CUDA OOM error occurred. Waiting for 5 minutes... it is {i} times')
                torch.cuda.empty_cache()
                time.sleep(300)
                refit = True
                i += 1
                continue
            else:
                break
        if os.path.exists(os.path.join(path_weights, f'results_eval.pickle')):
            res = trainer.evaluate()
            joblib.dump(res, os.path.join(path_weights, f'results_eval.pickle'))
            break
        else:
            time.sleep(300)
            continue
    res = trainer.evaluate()
    joblib.dump(res, os.path.join(path_weights, f'results_eval.pickle'))



