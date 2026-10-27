import os
import sys
import joblib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from eforecast.shallow_models.shallow_model import ShallowModel
from eforecast.common_utils.devops_utils import run_exe
from eforecast.common_utils.devops_utils import init_exe_files_group
from eforecast.common_utils.devops_utils import get_results
from eforecast.training.fine_tuning.run_net_gpu_temporary import path_weights

def run_tasks_group(static_data,  runtime_file, tasks, exe_files):
    if not os.path.exists('/media/sider/data'):
        python_file = f"~/{static_data['env_name']}/bin/python"
    else:
        python_file = 'python'

    if sys.platform == 'linux':
        path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
        path_pycharm = os.path.join(*path_pycharm[:-2])
        path_pycharm = '/' + path_pycharm
    else:
        path_pycharm = (os.path.normpath(os.path.dirname(__file__))).split(os.sep)
        path_pycharm = '/'.join(path_pycharm[:-2])

    for task in tasks:
        gpu_id, job_id = task['gpu_id'], task['job_id']
        exe_file_name = f"exe_{task['method']}_gpu_{gpu_id}_job{job_id}.sh" if sys.platform == 'linux' \
            else f"exe_{task['method']}_gpu_{gpu_id}_job{job_id}.bat"
        exe_file = os.path.join(static_data['path_group'], 'exe_files', exe_file_name)
        if not exe_file in exe_files:
            raise ValueError('Cannot find exefile')
        with open(exe_file, mode='a') as fp:
            path_objective = os.path.join(path_pycharm, 'training', 'fine_tuning')
            if str(task['trial_number']).startswith('tl'):
                fp.write(f"{python_file} {path_objective}/{runtime_file} "
                         f"{task['trial_number']}  "
                         f"{task['method']}  "
                         f"{task['cluster_name']}  "
                         f"{task['cluster_dir']}  "
                         f"{task['tl_path_weights']}  "
                         f"{task['tl_cluster_name']}  "
                         f"{task['tl_cluster_dir']}  "
                         f"{gpu_id} "
                         f"{task['refit']}\n")
            else:
                fp.write(f"{python_file} {path_objective}/{runtime_file} "
                         f"{task['trial_number']}  "
                         f"{task['method']}  "
                         f"{task['cluster_name']}  "
                         f"{task['cluster_dir']}  "
                         f"{gpu_id} "
                         f"{task['refit']}\n")

            file_weights = os.path.join(task['cluster_dir'], task['method'], f"test_{task['trial_number']}", 'net_weights.pickle')
            fp.write(f'if [ -f {file_weights} ]; then\n')
            fp.write(f'\techo "succeed"\n')
            fp.write('else\n')
            fp.write(f'\techo "failed" > {exe_file.replace(".sh", ".txt").replace(".bat", ".txt")}\n')
            fp.write('fi\n')
    run_exe(exe_files)

def thread_project(static_data_temp, clusters_temp, n_gpus, n_jobs, gpu_id, tasks_gpu=None, tasks_cpu=None):
    if gpu_id is not None:
        n_gpus = 1
    i = 0
    j = 0
    if tasks_gpu is None:
        tasks_gpu = []
    if tasks_cpu is None:
        tasks_cpu = []
    for cluster_name, cluster_dir in clusters_temp.items():
        joblib.dump(static_data_temp, os.path.join(cluster_dir, 'static_data.pickle'))
        methods = [d for d in os.listdir(cluster_dir) if
                   os.path.isdir(os.path.join(cluster_dir, d)) and not d.startswith('result')]
        for method in methods:
            for exp_name in os.listdir(os.path.join(cluster_dir, method)):
                if 'trial' in exp_name:
                    continue
                trial_number = int(exp_name.split('_')[-1])
                task = {'trial_number': trial_number, 'method': method,
                        'cluster_name': cluster_name, 'cluster_dir': cluster_dir,
                        'gpu_id': i % n_gpus if gpu_id is None else gpu_id,
                        'job_id': j % n_jobs,
                        'refit': 1}
                if method in static_data_temp['methods_cpu'].keys():
                    tasks_cpu.append(task)
                else:
                    tasks_gpu.append(task)
                    j += 1
                    if j % n_jobs == 0:
                        i += 1
    return tasks_gpu, tasks_cpu


def run_cpu_tasks_group(tasks_cpu):
    for task in tasks_cpu:
        static_data = joblib.load(os.path.join(task['cluster_dir'], 'static_data.pickle'))
        cv_masks = joblib.load(os.path.join(task['cluster_dir'] , 'cv_mask.pickle'))
        cv_masks = [cv_masks[i] for i in [0, 2, 1]]
        path_weights = os.path.join(task['cluster_dir'], task['method'], f'test_{str(task["trial_number"])}')
        model = ShallowModel(static_data, path_weights, n_jobs=static_data['n_jobs'], train=True)
        model.params['cluster_dir'] = task['cluster_dir']
        model.refit = True
        model.save()
        model.reset(model.params)
        model.fit(cv_masks)

def GPU_thread_projects(static_data_list, n_gpus, n_jobs, method, clusters_list, runtime_file, refit=False, gpu_id=None):
    gpu_exe_files = init_exe_files_group(static_data_list[0], list(range(n_gpus)) if gpu_id is None else [gpu_id], n_jobs, method)
    tasks_gpu, tasks_cpu = [], []
    for static_data_temp, clusters_temp in zip(static_data_list, clusters_list):
        tasks_gpu, tasks_cpu = thread_project(static_data_temp, clusters_temp, n_gpus, n_jobs, gpu_id, tasks_gpu, tasks_cpu)
    res = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        if len(tasks_gpu) > 0:
            futures.append(executor.submit(run_tasks_group, static_data_list[0], runtime_file, tasks_gpu, gpu_exe_files))
        if len(tasks_cpu) > 0:
            futures.append(executor.submit(run_cpu_tasks_group, tasks_cpu))
        for future in as_completed(futures):
            res.append(future.result())
    for clusters in clusters_list:
        best_trials = get_results(clusters, method, refit)

def train_on_gpus(static_data_list, cluster=None, global_method=None, cluster_method=None,
                  refit=False, gpu_id=None, n_jobs=None):
    print('Train Deep learning models on gpu')
    is_group = True if isinstance(static_data_list, list) else False

    static_data = static_data_list[0] if is_group else static_data_list
    if cluster_method is None:
        cluster_methods = [method for method, values in static_data['cluster_methods'].items() if values]
    else:
        print(f'Start training ONLY method {cluster_method}')
        cluster_methods = [cluster_method]

    if global_method is None:
        global_methods = [method for method, values in static_data['global_methods'].items() if values]
        global_methods += [cl_m for cl_m in cluster_methods if static_data[cl_m]['transfer_learning_from_global'] and
                                                                cl_m not in global_methods]
    else:
        print(f'Start training ONLY method {global_method}')
        global_methods = [global_method]
    n_gpus = static_data['n_gpus']
    if not is_group:
        static_data_list = [static_data_list]

    for method in global_methods:
        runtime_file = 'objective_process.py'
        global_dir = os.path.join(static_data['path_model'], 'global')
        if not os.path.exists(global_dir):
            os.makedirs(global_dir)
        clusters_list = [{'global': global_dir} for _ in range(len(static_data_list))]

        GPU_thread_projects(static_data_list, n_gpus, static_data[method]['n_jobs'] if n_jobs is None else n_jobs, method,
                   clusters_list, runtime_file, refit=refit, gpu_id=gpu_id)

        print(f'Training of {method} ends successfully')

    if static_data['is_Fuzzy']:
        clusters_list = []
        for static_data_temp in static_data_list:
            clusters = joblib.load(os.path.join(static_data_temp['path_model'], 'clusters.pickle'))
            clusters_list.append(clusters)

        for method in cluster_methods:
            runtime_file = 'objective_process.py'
            GPU_thread_projects(static_data_list, n_gpus, static_data[method]['n_jobs'] if n_jobs is None else n_jobs, method,
                       clusters_list, runtime_file, refit=refit, gpu_id=gpu_id)
            print(f'Training of {method} ends successfully')
