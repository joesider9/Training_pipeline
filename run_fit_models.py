#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is responsible for running the model fitting process.
It handles the initialization of the system and manages the training process across different backends.
"""

import sys
import os
import joblib
import importlib
print(os.getcwd())
sys.path.append(os.getcwd())  # Add current working directory to Python path

# Import the initialization module
from eforecast.init.initialize import initializer
from eforecast.clustering.clustering_manager import ClusterOrganizer
# Determine the backend based on the operating system
# Linux systems use command_line backend, while other systems (Windows) use threads
if sys.platform == 'linux' or sys.platform == 'linux2':
    BACKEND = 'command_line'  # Available options: command_line, threads
else:
    BACKEND = 'threads'  # Available options: command_line, threads

# Import the appropriate training manager based on the selected backend
if BACKEND == 'command_line':
    from eforecast.training.command_line.train_manager import fit_clusters
elif BACKEND == 'threads':
    from eforecast.training.threading.train_manager import fit_clusters
else:
    raise ValueError('Unknown backend')


def run_fit_models(project_name_list):
    name = '.'.join(project_name_list)
    module = importlib.import_module(f'{name}.configuration.config')
    config = getattr(module, 'config')
    # Initialize the system with configuration settings
    static_data = initializer(config())
    # cluster_organizer = ClusterOrganizer(static_data, is_online=True, train=False, refit=False)
    # cluster_organizer.update_cluster_folders()
    # Start the model fitting process using the selected backend
    if static_data['transfer_learning']:
        static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
        static_data['static_data_base'] = static_data_base
    fit_clusters(static_data)

if __name__ == '__main__':
    # Import configuration settings
    from case_study.short_term_image.configuration.config import config

    # Initialize the system with configuration settings
    static_data = initializer(config())
    cluster_organizer = ClusterOrganizer(static_data, is_online=True, train=False, refit=False)
    cluster_organizer.update_cluster_folders()
    # Start the model fitting process using the selected backend
    if static_data['transfer_learning']:
        static_data_base = joblib.load(static_data['transfer_learning_from']['configuration'])
        static_data['static_data_base'] = static_data_base
    fit_clusters(static_data)

