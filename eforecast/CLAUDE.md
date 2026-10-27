# CLAUDE.md — eforecast Pipeline

## Project Overview

`eforecast` is an end-to-end energy forecasting pipeline for **wind power**, **solar power**, and **load** forecasting. It supports:
- Multiple data modalities: time series, NWP (Numerical Weather Predictions), satellite images
- Shallow ML models: Random Forest, LightGBM, CatBoost, Lasso, RBF networks
- Deep learning models: LSTM, GRU, Transformers (Informer, Autoformer, FEDformer, iTransformer, etc.), CNNs, KAN
- Fuzzy rule-based systems via RBF clustering
- Ensemble/combination methods: Bayesian Combination (BCP), K-Means routing, shallow classifiers
- Hyperparameter optimization via hyperopt (TPE) and TuRBO

---

## Directory Structure

```
eforecast/
├── common_utils/       # Shared utilities (eval, dataset ops, training helpers, logging)
├── clustering/         # RBF-OLS fuzzy clustering for rule-based models
├── combine_predictions/# Ensemble algorithms (BCP, KMeans, shallow classifier)
├── datasets/           # Data I/O, preprocessing, feature engineering
│   ├── data_preprocessing/  # Scaling, imputation, pipeline
│   ├── data_resampling/     # Sampling strategies
│   ├── image_data/          # Satellite image dataset creation and loading
│   └── nwp_data/            # NWP dataset creation, compression, organisation
├── deep_models/        # PyTorch deep learning architectures
│   └── pytorch_2x/
│       ├── nixlta_transformers/  # LSTM, GRU, Informer, Autoformer, etc.
│       ├── image/               # CNN, CrossViT, Attention-UNet
│       └── conv_att/            # CBAM, GeoDANet attention modules
├── feature_selection/  # ARFS, MRMR, Lasso, importance-based selection
├── init/               # Project initialisation and directory setup
├── nwp_extraction/     # Downloading/organising ECMWF, GFS, Skiron NWP data
├── prediction/         # Inference engine (Predictor class)
├── runtime_templates/  # Scripts for online prediction and batch training
├── shallow_models/     # RF, LightGBM, CatBoost, Lasso, RBF (ShallowModel class)
├── training/           # Training orchestration and hyperparameter optimisation
│   ├── command_line/   # Objective functions, train manager
│   └── optimizers/     # Hyperopt wrapper, TuRBO
└── trial_scripts/      # Experimental / one-off scripts
```

---

## Key Classes and Entry Points

| Class / Function | File | Purpose |
|---|---|---|
| `DatasetCreator` | `datasets/dataset_creator.py` | Creates all dataset types from raw CSV |
| `FilesManager` | `datasets/files_manager.py` | Caches/loads datasets as pickle/CSV |
| `DataFeeder` | `datasets/data_feeder.py` | Loads cached data and feeds it to models |
| `DataTransformer` | `datasets/data_transformations.py` | Scaling, imputation pipeline |
| `ShallowModel` | `shallow_models/shallow_model.py` | Unified shallow model wrapper |
| `RBFols` | `shallow_models/rbf_ols_network.py` | Custom RBF-OLS network |
| `ClusterOrganizer` | `clustering/clustering_manager.py` | Fits and applies fuzzy clustering |
| `Predictor` | `prediction/predict.py` | Inference engine (all modalities) |
| `initializer` | `init/initialize.py` | Sets up project directories and static_data |
| `fit_clusters` | `training/command_line/train_manager.py` | Main training entry point |
| `compute_metrics` | `common_utils/eval_utils.py` | MAE, RMSE, SSE computation |
| `create_centroids` | `common_utils/train_utils.py` | RBF centroid initialisation |

---

## Configuration: `static_data` Dict

The pipeline is driven by a `static_data` dict (typically from a YAML/Python config and then processed by `initializer()`). Key fields:

```python
static_data = {
    # Identity
    'project_name': str,
    'type': 'wind' | 'solar' | 'load' | 'FA',
    'horizon_type': 'single-step' | 'multi-output' | 'intra-ahead' | 'day-ahead',
    'country': str,                    # For SP index and last-year lags
    'coord': (lat, lon),               # Project location
    'rated': float | None,             # Capacity for normalisation

    # Data paths
    'filename': str,                   # Path to main CSV file
    'path_data': str,                  # Where processed datasets are cached
    'sys_folder': str,                 # Root folder

    # Time series resolution
    'ts_resolution': 'h' | '15min' | 'D',

    # Variables config — list of dicts, one per feature
    'variables': [{'name': str, 'type': 'timeseries'|'nwp'|'calendar'|'image',
                   'source': str, 'lags': list, ...}],
    'target_variable': {'name': str, 'source': str, 'lags': list, 'columns': list},

    # NWP
    'NWP': list | None,                # NWP model configs (ecmwf, gfs, skiron)
    'nwp_data_merge': 'all'|'area'|'variable',
    'compress_data': 'dense'|'full'|'minimal'|'semi_full'|'load',

    # Images
    'use_image': str | None,
    'sat_image_type': str,

    # Scaling
    'scale_target_method': 'standard'|'minmax'|'robust',
    'scale_row_method': str | None,
    'scale_nwp_method': str | None,
    'scale_img_method': str | None,

    # Fuzzy / Clustering
    'is_Fuzzy': bool,
    'clustering': {'method': 'RBF', 'thres_act': float, 'n_rules': int, ...},

    # Model selection
    'cluster_methods': dict,           # Deep models per cluster
    'methods_cpu': dict,               # Shallow models (cpu)
    'global_methods': dict,

    # Combining
    'combining': {'method': 'kmeans'|'bcp'|'shallow_classifier', 'methods': list},

    # Training
    'n_jobs': int,
    'use_polars': bool,
    'global_past_lags': int | list,
    'global_future_lags': int | list,
}
```

---

## Data Flow

```
CSV Input
   ↓ DatasetCreator.load_data()     — fix_timeseries_dates(), RowDataTransformer
   ↓ create_[nwp|image|lstm|row]_dataset()  — cached as pickle files
   ↓ DataTransformer.fit_pipe()     — fit scalers/imputers, save
   ↓ DataFeeder.feed_inputs()       — load cached, transform, feature-select
   ↓ FeatureSelector.transform()    — filter relevant features
   ↓ FilesManager.split()           — train / val / test via cv_mask
   ↓ ShallowModel.fit() / DeepModel training loop
   ↓ ClusterOrganizer (if is_Fuzzy) — per-rule models
   ↓ Predictor / combine_predictions — ensemble
   ↓ compute_metrics()              — evaluate final predictions
```

---

## Metrics

All errors are **normalised by `rated`** (installed capacity). Metrics returned by `compute_metrics()`:

| Metric | Formula |
|---|---|
| `mae` | `100 × mean(|pred - y| / rated)` % |
| `rms` | `100 × sqrt(mean((|pred - y| / rated)²))` % |
| `sse` | `sum((pred - y)²)` (unnormalised) |
| `mse` | `sse / N` |

For `multi-output`, per-horizon MAE is returned with an `average` column.

---

## Development Notes

### Running Training
Use `runtime_templates/run_model_script.py` as the main batch training script.
Use `runtime_templates/run_model_online.py` for online/real-time inference.

### GPU Management
`GpuQueue` (`common_utils/train_utils.py`) is a multiprocessing context-manager that assigns one GPU per process from a pool. Set `N_GPUS` to the number of available devices.

### Adding a New Shallow Model
1. Add a branch in `ShallowModel.__init__` and `ShallowModel.reset` for the new method name.
2. Implement `fit` and `predict` logic in the `ShallowModel.fit` / `ShallowModel.predict` methods.
3. Register the method name in the `static_data['methods_cpu']` config.

### Adding a New Deep Architecture
1. Add the model class under `deep_models/pytorch_2x/nixlta_transformers/` or `image/`.
2. Register it in the builder (`deep_models/pytorch_2x/builders.py`).
3. Add an experiment tag in `static_data['experiments']`.

### Feature Selection
Feature selectors are stored as pickles at `{cluster_path}/feature_selectors.pickle`.
The `FeatureSelector.transform()` method in `feature_selection/feature_selection_transform.py` applies them at inference time.

### Fuzzy Systems
When `is_Fuzzy=True`, `ClusterOrganizer` fits RBF centroids, computes rule activations, and groups training dates per rule. Each rule gets its own set of models (both shallow and deep). `Predictor` routes test samples to the appropriate rule models at inference.

---

## Known Issues / Improvement Areas

See the code review document for a full list. Critical items:
- `send_predictions()` in `common_utils/train_utils.py` contains **hardcoded email credentials** — move to environment variables or a secrets manager immediately.
- `bare except:` clauses in several places silently swallow errors — replace with specific exception types.
- `ShallowModel.__init__` and `ShallowModel.reset` duplicate ~60 lines — extract to a `_build_model()` helper.
- `linear_output()` has a loop that silently overwrites `x_`, `x_val_`, `x_test_` on each dict key iteration — only the last key's data is used.
- `check_early_stop()` accepts `wait` and `patience` but does not use them — dead parameters.
- `pad_images()` has a typo: `panding` instead of `padding`.
- RF `max_size` param in `ShallowModel.fit` is not a valid sklearn RF parameter.

---

## Dependencies (key)

- **PyTorch** 2.x (CUDA 11.8) — deep models
- **scikit-learn** — shallow models, preprocessing
- **LightGBM**, **CatBoost** — gradient boosting
- **hyperopt** — Bayesian HPO
- **pandas**, **polars**, **numpy** — data manipulation
- **pvlib** — solar position calculation
- **astral** — sunrise/sunset for night filtering
- **yagmail** — email notifications (requires credentials)
- **joblib** — model serialisation and parallelism
- **psutil** — CPU load monitoring

Full dependency list: `../requirements.txt` (398 packages).
