# Training Phase

Python code for preparing data, training models, and combining forecasts for energy forecasting projects. The main package is `eforecast`; `case_study` contains a configured photovoltaic forecasting example that uses weather forecasts and satellite imagery.

## Repository layout

- `eforecast/` — data processing, feature selection, clustering, training, prediction, and evaluation code.
- `case_study` — project, input data, and model settings for the Ptolemaida example.
- `run_*.py` — scripts for dataset preparation, model fitting, forecast combination, and online prediction.
- `requirements.txt` — pinned Python dependencies.

## Setup

Use Python 3.10 or newer from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list is large and includes machine learning packages that may require a compatible GPU or system libraries. `pyproject.toml` currently contains only project metadata, so `requirements.txt` is the dependency source for this repository.

Before running a pipeline, review the example configuration:

- `config_project.py` sets the project identity, forecast horizon, data file, time zones, and weather and satellite sources.
- `config_input_data.py` selects input variables and preprocessing.
- `config_methods.py` selects the training backend and model families.
- `config_utils.py` defines machine specific paths for model output, weather data, and satellite data.

The configured Ptolemaida input file and external weather, satellite, and model directories are **not included** here. Set their paths for your environment and supply the corresponding data before running the scripts. Several `run_*.py` files also contain hard coded project names or dates in their `__main__` blocks; check these before execution.

## Pipeline entry points

| Script | Purpose |
| --- | --- |
| `run_datasets.py` | Extract weather data, create datasets, preprocess, cluster, and select features. Its `project_name_list` must match an importable configuration module. |
| `run_fit_models.py` | Initialize the configured project and fit cluster models. |
| `run_combine_models.py` | Evaluate and combine fitted model predictions. Check its `__main__` block for the selected steps and dates. |
| `run_model_script.py` | Provide the `predict(...)` function for an initialized project. |
| `run_model_online.py` | Run an online forecast workflow for its configured project. |

Run scripts from the repository root so local imports resolve. The typical sequence is to prepare datasets, fit models, and then evaluate or combine predictions. Online prediction requires the fitted model artifacts and current input data.

## How the pieces fit

* **Datasets.** Daily NWP pickles per area, merged into `(samples, lags, variables, lat, lon)`
  tensors; row data (observation lags, calendar, NWP statistics) and LSTM sequences; scalers per
  dataset; a time split into train, validation and test with a fixed seed.
* **Clustering.** A fuzzy RBF network assigns every sample to regimes (rules); each rule gets its
  own models, and a global model sees everything.
* **Deep models.** A declarative experiment maps data tags to branches (vLLMs, Time-distributed convolution blocks shared
  over the NWP lags, LSTM encoder-decoders, transformers over the grid or the sequence), merged by
  token-level cross attention; CLIP learning, one HPO per (cluster, method); the orchestrator packs
  trials of every study on the GPUs by memory, prunes with the study, retries on OOM.
* **Shallow models.** One Optuna study per (cluster, method) on the CPU, validation held out.
* **Ensemble.** The evaluation period is split into a selection slice (choose the best trials,
  fit the combiners per cluster and step) and a holdout slice (the report).  Levels: trials,
  methods per cluster, combiners, cluster averages, global models, models combiner.