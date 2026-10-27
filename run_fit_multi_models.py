from run_fit_models import run_fit_models


if __name__ == "__main__":
    for park in ['pv_16005', 'pv_24792', 'pv_27533']:
        run_fit_models([park, 'day_ahead'])
        run_fit_models([park, 'intra_day'])