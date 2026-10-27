from run_combine_models import run_step1
from run_combine_models import run_step2
from run_combine_models import run_step3


if __name__ == "__main__":
    for park in ['pv_16005', 'pv_24792', 'pv_27533']:
        # run_step1([park, 'day_ahead'], parallel=True)
        # run_step1([park, 'intra_day'], parallel=True)
        run_step2([park, 'day_ahead'], parallel=True)
        run_step2([park, 'intra_day'], parallel=True)
        run_step3([park, 'day_ahead'])
        run_step3([park, 'intra_day'])