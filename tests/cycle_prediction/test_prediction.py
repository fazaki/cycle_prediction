import os
from cycle_prediction.t2e import t2e
# from cyclepred.weibull_utils import check_dir
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

path = 'tests/dataset/'
a = pd.read_csv(os.path.join(path, 'helpdesk_mini.csv'))
dataset = a
df_dict = {
    'a': a
}
range_dict = {
    'a': range(2, 8, 1)
}


def test_method_1():
    # mae_path = 'output/maes/t2e/'
    # check_dir(mae_path)
    # cols = ["prefix", "Layer_Size", "MAE", "unique_pred", "train_size",
    #         "val_size", "test_size", "Censored %", "fit_time"]
    # results = pd.DataFrame(columns=cols)
    t2e_obj = t2e(
        dataset=dataset,
        prefix=2,
        resolution='s',
        censored=False,
        cen_prc=0,
        fit_type='t2e',
        transform=False
        )
    t2e_obj.preprocess_dev()
    assert t2e_obj.dataset.shape == (95, 11)
    assert len(t2e_obj.all_cases) == 36
    assert len(t2e_obj.censored_cases) == 0
    assert t2e_obj.dataset.loc[94, 'fvt3'] == 528120.0


def test_method_2():
    # mae_path = 'output/maes/t2e/'
    # check_dir(mae_path)
    # cols = ["prefix", "Layer_Size", "MAE", "unique_pred", "train_size",
    #         "val_size", "test_size", "Censored %", "fit_time"]
    # results = pd.DataFrame(columns=cols)
    t2e_obj = t2e(
        dataset=dataset,
        prefix=2,
        resolution='s',
        censored=False,
        end_event_list=[6],
        fit_type='t2e',
        transform=False
        )
    t2e_obj.preprocess()
    assert t2e_obj.dataset.shape == (95, 11)
    assert len(t2e_obj.all_cases) == 36
    assert len(t2e_obj.censored_cases) == 2
    assert t2e_obj.dataset.loc[94, 'fvt3'] == 528120.0
