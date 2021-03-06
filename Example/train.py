import sys
sys.path.insert(0, '../')
import os
from cycle_prediction.t2e import t2e
from cycle_prediction.weibull_utils import check_dir
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


path = '../data/'
check_dir(path)

a = os.path.join(path, 'helpdesk.csv')
b = os.path.join(path, 'bpi_12_w.csv')
c = os.path.join(path, 'bpi_12_w_no_repeat.csv')
d = os.path.join(path, 'env_permit.csv')
e = os.path.join(path, 'bpic13_all.csv')
f = os.path.join(path, 'Sepsis_Cases_processed.csv')
g = os.path.join(path, 'road_traffic.csv')
h = os.path.join(path, 'bpic15_2.csv')

a = pd.read_csv(a)
b = pd.read_csv(b)
c = pd.read_csv(c)
d = pd.read_csv(d)
e = pd.read_csv(e)
f = pd.read_csv(f)
g = pd.read_csv(g)
h = pd.read_csv(h)


df_name = {
    'a': 'helpdesk',
    'b': 'bpi12',
    'c': 'bpi12_norep',
    'd': 'env',
    'e': 'bpic13_all',
    'f': 'Sepsis',
    'g': 'Road_traffic',
    'h': 'bpic15'
}

df_dict = {
    'a': a,
    'b': b,
    'c': c,
    'd': d,
    'e': e,
    'f': f,
    'g': g,
    'h': h,
}

range_dict = {
    'a': range(2, 8, 1),
    'b': range(2, 22, 2),
    'c': range(2, 12, 2),
    'd': range(2, 22, 2),
    'e': range(2, 22, 2),
    'f': range(3, 11, 1),
    'g': range(1, 6, 1),
    'h': range(5, 30, 5)
}

end_event_dict = {
    'a': df_dict['a'].drop_duplicates(subset=['CaseID','ActivityID'], keep='last')['ActivityID'].unique().tolist(),
    'b': df_dict['b'].drop_duplicates(subset=['CaseID','ActivityID'], keep='last')['ActivityID'].unique().tolist(),
    'c': df_dict['c'].drop_duplicates(subset=['CaseID','ActivityID'], keep='last')['ActivityID'].unique().tolist(),
    'd': df_dict['d'].drop_duplicates(subset=['CaseID','ActivityID'], keep='last')['ActivityID'].unique().tolist(),
    'e': ['Completed+Closed', 'Completed+In Call', 'Completed-Closed',
          'Completed+Resolved', 'Completed+Cancelled', 'Completed-Cancelled'],
    'f': ['Release A', 'Release B', 'Release C', 'Release D', 'Release E',
          'Return ER'],
    'g': ['Payment', 'Send for Credit Collection'],
    'h': ['01_HOOFD', '01_BB', '08_AWB45', '05_EIND', '13_CRD', '14_VRIJ',
          '10_UOV', '99_NOCODE', '04_BPT', '06_VD', '12_AP', '01_OLO',
          '02_DRZ', '16_LGSD', '09_AH', '16_LGSV', '03_VD', '03_GBH',
          '07_OPS', '10_OLO', '11_AH']
}

cols = ["prefix", "Layer_Size", "MAE", "unique_pred", "train_size",
        "val_size", "test_size", "Censored %", "fit_time"]

exp_dict = {
    1: {
      'mae_path': '../output/maes/prc/',
      'dynamic_features': ['ActivityID'],
      'static_features': [],
      'transform': 'log'
    },
#     2: {
#       'mae_path': '../output/maes/low_censored_dyn/',
#       'dynamic_features': ['ActivityID', 'impact', 'type'],
#       # 'dynamic_features': ['ActivityID','case_length_cat'],
#       'static_features': [],
#       'transform': None
#     },
#     3: {
#       'mae_path': '../output/maes/low_censored_sta/',
#       'dynamic_features': ['ActivityID'],
#       'static_features': ['impact', 'type'],
#       # 'static_features': ['case_length_cat'],
#       'transform': None
#     },
#     4: {
#       'mae_path': '../output/maes/low_censored_trans/',
#       'dynamic_features': ['ActivityID'],
#       'static_features': [],
#       'transform': 'log'
#     },
#     5: {
#       'mae_path': '../output/maes/low_censored_dyn_trans/',
#       'dynamic_features': ['ActivityID', 'impact', 'type'],
#       # 'dynamic_features': ['ActivityID','case_length_cat'],
#       'static_features': [],
#       'transform': 'log'
#     },
#     6: {
#       'mae_path': '../output/maes/low_censored_sta_trans/',
#       'dynamic_features': ['ActivityID'],
#       'static_features': ['impact', 'type'],
#       # 'static_features': ['case_length_cat'],
#       'transform': 'log'
#     },
}

def grid_search(dataset, exp, extra_censored):

    grid_results = pd.DataFrame(columns=cols)
    for prefix in range_dict[dataset]:
        t2e_obj = t2e(
            dataset=df_dict[dataset],
            prefix=prefix,
            resolution=res,
            dynamic_features=exp_dict[exp]['dynamic_features'],
            static_features=exp_dict[exp]['static_features'],
            fit_type='t2e',
            transform=exp_dict[exp]['transform'],
            end_event_list=end_event_dict[dataset],
            censored=censored
        )
        t2e_obj.train_val_test_split()
        t2e_obj.preprocess(extra_censored=extra_censored)
        X_train, X_val, X_test, y_train, y_val, y_test = t2e_obj.xy_split()
        try:
            t2e_obj.build_model(X_train, y_train,
                                size_dyn=8, size_sta=4)
            t2e_obj.fit(X_train, y_train, X_val, y_val,
                        bs=128, exp_dir=dataset+'_'+str(exp)+'_'+str(prefix),
                        vb=False)
        except Exception:
            try:
                t2e_obj.build_model(X_train, y_train,
                                    size_dyn=8, size_sta=4)
                t2e_obj.fit(X_train, y_train, X_val, y_val,
                            bs=64,
                            exp_dir=dataset+'_'+str(exp)+'_'+str(prefix),
                            vb=False)
            except Exception:
                t2e_obj.build_model(X_train, y_train,
                                    size_dyn=8, size_sta=4)
                t2e_obj.fit(X_train, y_train, X_val, y_val,
                            bs=32,
                            exp_dir=dataset+'_'+str(exp)+'_'+str(prefix),
                            vb=False)

        test_result_df, mae = t2e_obj.evaluate(X_test, y_test)
        nunique = test_result_df["T_pred"].nunique()
        cen_percentage = t2e_obj.get_cen_prc()
        grid_results = grid_results.append(
            pd.DataFrame([[prefix, size_dyn, mae, nunique, len(X_train[0]),
                           len(X_val[0]), len(X_test[0]), cen_percentage,
                           t2e_obj.fit_time]], columns=cols),
            ignore_index=True)
    return grid_results




dataset = 'd'
res = 'd'
fit_type = 't2e'
censored = True
size_dyn = 4


def main():
    
    for extra_censored in np.arange (1.0, 1.1, 0.1):
        print(extra_censored)
        for exp, v in exp_dict.items():
            mae_path = v['mae_path']
            check_dir(mae_path + dataset + '/')
            # print('saving  ==> ', mae_path)
            grid_results = grid_search(dataset, exp, extra_censored)
            pickle.dump(grid_results, open(mae_path + 
                        dataset + '/' + str(extra_censored)+
                        '_GRU.pkl', 'wb'))


if __name__ == "__main__":
    main()
