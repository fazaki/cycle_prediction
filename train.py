import os
from cycle_prediction.t2e import t2e
from cycle_prediction.weibull_utils import check_dir
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

path = 'data/'
check_dir(path)

a = os.path.join(path, 'helpdesk.csv')
b = os.path.join(path, 'bpi_12_w.csv')
c = os.path.join(path, 'bpi_12_w_no_repeat.csv')
d = os.path.join(path, 'env_permit.csv')

a = pd.read_csv(a)
b = pd.read_csv(b)
c = pd.read_csv(c)
d = pd.read_csv(d)

df_dict = {
    'a': a,
    'b': b,
    'c': c,
    'd': d
}
range_dict = {
    'a': range(2, 8, 1),
    'b': range(2, 22, 2),
    'c': range(2, 12, 2),
    'd': range(2, 22, 2)
}


def grid_search(dataset, res, censored, cen_per, fit_type, transform):

    cols = ["prefix", "Layer_Size", "MAE", "unique_pred", "train_size",
            "val_size", "test_size", "Censored %", "fit_time"]
    grid_results = pd.DataFrame(columns=cols)
    for prefix in range_dict[dataset]:
        print("\ndataset:", dataset, "\tprefix:", prefix)
        t2e_obj = t2e(
            df_dict[dataset],
            prefix=prefix,
            resolution=res,
            censored=censored,
            cen_prc=cen_per,
            fit_type=fit_type,
            transform=transform
        )
        t2e_obj.preprocess_dev()
        X_train, X_test, X_val, y_train, y_test, y_val,\
            len_train, len_val, len_test = t2e_obj.smart_split(
                                                            train_prc=0.7,
                                                            val_prc=0.45,
                                                            scaling=True)
        for layer_size in [8]:
            print("Layer size:", layer_size, end=" ..... ")
            t2e_obj.fit(X_train, y_train, X_val, y_val,
                        size=layer_size, vb=False)
            print("Done")
            print("Training accomplished in", t2e_obj.fit_time, "seconds")
            test_result_df, mae = t2e_obj.evaluate(X_test, y_test)
            nunique = test_result_df["T_pred"].nunique()
            cen_percentage = t2e_obj.get_cen_prc()
            grid_results = grid_results.append(
                pd.DataFrame([[prefix, layer_size, mae, nunique, len_train,
                               len_val, len_test, cen_percentage,
                               t2e_obj.fit_time]], columns=cols),
                ignore_index=True)
    return grid_results


def main():
    for dataset in ['a']:  # ['b', 'c', 'd']:

        #   Experiment #1: T2E Un-censored No transformation
        for cen_prc in [0]:

            mae_path = 'output/maes/t2e/'+str(cen_prc) + '/'
            check_dir(mae_path)
            print('### Censored percentage:', cen_prc/100.0)
            grid_results = grid_search(
                                dataset=dataset,
                                res='s',
                                censored=False,
                                cen_per=cen_prc/100.0,
                                fit_type='t2e',
                                transform=False
                        )
            pickle.dump(grid_results, open(mae_path + 't2e_'
                                           + dataset + '_GRU.pkl', 'wb'))

        # Experiment #2: T2E Un-censored transform
        # for cen_prc in [0]:

        #     mae_path = 'output/maes/t2e_transform/'+str(cen_prc) + '/'
        #     check_dir(mae_path)
        #     print('### Censored percentage:', cen_prc/100.0)
        #     grid_results = grid_search(
        #                         dataset=dataset,
        #                         res='s',
        #                         censored=False,
        #                         cen_per=cen_prc/100.0,
        #                         fit_type='t2e',
        #                         transform=True
        #                 )
        #     pickle.dump(grid_results, open(mae_path + 't2e_'
        #                                    + dataset + '_GRU.pkl', 'wb'))

        #    Experiment #3: T2E Censored transform
        # for cen_prc in [50]:
        #     mae_path = 'output/maes/t2e_transform/'+str(cen_prc) + '/'
        #     check_dir(mae_path)

        #     print('### Censored percentage:', cen_prc/100.0)
        #     grid_results = grid_search(
        #                         dataset=dataset,
        #                         res='s',
        #                         censored=True,
        #                         cen_per=cen_prc/100.0,
        #                         fit_type='t2e',
        #                         transform=True
        #                 )
        #     pickle.dump(grid_results, open(mae_path + 't2e_'
        #                                    + dataset + '_GRU.pkl', 'wb'))


if __name__ == "__main__":
    main()
