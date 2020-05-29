from t2e_utils import *
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
import os


path = '../Tax_paper/data/'
b = os.path.join(path,'bpi_12_w.csv')
b = pd.read_csv(b)

df_dict = {
    'a': None,
    'b': b,
    'c': None,
    'd':None
}
range_dict = {
    'a': range(2,8,1),
    'b': range(2,22,2),
    'c': range(2,12,2),
    'd': range(2,22,2)
}



def grid_search(dataset, res, censored, cen_per):
    cols = ["suffix", "Layer_Size", "MAE", "unique_pred"]
    grid_results = pd.DataFrame(columns = cols)
    for suffix in range_dict[dataset]:
        print("\ndataset:", dataset, "\tSuffix:", suffix)
        t2e_obj = t2e(df_dict[dataset], 
                      suffix = suffix, 
                      resolution = res, 
                      censored = censored , 
                      cen_prc = cen_per)
        dataset_preprocessed = t2e_obj.preprocess()
        X_train, X_test, X_val, y_train, y_test, y_val = t2e_obj.smart_split(train_prc = 0.7,
                                                                             val_prc = 0.4,
                                                                             scaling=True)
        for l in [4]:
            print("Layer size:",l, end = " ..... ")
            t2e_obj.fit(X_train, y_train, X_val, y_val,size=4, vb = True, seed=0)
            print("Done")
            test_result_df, mae, accuracy = t2e_obj.evaluate(X_test,y_test)            
            nunique = test_result_df["predicted_mode"].nunique()
            grid_results = grid_results.append(pd.DataFrame([[suffix, l,mae,nunique]] , columns = cols), ignore_index=True)
    return grid_results


grid_search(dataset='b',res='s',censored=True,cen_per=0.4)

