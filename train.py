from t2e_utils import *
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
import os

path = '../data/'

a = os.path.join(path,'helpdesk.csv')
b = os.path.join(path,'bpi_12_w.csv')
c = os.path.join(path,'bpi_12_w_no_repeat.csv')
d = os.path.join(path,'env_permit.csv')

a = pd.read_csv(a)
b = pd.read_csv(b)
c = pd.read_csv(d)
d = pd.read_csv(d)

df_dict = {
    'a': a,
    'b': b,
    'c': c,
    'd': d
}
range_dict = {
    'a': range(2,8,1),
    'b': range(2,22,2),
    'c': range(2,12,2),
    'd': range(2,22,2)
}

def grid_search(dataset, res, censored, cen_per):
    
    cols = ["suffix", "Layer_Size", "MAE", "unique_pred", "train_size", "test_size"]
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
        for layer_size in [2,4,8,16,32,64]:
            
            print("Layer size:",layer_size, end = " ..... ")
            t2e_obj.fit(X_train, y_train, X_val, y_val,size=layer_size, vb = False)
            print("Done")
            test_result_df, mae, accuracy = t2e_obj.evaluate(X_test,y_test)            
            nunique = test_result_df["predicted_mode"].nunique()
            grid_results = grid_results.append(pd.DataFrame([[suffix,layer_size,mae,nunique, len(y_train), len(y_test)]] , columns = cols), ignore_index=True)
    return grid_results

mae_path = 'output_files/maes/double_layer/'

grid_results_a_censored = grid_search(dataset='a',res='s',censored=True,cen_per=0.4)
pickle.dump(grid_results_a_censored, open(mae_path + 'grid_results_a_censored.pkl', 'wb'))
grid_results_a_observed = grid_search(dataset='a',res='s',censored=False,cen_per=0.4)
pickle.dump(grid_results_a_observed, open(mae_path + 'grid_results_a_observed.pkl', 'wb'))

# grid_results_b_censored = grid_search(dataset='b',res='s',censored=True,cen_per=0.4)
# pickle.dump(grid_results_b_censored, open(mae_path + 'grid_results_b_censored.pkl', 'wb'))

# grid_results_b_observed = grid_search(dataset='b',res='s',censored=False,cen_per=0.4)
# pickle.dump(grid_results_b_observed, open(mae_path + 'grid_results_b_observed.pkl', 'wb'))

# grid_results_c_censored = grid_search(dataset='c',res='s',censored=True,cen_per=0.4)
# pickle.dump(grid_results_c_censored, open(mae_path + 'grid_results_c_censored.pkl', 'wb'))

# grid_results_c_observed = grid_search(dataset='c',res='s',censored=False,cen_per=0.4)
# pickle.dump(grid_results_c_observed, open(mae_path + 'grid_results_c_observed.pkl', 'wb'))

grid_results_d_censored = grid_search(dataset='d',res='s',censored=True,cen_per=0.4)
pickle.dump(grid_results_d_censored, open(mae_path + 'grid_results_d_censored.pkl', 'wb'))
grid_results_d_observed = grid_search(dataset='d',res='s',censored=False,cen_per=0.4)
pickle.dump(grid_results_d_observed, open(mae_path + 'grid_results_d_observed.pkl', 'wb'))