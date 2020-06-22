from t2e_utils import *
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
import os

path = 'data/'

a = os.path.join(path,'helpdesk.csv')
b = os.path.join(path,'bpi_12_w.csv')
c = os.path.join(path,'bpi_12_w_no_repeat.csv')
d = os.path.join(path,'env_permit.csv')

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
    'a': range(2,8,1),
    'b': range(2,22,2),
    'c': range(2,12,2),
    'd': range(2,22,2)
}

def grid_search(dataset, res, censored, cen_per):
    
    cols = ["suffix", "Layer_Size", "MAE", "unique_pred", "train_size", "val_size", "test_size","Censored %"]
    grid_results = pd.DataFrame(columns = cols)
    for suffix in range_dict[dataset]:
        print("\ndataset:", dataset, "\tSuffix:", suffix)
        t2e_obj = t2e(df_dict[dataset], 
                      suffix = suffix, 
                      resolution = res, 
                      censored = censored , 
                      cen_prc = cen_per)
        dataset_preprocessed = t2e_obj.preprocess()
        X_train, X_test, X_val, y_train, y_test, y_val, len_train, len_val, len_test = t2e_obj.smart_split(train_prc = 0.7,
                                                                                                             val_prc = 0.45,
                                                                                                             scaling=True)
        for layer_size in [2,4,8,32]:
            
            print("Layer size:",layer_size, end = " ..... ")
            t2e_obj.fit_regression(X_train, y_train, X_val, y_val,size=layer_size, vb = False)
            print("Done")
            test_result_df, mae, accuracy = t2e_obj.evaluate(X_test,y_test)            
            nunique = test_result_df["T_pred"].nunique()
            cen_percentage = t2e_obj.get_cen_prc()
            grid_results = grid_results.append(pd.DataFrame([[suffix,layer_size,mae,nunique, len_train,len_val,len_test, cen_percentage]] , columns = cols), ignore_index=True)
    return grid_results



for dataset in ['a','b','c','d']:
     ## Regression
#     mae_path = 'output_files/maes/regression/00/'
#     grid_results = grid_search(dataset=dataset,res='s',censored=False,cen_per=0.0)
#     pickle.dump(grid_results, open(mae_path + 'regression_'+dataset+'_complete.pkl', 'wb'))
#     mae_path = 'output_files/maes/regression/10/'
#     grid_results = grid_search(dataset=dataset,res='s',censored=False,cen_per=0.4)
#     pickle.dump(grid_results, open(mae_path + 'regression_'+dataset+'_reduced.pkl', 'wb'))
    
#    ## T2E
    for cen_prc in [10,20,30,50,60,70,80,90]:
        mae_path = 'output_files/maes/t2e/'+str(cen_prc) +'/'

#         grid_results = grid_search(dataset=dataset,res='s',censored=False,cen_per=0.0)
#         pickle.dump(grid_results, open(mae_path + 't2e_'+dataset+'_complete.pkl', 'wb'))
        print('### Censored percentage:',cen_prc/100.0)
        grid_results = grid_search(dataset=dataset,res='s',censored=False,cen_per=cen_prc/100.0)
        pickle.dump(grid_results, open(mae_path + 't2e_'+dataset+'_reduced.pkl', 'wb'))

        grid_results = grid_search(dataset=dataset,res='s',censored=True,cen_per=cen_prccen_prc/100.0)
        pickle.dump(grid_results, open(mae_path + 't2e_'+dataset+'_censored.pkl', 'wb'))
