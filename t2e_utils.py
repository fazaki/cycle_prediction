import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import seaborn as sns
import math
from math import ceil
import tensorflow as tf
import wtte.weibull as weibull
import wtte.wtte as wtte
from wtte.wtte import WeightWatcher
from sklearn.preprocessing import normalize
from six.moves import xrange
from keras import backend as k
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential, load_model,Model
from keras.layers import Dense,LSTM,GRU,Activation,Masking,BatchNormalization,Lambda,Input
from keras import callbacks
from keras.optimizers import RMSprop,adam
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
##########################################################################################################################
##########################################################################################################################
def time_features(case):
    endtime = case["CompleteTimestamp"].reset_index(drop=True)[len(case)-1]
    starttime = case["CompleteTimestamp"].reset_index(drop=True)[0]
    case["fvt1"] = case["CompleteTimestamp"].diff(periods=1)
    case["fvt2"] = case["CompleteTimestamp"].dt.hour
    case["fvt3"] = case["CompleteTimestamp"].dt.weekday*24 + case["CompleteTimestamp"].dt.hour
    case["T2E"] =  endtime - case["CompleteTimestamp"]
    return case

def xy_split(processed_dataset, n_steps, resolution = 'hourly'):
    X = processed_dataset.groupby(["CaseID"]).apply(lambda df:extract_X(df, n_steps))
    X = np.array(tf.convert_to_tensor(X))
    y = processed_dataset.groupby(["CaseID"]).apply(lambda df:extract_y(df, n_steps, resolution = resolution))
    y = np.concatenate(y.values)
    y = y.reshape(y.shape[0],1)
    y = np.append(y,np.ones_like(y),axis=1)
    return X,y

def extract_X(df,n_steps):
    feature_idx = np.concatenate(\
               np.where(df.columns.str.contains('ActivityID_')) + \
#                np.where(df_helpdesk_preprocessed.columns.str.contains('weekday_')) + \
               np.where(df.columns.str.contains('fvt'))
              )
    x = []
    if len(df) < n_steps:
        x.append(
            np.concatenate(
                (np.full((n_steps - df.shape[0] , len(feature_idx)),fill_value=-100), 
                 df.values[0:n_steps,feature_idx]), 
                 axis=0))
    else:
        x.append(df.values[0:n_steps,feature_idx])

    x = np.hstack(np.array(x)).flatten()
    x = x.reshape((n_steps, len(feature_idx)))

    return(x)

def extract_y(df,n_steps,resolution='hourly'):
    y = []
    if resolution == 'hourly':
        time_idx = np.where(df.columns.str.contains("H2E"))[0]
    else:
        time_idx = np.where(df.columns.str.contains("D2E"))[0]
    if len(df) <= n_steps: 
        y.append(df.values[-1,time_idx])
    else:
        y.append(df.values[n_steps-1,time_idx])
    return np.array(y)

def preprocess(dataset,min_length = 3):
    tmp = dataset.groupby(["CaseID"]).count()
    to_drop = list(tmp.loc[tmp["ActivityID"] < min_length].index)
    dataset.CompleteTimestamp = pd.to_datetime(dataset.CompleteTimestamp)
    dataset.sort_values(["CaseID", "CompleteTimestamp"],ascending=True)
#     dataset = dataset.loc[~dataset["CaseID"].isin(to_drop)].reset_index(drop=True)
    dataset = dataset.groupby("CaseID").apply(lambda case:time_features(case))
    dataset["D2E"] = dataset["T2E"].apply(lambda x:x.days)
    dataset["H2E"] = dataset["T2E"].apply(lambda x:x.total_seconds()/3600)
    dataset["fvt1"] = dataset["fvt1"].apply(lambda x:x.total_seconds()/3600)
    dataset.fvt1.fillna(0,inplace=True)
#     dataset["weekday"] = dataset["CompleteTimestamp"].dt.weekday
#     dummy1 = pd.get_dummies(dataset["weekday"],prefix="weekday",drop_first=True)
    dummy2 = pd.get_dummies(dataset["ActivityID"],prefix="ActivityID",drop_first=True)
#     dataset = pd.concat([dataset,dummy1,dummy2],axis=1)
    dataset = pd.concat([dataset,dummy2],axis=1)
    last_step = dataset.drop_duplicates(subset=["CaseID"],keep='last')["ActivityID"].index
    dataset = dataset.drop(last_step,axis=0).reset_index(drop=True)
    return dataset

def smart_split(df, train_perc,val_perc,  suffix, scaling = True):
    min_case_length = suffix+1
    all_suffixes = set(df.CaseID.unique())
    tmp = df.groupby(["CaseID"]).count()
    below_suffix = set(tmp.loc[tmp["ActivityID"] < min_case_length].index)
    above_suffix = list(all_suffixes.difference(below_suffix))
    
    len_train = int(train_perc * len(above_suffix))
    cases_train = above_suffix[0:len_train] + list(below_suffix)
    
    len_val = int(len(cases_train)*val_perc)
    np.random.seed(10)
    np.random.shuffle(cases_train)
    cases_val   = [cases_train.pop() for i in range(len_val)]
    
    cases_test  = above_suffix[len_train:]
    
    df_train = df.loc[df['CaseID'].isin(cases_train)].reset_index(drop=True)
    df_val = df.loc[df['CaseID'].isin(cases_val)].reset_index(drop=True)
    df_test  = df.loc[df['CaseID'].isin(cases_test)].reset_index(drop=True)
    if scaling:
        sc = StandardScaler()
        df_train.loc[:,["fvt1", "fvt2", "fvt3"]] = sc.fit_transform(df_train[["fvt1", "fvt2", "fvt3"]])
        df_val.loc[:,["fvt1", "fvt2", "fvt3"]] = sc.transform(df_val[["fvt1", "fvt2", "fvt3"]])
        df_test.loc[:,["fvt1", "fvt2", "fvt3"]] = sc.transform(df_test[["fvt1", "fvt2", "fvt3"]])
   
    X_train,y_train = xy_split(df_train,resolution='daily',n_steps=suffix)
    X_val,y_val = xy_split(df_val,resolution='daily',n_steps=suffix)
    X_test,y_test = xy_split(df_test,resolution='daily',n_steps=suffix)

    return X_train, X_test, X_val, y_train, y_test, y_val

def balance_labels(X,y):
    unique = np.unique(y[:,0])
    count_all = len(unique)
    for i in unique:
        count_i = np.squeeze(np.argwhere(y[:,0] == i)).size
        if count_i > (len(y) // count_all) +1:
            num_delete = count_i - (len(y) // count_all)
            idx_delete = np.random.choice(np.squeeze(np.argwhere(y[:,0]==i)), num_delete, replace=False)
            X = np.delete(X, idx_delete , 0)
            y = np.delete(y, idx_delete , 0)
    return X, y

batch_size = 128
def batch_gen(X, y):
    n_batches = math.ceil(len(X) / batch_size)
    while True: 
        for i in range(n_batches):
            X_batch = X[i*batch_size:(i+1)*batch_size, :,:]
            y_batch = y[i*batch_size:(i+1)*batch_size, :]
            yield X_batch, y_batch
            
def batch_gen_test(X):
    n_batches = math.ceil(len(X) / batch_size)
    while True: 
        for i in range(n_batches):
            X_batch = X[i*batch_size:(i+1)*batch_size, :,:]
            yield X_batch
            
def evaluating(X,y,model):    
    # Make some predictions and put them alongside the real TTE and event indicator values
    mg_test = batch_gen_test(X)
    nb_samples = len(X)
    test_predict = model.predict_generator(mg_test, steps= ceil(len(X) / batch_size))
    
#     test_predict = model.predict(X)
    test_result = np.concatenate((y, test_predict), axis=1)
    test_results_df = pd.DataFrame(test_result, columns=['T', 'U', 'alpha', 'beta'])
    test_results_df['predicted_mode'] =   test_results_df[['alpha', 'beta']].apply(lambda row: weibull_mode(row[0], row[1]), axis=1)
    test_results_df['error'] = test_results_df['T'] - test_results_df['predicted_mode']
    test_results_df["abs_error"] = np.absolute(test_results_df["error"])
    test_results_df["Accurate"] = ((test_results_df["U"] == 1) & (test_results_df["abs_error"] <= 1)) | \
                                  ((test_results_df["U"] == 0) & (test_results_df["predicted_mode"] >= test_results_df["T"]-1)) 
#     print("Accuray =", round(test_results_df["Accurate"].mean()*100,3), "%")
    mae = mean_absolute_error(test_results_df['T'], test_results_df['predicted_mode'])
    return test_results_df, mae

def train(X_train, y_train, X_val, y_val):
    tte_mean_train = np.nanmean(y_train[:,0].astype('float'))
    mean_u = np.nanmean(y_train[:,1].astype('float'))
    init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
    init_alpha = init_alpha/mean_u
    history = History()
    n_features = X_train.shape[-1]
    main_input = Input(shape=(None, n_features), name='main_input')
    l1 = GRU(10, activation='tanh', recurrent_dropout=0.25)(main_input)
    b1 = BatchNormalization()(l1)
    l2 = Dense(2, name='Dense')(b1)
    b2 = BatchNormalization()(l2)
    output = Lambda(wtte.output_lambda, arguments={"init_alpha":init_alpha,"max_beta_value":100, "scalefactor":0.5})(b2)
    # Use the discrete log-likelihood for Weibull survival data as our loss function
    loss = wtte.loss(kind='continuous',reduce_loss=False).loss_function

    model = Model(inputs=[main_input], outputs=[output])
    model.compile(loss=loss, optimizer=adam(lr=0.005))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=200, restore_best_weights=True)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=False, save_best_only=True, save_weights_only=True)

    mg_train = batch_gen(X_train, y_train)
    mg_val = batch_gen(X_val, y_val)
    model.fit_generator(mg_train, 
                    epochs=500,
                    steps_per_epoch = ceil(len(X_train) / batch_size),
                    validation_data=(mg_val),
                    validation_steps= ceil(len(X_val) / batch_size),
                    verbose=False,
                    callbacks=[history,es,mc]
                   )
    return model

"""
Discrete log-likelihood for Weibull hazard function on censored survival data
    y_true is a (samples, 2) tensor containing time-to-event (y), and an event indicator (u)
    ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""
def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = k.pow((y_ + 1e-35) / a_, b_)
    hazard1 = k.pow((y_ + 1) / a_, b_)

    return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)

"""
    Not used for this model, but included in case somebody needs it
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""
def weibull_loglik_continuous(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    ya = (y_ + 1e-35) / a_
    return -1 * k.mean(u_ * (k.log(b_) + b_ * k.log(ya)) - k.pow(ya, b_))

"""
    Custom Keras activation function, outputs alpha neuron using exponentiation and beta using softplus
"""
def activate(ab):
    a = k.exp(ab[:, 0])
    b = k.softplus(ab[:, 1])

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)
##########################################################################################################################
##########################################################################################################################

def validate_df(df):
    df_len = df.shape[0]
    summary = []
    for i in range(len(df.columns)):
        summary_1 = [0]*5
        summary_1[0] = df.columns[i]
        summary_1[1] = sum(df.iloc[:,i]== 0.0)
        summary_1[2] = "{0:.1f}".format(sum(df.iloc[:,i]== 0.0)/df_len*100)
        summary_1[3] = sum(df.iloc[:,i].isnull())
        summary_1[4] = "{0:.1f}".format(sum(df.iloc[:,i].isnull())/df_len*100)
        summary.append(summary_1)
    
    report = pd.DataFrame(summary, columns=["Feature_Name","Zero_Count","Zero_Percentage", "Null_Count", "Null_Percentage"])
    return report
##########################################################################################################################
##########################################################################################################################
def interpolate_df(x,lim = 2):
    x.iloc[0]  = x.fillna(method='bfill',axis=0,limit=1).iloc[0]
    x.iloc[-1] = x.fillna(method='ffill',axis=0,limit=1).iloc[-1]
    try:
        x.interpolate(method='slinear',limit =lim,axis=0,inplace =True)
    except:
        pass
    return(x)
##########################################################################################################################
##########################################################################################################################
def weibull_pdf(alpha, beta, t):
    return (beta/alpha) * ((t+1e-35)/alpha)**(beta-1)*np.exp(- ((t+1e-35)/alpha)**beta)

def weibull_median(alpha, beta):
    return alpha*(-np.log(.5))**(1/beta)

def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)

def weibull_mode(alpha, beta):
    if beta < 1:
        return 0
    return alpha * ((beta-1)/beta)**(1/beta)
##########################################################################################################################
##########################################################################################################################
def plot_predictions_insights(results_df):

    plt.figure(figsize=(12,12))
    t=np.arange(0,300)
    
    plt.subplot(3,2,1)
    observed = results_df.loc[results_df['U'] == 1]
    sns.scatterplot(observed['T'], observed["predicted_mode"],hue=observed["Accurate"])
    plt.xlabel("Actual failure time",fontsize=12)
    plt.ylabel("Predicted failure time",fontsize=12)
    plt.title('Actual Vs. Predicted (Observed)',fontsize=18)
    plt.legend(loc = 'upper left')
    
    plt.subplot(3,2,2)
    censored = results_df.loc[results_df['U'] == 0]
    sns.scatterplot(censored['T'], censored["predicted_mode"],hue=censored["Accurate"])
    plt.xlabel("Time of observation",fontsize=12)
    plt.ylabel("Predicted failure time",fontsize=12)
    plt.title('Actual Vs. Predicted (Censored)',fontsize=18)
    plt.legend(loc = 'upper left')
        
    plt.subplot(3,2,(3,4))
    sns.distplot(results_df['error'], bins=100, kde=True,hist=True, norm_hist=False, label="Error Rate")
    plt.title('Error distribution',fontsize=18)
    plt.legend(loc = 'upper left')
    plt.xlabel("Error Shift",fontsize=12)

    plt.subplot(3,2,(5,6))
    x = results_df.groupby(["T","U"]).agg({"Accurate":"mean"}).reset_index()
    x["Error_Rate"] = 1-x["Accurate"]
    sns.barplot(x= x["T"].astype("int"), y=x["Error_Rate"], hue=x["U"])
    plt.title('Error rate per time step')
    
    plt.tight_layout()
    plt.show()
    
    
##########################################################################################################################
##########################################################################################################################

def plot_top_predictions(result_df, lim=10, top_feature = "abs_error", ascending = True, U = 1,accurate = True):
    
    result_df = result_df.loc[(result_df["Accurate"] == accurate) & (result_df["U"] == U)]
    top_accurate = result_df.sort_values(by=top_feature, ascending = ascending).head(lim).index
    result_df.sort_values(by=top_feature, ascending = ascending).head(lim)
    fig, axarr = plt.subplots(len(top_accurate), figsize=(15,len(top_accurate)*3))

    for n,i in enumerate(top_accurate):
        ax = axarr[n]
        T    = result_df.loc[i,'T']
        U    = result_df.loc[i,'U']
        alpha= result_df.loc[i,'alpha']
        beta = result_df.loc[i,'beta']
        mode = result_df.loc[i,'predicted_mode']

        y_max_1 = weibull_pdf(alpha, beta, mode)    
        y_max_2 = weibull_pdf(alpha, beta, T)    

        t=np.arange(0,20)
        ax.plot(t, weibull_pdf(alpha, beta, t), color='w', label="Weibull distribution")
        ax.vlines(mode, ymin=0, ymax=y_max_1, colors='k', linestyles='--', label="Predicted failure time")
        ax.scatter(T, weibull_pdf(alpha,beta, T), color='r', s=100, label="Actual failure time")
        ax.set_facecolor('#A0A0A0')
        ax.xaxis.grid(b=True, which="major", color='k', linestyle='-.', linewidth=0.1)
        ax.set_xticklabels(np.append(0, np.unique(t)))
        ax.set_xlim(left = 0)
        ax.set_ylim(bottom = 0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
        if n == 0:
            ax.legend(frameon=True,fancybox=True,shadow=True,facecolor='lightgray')
            ax.set_title("Time-to-Failure distribution",pad =20, fontsize = 20)
    #plt.tight_layout()
    plt.show()