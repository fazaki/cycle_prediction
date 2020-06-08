import numpy as np
np.random.seed(42)
import pandas as pd
# np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import seaborn as sns
import math
from math import ceil
from six.moves import xrange
import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.models import load_model,Model
from tensorflow.keras.initializers import glorot_uniform
# my_init = glorot_uniform(seed =42)

from tensorflow.keras.layers import Dense,LSTM,GRU,Activation,Masking,BatchNormalization,Lambda,Input
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import RMSprop,Adam,Nadam
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

import wtte.weibull as weibull
import wtte.wtte as wtte
from wtte.wtte import WeightWatcher
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

##########################################################################################################################
##########################################################################################################################

class t2e:

    def __init__(self, dataset, suffix, resolution, censored, cen_prc):
        self.dataset = dataset
        self.suffix = suffix
        self.censored = censored
        self.cen_prc = cen_prc
        self.censored_cases = []
        self.batch_size = 128
        self.resolution = resolution
        self.model = None
        
    def preprocess(self):
        
        # Retrieve all sequences with their records count
        self.dataset.sort_values(["CaseID", "CompleteTimestamp"],ascending=True)
        case_counts = self.dataset.groupby(["CaseID"]).count()
        
        # Only keep sequences that has at least suffix + 1 records
        to_drop = list(case_counts.loc[case_counts["ActivityID"] <= self.suffix].index)
        self.dataset = self.dataset.loc[~self.dataset["CaseID"].isin(to_drop)].reset_index(drop=True)

        ## Generate censored data:
        to_censor_from = list(self.dataset.groupby(["CaseID"]).count().loc[case_counts["ActivityID"] > self.suffix+1].index)
        np.random.seed(42)
        self.censored_cases = np.random.choice(to_censor_from, int(len(to_censor_from)*self.cen_prc), replace=False)
        print("first 10 censored cases",self.censored_cases[0:10])
        
        self.dataset["U"] = ~self.dataset['CaseID'].isin(self.censored_cases)*1
        self.dataset.CompleteTimestamp = pd.to_datetime(self.dataset.CompleteTimestamp)
        self.dataset = self.dataset.groupby("CaseID").apply(lambda case:self.__time_features(case))
        self.dataset.fvt1.fillna(0,inplace=True)
    #     dataset["weekday"] = dataset["CompleteTimestamp"].dt.weekday
    #     dummy1 = pd.get_dummies(dataset["weekday"],prefix="weekday",drop_first=True)
        dummy2 = pd.get_dummies(self.dataset["ActivityID"],prefix="ActivityID",drop_first=True)
    #     dataset = pd.concat([dataset,dummy1,dummy2],axis=1)
        self.dataset = pd.concat([self.dataset,dummy2],axis=1)
        last_step = self.dataset.drop_duplicates(subset=["CaseID"],keep='last')["ActivityID"].index
        self.dataset = self.dataset.drop(last_step,axis=0).reset_index(drop=True)
        return self.dataset

    def smart_split(self, train_prc, val_prc, scaling):
        all_cases = set(self.dataset.CaseID.unique())
        censored  = set(self.censored_cases)
        observed  = list(all_cases.difference(censored))

        len_train = int(train_prc * len(observed))
        len_val = int(len_train*val_prc)
        cases_train = observed[0:len_train]
        cases_val   = [cases_train.pop() for i in range(len_val)]
        cases_test  = observed[len_train:]

        print("\tTotal Observed:", len(observed))
        print("\tTraining data Observed:", len(cases_train))
        print("\tTraining data Censored:", len(censored))
        if self.censored:
            cases_train = cases_train + list(censored)
        print("\tTraining data to use:", len(cases_train))

        print("\tValidation data:", len(cases_val ))
        print("\tTesting data   :", len(cases_test))
        df_train = self.dataset.loc[self.dataset['CaseID'].isin(cases_train)].reset_index(drop=True)
        df_val   = self.dataset.loc[self.dataset['CaseID'].isin(cases_val)  ].reset_index(drop=True)
        df_test  = self.dataset.loc[self.dataset['CaseID'].isin(cases_test) ].reset_index(drop=True)
        if scaling:
            sc = StandardScaler()
            df_train.loc[:,["fvt1", "fvt2", "fvt3"]] = sc.fit_transform(df_train[["fvt1", "fvt2", "fvt3"]])
            df_val.loc[:,["fvt1", "fvt2", "fvt3"]] = sc.transform(df_val[["fvt1", "fvt2", "fvt3"]])
            df_test.loc[:,["fvt1", "fvt2", "fvt3"]] = sc.transform(df_test[["fvt1", "fvt2", "fvt3"]])

        X_train,y_train = self.__xy_split(df_train)
        X_val,y_val     = self.__xy_split(df_val)
        X_test,y_test   = self.__xy_split(df_test)

        return X_train, X_test, X_val, y_train, y_test, y_val, len(cases_train), len(cases_val), len(cases_test)


    def fit(self, X_train, y_train, X_val, y_val,size, vb = True):
        
        test_out_path = "testing_output/"
        vb = vb
        if vb:
            print("\n")
        tte_mean_train = np.nanmean(y_train[:,0].astype('float'))
        mean_u = np.nanmean(y_train[:,1].astype('float'))
        init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
        init_alpha = init_alpha/mean_u
        history = History()
        cri = 'val_loss'
        csv_logger = CSVLogger(test_out_path + 'training.log', separator=',', append=False)
        es = EarlyStopping(monitor=cri, mode='min', verbose=vb, patience=50, restore_best_weights=False)
        mc = ModelCheckpoint(test_out_path + 'best_model.h5', monitor=cri, mode='min', verbose=vb, save_best_only=True, save_weights_only=True)
        lr_reducer = ReduceLROnPlateau(monitor=cri, factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        n_features = X_train.shape[-1]

        np.random.seed(42)
        tf.random.set_seed(42)
        main_input = Input(shape=(None, n_features), name='main_input')
        l1 = LSTM(size, activation='tanh', recurrent_dropout=0.2, return_sequences=True)(main_input)
        b1 = BatchNormalization()(l1)
        l2 = LSTM(size, activation='tanh', recurrent_dropout=0.2,return_sequences=False)(b1)
        b2 = BatchNormalization()(l2)
        l4 = Dense(2, name='Dense_1')(b2)

        output = Lambda(wtte.output_lambda, arguments={"init_alpha":init_alpha,"max_beta_value":100, "scalefactor":0.5})(l4)
        loss = wtte.loss(kind='continuous',reduce_loss=False).loss_function
        np.random.seed(42)
        tf.random.set_seed(42)
        self.model = Model(inputs=[main_input], outputs=[output])
        self.model.compile(loss=loss, optimizer=Nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3))
        mg_train = self.__batch_gen_train(X_train, y_train)
        mg_val = self.__batch_gen_train(X_val, y_val)
        self.model.fit_generator(mg_train, 
                            epochs=500,
                            steps_per_epoch = ceil(len(X_train) / self.batch_size),
                            validation_data=(mg_val),
                            validation_steps= ceil(len(X_val) / self.batch_size),
                            verbose=vb,
                            callbacks=[history,mc,es,csv_logger],
                            shuffle=False)
        try:
            self.model.load_weights(test_out_path + 'best_model.h5')
        except:
            self.model == None
        return

    def evaluate(self, X,y):
        # Make some predictions and put them alongside the real TTE and event indicator values
        if self.model == None:
            return np.nan, np.nan, np.nan
        else:
            mg_test = self.__batch_gen_test(X)
            nb_samples = len(X)
            y_pred = self.model.predict_generator(mg_test, steps= ceil(len(X) / self.batch_size))
        #     y_pred = model.predict(X)

            test_result = np.concatenate((y, y_pred), axis=1)
            test_results_df = pd.DataFrame(test_result, columns=['T', 'U', 'alpha', 'beta'])
            test_results_df['predicted_mode'] =   test_results_df[['alpha', 'beta']].apply(lambda row: weibull_mode(row[0], row[1]), axis=1)

            if self.resolution == 's':
                test_results_df['error (days)'] = (test_results_df['T'] - test_results_df['predicted_mode']) / 86400
                test_results_df["MAE"] = np.absolute(test_results_df["error (days)"])
                mae = mean_absolute_error(test_results_df['T'], test_results_df['predicted_mode']) / 86400
            elif self.resolution == 'h':
                test_results_df['error (days)'] = (test_results_df['T'] - test_results_df['predicted_mode']) / 24
                test_results_df["MAE"] = np.absolute(test_results_df["error (days)"])
                mae = mean_absolute_error(test_results_df['T'], test_results_df['predicted_mode']) / 24
            elif self.resolution == 'd':
                test_results_df['error (days)'] = (test_results_df['T'] - test_results_df['predicted_mode'])
                test_results_df["MAE"] = np.absolute(test_results_df["error (days)"])
                mae = mean_absolute_error(test_results_df['T'], test_results_df['predicted_mode'])
            test_results_df["Accurate"] = ((test_results_df["U"] == 1) & (test_results_df["MAE"] <= 2)) | \
                                          ((test_results_df["U"] == 0) & (test_results_df["predicted_mode"] >= test_results_df["T"]))
            accuracy = round(test_results_df["Accurate"].mean()*100,3)

            return test_results_df, mae, accuracy
    
    def __time_features(self,case):
        first_index = case.index[0]
        last_index  = case.index[-1]
        if case.iloc[0].loc["U"] == 1:
            endtime = case["CompleteTimestamp"][last_index]
        else:
            case.drop(last_index,axis=0,inplace=True)
            last_index   = case.index[-1]
            endtime = case["CompleteTimestamp"][last_index]

        starttime = case["CompleteTimestamp"][first_index]
        case["fvt1"] = case["CompleteTimestamp"].diff(periods=1).dt.total_seconds()
        case["fvt2"] = case["CompleteTimestamp"].dt.hour
        case["fvt3"] = (case["CompleteTimestamp"] - starttime).dt.total_seconds()
        case["T2E"]  = endtime - case["CompleteTimestamp"]
        case["D2E"] = case["T2E"].dt.days
        case["S2E"] = case["T2E"].dt.total_seconds()
        case["H2E"] = case["S2E"]/3600
        return case

    def __xy_split(self, data):
        X = data.groupby(["CaseID"]).apply(lambda df:self.__extract_X(df))
        X = np.array(tf.convert_to_tensor(X))
        y = data.groupby(["CaseID"]).apply(lambda df:self.__extract_y(df))
        y = np.array(tf.convert_to_tensor(y))
        return X,y

    def __extract_X(self, df):
        feature_idx = np.concatenate(\
                   np.where(df.columns.str.contains('ActivityID_')) + \
    #                np.where(df_helpdesk_preprocessed.columns.str.contains('weekday_')) + \
                   np.where(df.columns.str.contains('fvt'))
                  )
        x = []
        if len(df) < self.suffix:
            x.append(
                np.concatenate(
                    (np.full((self.suffix - df.shape[0] , len(feature_idx)),fill_value=0), 
                     df.values[0:self.suffix,feature_idx]), 
                     axis=0))
        else:
            x.append(df.values[0:self.suffix,feature_idx])

        x = np.hstack(np.array(x)).flatten()
        x = x.reshape((self.suffix, len(feature_idx)))

        return(x)

    def __extract_y(self, df):
        y = []
        if self.resolution == 's':
            time_idx = np.where(df.columns.str.contains("S2E"))[0]
        elif self.resolution == 'h':
            time_idx = np.where(df.columns.str.contains("H2E"))[0]
        elif self.resolution == 'd':
            time_idx = np.where(df.columns.str.contains("D2E"))[0]
        else:
            print("Defualt option chosen ==> daily")
            time_idx = np.where(df.columns.str.contains("D2E"))[0]

        if len(df) <= self.suffix: 
             y.append(df.values[-1,time_idx])
        else:
             y.append(df.values[self.suffix-1,time_idx])
        y.append(df.loc[0,"U"])
        y = np.hstack(np.array(y)).flatten()
        y.reshape((1, 2))
        return y

    
    def __batch_gen_train(self,X, y):
        n_batches = math.ceil(len(X) / self.batch_size)
        while True: 
            for i in range(n_batches):
                X_batch = X[i*self.batch_size:(i+1)*self.batch_size, :,:]
                y_batch = y[i*self.batch_size:(i+1)*self.batch_size, :]
                yield X_batch, y_batch
    
    def __batch_gen_test(self,X):
        n_batches = math.ceil(len(X) / self.batch_size)
        while True: 
            for i in range(n_batches):
                X_batch = X[i*self.batch_size:(i+1)*self.batch_size, :,:]
                yield X_batch



    
######################################################################################################################################
    
def balance_labels_nb(X,y):
    bins = np.arange(0,y[:,0].max()+24, 24)
    counts = np.histogram(y[:,0], bins=bins)[0]
    count_all = np.count_nonzero(counts)
    avg = math.ceil(len(y) / count_all)
    avg = 30
    for i in range(len(bins)-1):
        count_i = counts[i]
        if count_i > avg:
            num_delete = count_i - avg
            print(i, "count:",count_i, "avg:", avg, "to_del:", num_delete, "Act_count:", len(np.squeeze(np.argwhere((y[:,0] >= bins[i]) & (y[:,0] < bins[i+1])))))
            idx_delete = np.random.choice(np.squeeze(np.argwhere((y[:,0] >= bins[i]) & (y[:,0]< bins[i+1]))), num_delete, replace=False)
            X = np.delete(X, idx_delete , 0)
            y = np.delete(y, idx_delete , 0)
    return X, y



            

            




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

    plt.figure(figsize=(12,8))
    t=np.arange(0,300)
    
    plt.subplot(2,2,1)
    observed = results_df.loc[results_df['U'] == 1]
    sns.scatterplot(observed['T'], observed["predicted_mode"],hue=observed["Accurate"])
    plt.xlabel("Actual failure time",fontsize=12)
    plt.ylabel("Predicted failure time",fontsize=12)
    plt.title('Actual Vs. Predicted (Observed)',fontsize=18)
    plt.legend(loc = 'upper left')
    
    plt.subplot(2,2,2)
    censored = results_df.loc[results_df['U'] == 0]
    sns.scatterplot(censored['T'], censored["predicted_mode"],hue=censored["Accurate"])
    plt.xlabel("Time of observation",fontsize=12)
    plt.ylabel("Predicted failure time",fontsize=12)
    plt.title('Actual Vs. Predicted (Censored)',fontsize=18)
    plt.legend(loc = 'upper left')
        
    plt.subplot(2,2,(3,4))
    sns.distplot(results_df['error (days)'], bins=100, kde=True,hist=True, norm_hist=False, label="Error Rate")
    plt.title('Error distribution',fontsize=18)
    plt.legend(loc = 'upper left')
    plt.xlabel("Error Shift",fontsize=12)

#     plt.subplot(3,2,(5,6))
#     x = results_df.groupby(["T","U"]).agg({"Accurate":"mean"}).reset_index()
#     x["Error_Rate"] = 1-x["Accurate"]
#     sns.barplot(x= x["T"].astype("int"), y=x["Error_Rate"], hue=x["U"])
#     plt.title('Error rate per time step')
    
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
