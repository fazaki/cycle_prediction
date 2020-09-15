import pandas as pd
import numpy as np
import time
import math
from math import ceil
from collections import Counter
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.layers import BatchNormalization, Lambda, Input
from tensorflow.keras.optimizers import Nadam  # , RMSprop, Adam, Nadam
from tensorflow.keras.models import Model
from cycle_prediction.weibull_utils import weibull_mode, check_dir
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import wtte.wtte as wtte
from tensorflow.keras.callbacks import History, CSVLogger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# from matplotlib.gridspec import GridSpec
# import matplotlib.ticker as ticker
# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt


sd = 100
np.random.seed(sd)
tf.random.set_seed(sd)

#################################################
#################################################


class t2e:
    """A class for time to event data preprocessing, fitting and evaluation.
    Deals with datasets with the following format:

    +--------------------+-----------+--------------------------+
    | Column             | Data type | Content                  |
    +====================+===========+==========================+
    | CaseID             | int       | Case identifier          |
    +--------------------+-----------+--------------------------+
    | ActivityID         | int       | Activity identifier      |
    +--------------------+-----------+--------------------------+
    | CompleteTimestamp  | datetime  | Timestamp of the event   |
    +--------------------+-----------+--------------------------+

    """

    def __init__(
        self,
        dataset,
        prefix,
        resolution,
        censored,
        cen_prc=0,
        end_event_list=[],
        transform=False,
        fit_type='t2e',
    ):
        """Initializes the t2e object with the desired setup.

        Args:
            dataset (obj): Dataframe of the trace dataset in the form of:
            prefix (int): Number of history prefixes to train with.
            resolution (str): remaining time resolution
                {'s': 'seconds', 'h':'hours', 'd':'days'}
            censored (bool): Whether to randomely censor some traces.
            cen_prc (float): 0.0 -> 1.0, represents the percentage of
                censored traces to generate (used ONLY for development)
            end_event_list (obj): list of (int) containing the process's
                possible end events
            fit_type (str): t2e (default) => for furture development.
            transform (bool): Transform the output to a new space where
                it is less biased toward short traces.

        """
        self.dataset = dataset
        self.prefix = prefix
        self.censored = censored
        self.cen_prc = cen_prc
        self.censored_cases = []
        self.all_cases = []
        self.len_censored = 0
        self.len_observed = 0
        self.batch_size = 128
        self.end_event_list = end_event_list
        self.resolution = resolution
        self.transform = transform
        self.root = 1
        self.power = 1
        self.model = None
        if fit_type == 't2e':
            self.regression = False
        else:
            self.regression = True
        self.fit_time = 0

    def preprocess(self):
        """a method responsible of the creation of time and activity features.

        Args:
            __init__ Mandatory parameter:
                - dataset
                - prefix
                - resolution
                - end_event_list
            __init__ Optional parameter:
                - transform
                - fit_type

        Returns:
            Updated self.dataset: A pandas dataframe with the following format:

            +--------------------+-----------+-------------------------------+
            | Column             | Data type | Content                       |
            +====================+===========+===============================+
            | CaseID             | int       | Case identifier               |
            +--------------------+-----------+-------------------------------+
            | ActivityID         | int       | Activity identifier           |
            +--------------------+-----------+-------------------------------+
            | CompleteTimestamp  | datetime  | Timestamp of the event        |
            +--------------------+-----------+-------------------------------+
            | fvt1               | float     | delta time to the next event  |
            +--------------------+-----------+-------------------------------+
            | fvt2               | float     | hour since day start          |
            +--------------------+-----------+-------------------------------+
            | fvt3               | float     | hour since week start         |
            +--------------------+-----------+-------------------------------+
            | ActivityID_1       | int       | Activity in one-hot form      |
            +--------------------+-----------+-------------------------------+
            | ...                | ...       | ...                           |
            +--------------------+-----------+-------------------------------+
            | ActivityID_n       | ...       | ...                           |
            +--------------------+-----------+-------------------------------+
            | U                  | int       | 0/1 : censored/observed trace |
            +--------------------+-----------+-------------------------------+
            | T2E/D2E/S2E        | float     | Remaining time in seconds,    |
            |                    |           |  hours or days                |
            +--------------------+-----------+-------------------------------+

        """
        # Retrieve all sequences with their records count
        if self.end_event_list == []:
            raise ValueError(
             "end_event_list should not be empty when using self.preprocess()"
            )
        self.dataset.sort_values(
            ["CaseID", "CompleteTimestamp"], ascending=True)
        case_counts = self.dataset.groupby(["CaseID"]).count()

        # Only keep sequences that has at least prefix + 1 records
        to_drop = list(
            case_counts.loc[case_counts["ActivityID"] <= self.prefix].index)
        self.dataset = self.dataset.loc[~self.dataset["CaseID"].isin(
            to_drop)].reset_index(drop=True)
        # Store all cases length
        self.all_cases = self.dataset["CaseID"].unique()
        print('all cases', len(self.all_cases))
        # Create censored identifier based on end_event_list
        cen_dict = self.dataset.drop_duplicates('CaseID', keep='last')[
                            ['CaseID', 'ActivityID']]\
                       .set_index('CaseID')\
                       .to_dict()['ActivityID']
        for k, v in cen_dict.items():
            if v in self.end_event_list:
                cen_dict[k] = 1
            else:
                cen_dict[k] = 0
        # Assign the censorship state
        self.dataset['U'] = self.dataset['CaseID'].map(cen_dict)
        self.censored_cases = list(
            np.unique(self.dataset['CaseID']
                          .loc[self.dataset['U'] == 0]
                          .values))
        # process time features
        self.dataset.CompleteTimestamp = pd.to_datetime(
            self.dataset.CompleteTimestamp)
        self.dataset = self.dataset.groupby("CaseID").apply(
            lambda case: self.__time_features(case))
        self.dataset.fvt1.fillna(0, inplace=True)
        # process activity features
        dummy = pd.get_dummies(
            self.dataset["ActivityID"], prefix="ActivityID", drop_first=True)
        self.dataset = pd.concat([self.dataset, dummy], axis=1)
        last_step = self.dataset.drop_duplicates(
            subset=["CaseID"], keep='last')["ActivityID"].index
        self.dataset = self.dataset.drop(
            last_step, axis=0).reset_index(drop=True)
        # drop useless columns
        self.dataset = self.dataset.drop(
            ['CompleteTimestamp', 'T2E'], axis=1)
        # Set counts
        val_counts = Counter(cen_dict.values())
        self.len_censored = val_counts[0]
        self.len_observed = val_counts[1]

    def preprocess_dev(self):
        """This method is designed for developing purpose ONLY.

        - Generate censored traces from the observed traces
        - Feature engineering
        - One-hot-encoding

        Args:
            __init__ Mandatory parameter:
                - dataset
                - prefix
                - resolution
                - censored
                - cen_prc
            __init__ Optional parameter:
                - transform
                - fit_type

        Returns:
            :obj:`self`: updated self.dataset to a pandas dataframe with the\
                following format:

            +--------------------+-----------+-------------------------------+
            | Column             | Data type | Content                       |
            +====================+===========+===============================+
            | CaseID             | int       | Case identifier               |
            +--------------------+-----------+-------------------------------+
            | ActivityID         | int       | Activity identifier           |
            +--------------------+-----------+-------------------------------+
            | CompleteTimestamp  | datetime  | Timestamp of the event        |
            +--------------------+-----------+-------------------------------+
            | fvt1               | float     | delta time to the next event  |
            +--------------------+-----------+-------------------------------+
            | fvt2               | float     | hour since day start          |
            +--------------------+-----------+-------------------------------+
            | fvt3               | float     | hour since week start         |
            +--------------------+-----------+-------------------------------+
            | ActivityID_1       | int       | Activity in one-hot form      |
            +--------------------+-----------+-------------------------------+
            | ...                | ...       | ...                           |
            +--------------------+-----------+-------------------------------+
            | ActivityID_n       | ...       | ...                           |
            +--------------------+-----------+-------------------------------+
            | U                  | int       | 0/1 : censored/observed trace |
            +--------------------+-----------+-------------------------------+
            | T2E/D2E/S2E        | float     | Remaining time in seconds,    |
            |                    |           |  hours or days                |
            +--------------------+-----------+-------------------------------+

        """
        # Retrieve all sequences with their records count
        self.dataset.sort_values(
            ["CaseID", "CompleteTimestamp"], ascending=True)
        case_counts = self.dataset.groupby(["CaseID"]).count()

        # Only keep sequences that has at least prefix + 1 records
        to_drop = list(
            case_counts.loc[case_counts["ActivityID"] <= self.prefix].index)
        self.dataset = self.dataset.loc[~self.dataset["CaseID"].isin(
            to_drop)].reset_index(drop=True)
        self.all_cases = self.dataset["CaseID"].unique()
        print('all cases', len(self.all_cases))
        # Generate censored data:
        to_censor_from = list(
            self.dataset
                .groupby(["CaseID"])
                .count()
                .loc[case_counts["ActivityID"] > self.prefix+1].index
                )
        try:
            np.random.seed(sd)
            self.censored_cases = np.random.choice(to_censor_from, int(
                len(self.all_cases)*self.cen_prc), replace=False)
        except Exception:
            self.censored_cases = np.array(to_censor_from)

        print("first 10 censored cases", self.censored_cases[0:10])

        self.dataset["U"] = ~self.dataset['CaseID'].isin(self.censored_cases)*1
        self.dataset.CompleteTimestamp = pd.to_datetime(
            self.dataset.CompleteTimestamp)
        self.dataset = self.dataset.groupby("CaseID").apply(
            lambda case: self.__time_features_dev(case))
        self.dataset.fvt1.fillna(0, inplace=True)
        dummy = pd.get_dummies(
            self.dataset["ActivityID"], prefix="ActivityID", drop_first=True)
        self.dataset = pd.concat([self.dataset, dummy], axis=1)
        last_step = self.dataset.drop_duplicates(
            subset=["CaseID"], keep='last')["ActivityID"].index
        self.dataset = self.dataset.drop(
            last_step, axis=0).reset_index(drop=True)

    def smart_split(self, train_prc, val_prc, scaling):
        """Spliting the dataset to train, validation and test sets.

        The data nature requires a special function for this purpose

        Args:
            train_prc (float): Training percentage (include validation).
            val_prc (str): Validation percentage (% of the training set).
            scaling (bool): To scale numerical feature.

        Returns:
            X_train (object): tensor of shape [n_examples, prefix, n_features]

            X_test (object): tensor of shape [n_examples, prefix, n_features]

            X_val (object): tensor of shape [n_examples, prefix, n_features]

            y_train (object): tensor of shape [n_examples, 2]

            y_test (object): tensor of shape [n_examples, 2]

            y_val (object): tensor of shape [n_examples, 2]

            train_cases (int): Count of training traces

            val_cases (int): Count of validation traces

            test_cases (int): Count of testing traces

        """
        all_cases = set(self.dataset.CaseID.unique())
        censored = set(self.censored_cases)
        observed = list(all_cases.difference(censored))

        len_train = int(train_prc * len(observed))
        len_val = int(len_train*val_prc)
        cases_train = observed[0:len_train]
        cases_val = [cases_train.pop() for i in range(len_val)]
        cases_test = observed[len_train:]

        print("\tTotal Observed:", len(observed))
        print("\tTraining data Observed:", len(cases_train))
        print("\tTraining data Censored:", len(censored))
        if self.censored:
            cases_train = cases_train + list(censored)
        print("\tTraining data to use:", len(cases_train))

        print("\tValidation data:", len(cases_val))
        print("\tTesting data   :", len(cases_test))
        df_train = self.dataset.loc[self.dataset['CaseID'].isin(
            cases_train)].reset_index(drop=True)
        df_val = self.dataset.loc[self.dataset['CaseID'].isin(
            cases_val)].reset_index(drop=True)
        df_test = self.dataset.loc[self.dataset['CaseID'].isin(
            cases_test)].reset_index(drop=True)
        if scaling:
            sc = StandardScaler()
            df_train.loc[:, ["fvt1", "fvt2", "fvt3"]] = sc.fit_transform(
                df_train[["fvt1", "fvt2", "fvt3"]])
            df_val.loc[:, ["fvt1", "fvt2", "fvt3"]] = sc.transform(
                df_val[["fvt1", "fvt2", "fvt3"]])
            df_test.loc[:, ["fvt1", "fvt2", "fvt3"]] = sc.transform(
                df_test[["fvt1", "fvt2", "fvt3"]])

        X_train, y_train = self.__xy_split(df_train)
        X_val, y_val = self.__xy_split(df_val)
        X_test, y_test = self.__xy_split(df_test)

        if self.regression is True:
            return X_train, X_test, X_val, y_train[:, 0].reshape((-1, 1)), \
                y_test[:, 0].reshape((-1, 1)), y_val[:, 0].reshape((-1, 1)), \
                len(cases_train), len(cases_val), len(cases_test)
        return X_train, X_test, X_val, y_train, y_test, y_val, \
            len(cases_train), len(cases_val), len(cases_test)

    def fit(self, X_train, y_train, X_val, y_val, size, vb=True):
        """Fitting a time to event model using a GRU network.

        Args:
            X_train (object): training set input features of shape
                [n_examples, prefix, n_features]
            y_train (object): training set labels of shape
                [n_examples, n_features]
            X_val (object): validation set input features of shape
                [n_examples, prefix, n_features]
            y_val (object): validation set labels [n_examples, n_features]
            size (int): GRU units size.
            vb (bool): verbose (true/False)

        Returns:
            :obj:`self`: Updating self.model weights

        """
        if self.transform is True:
            self.root = 3
            self.root = 1/self.root
            self.power = 1/self.root
            y_train[:, 0] = y_train[:, 0]**self.root
            y_val[:, 0] = y_val[:, 0]**self.root
            print('Y_label has been transformed')
        out_path = "output/"
        check_dir(out_path)
        vb = vb
        if vb:
            print("\n")
        tte_mean_train = np.nanmean(y_train[:, 0].astype('float'))
        mean_u = np.nanmean(y_train[:, 1].astype('float'))
        init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0))
        init_alpha = init_alpha/mean_u
        history = History()
        cri = 'val_loss'
        csv_logger = CSVLogger(out_path + 'training.log',
                               separator=',', append=False)
        es = EarlyStopping(monitor=cri, mode='min', verbose=vb,
                           patience=15, restore_best_weights=False)
        mc = ModelCheckpoint(out_path + 'best_model.h5', monitor=cri,
                             mode='min', verbose=vb, save_best_only=True,
                             save_weights_only=True)
        n_features = X_train.shape[-1]

        np.random.seed(sd)
        tf.random.set_seed(sd)
        main_input = Input(shape=(None, n_features), name='main_input')
        l1 = GRU(size, activation='tanh', recurrent_dropout=0.2,
                 return_sequences=True)(main_input)
        b1 = BatchNormalization()(l1)
        l2 = GRU(size, activation='tanh', recurrent_dropout=0.2,
                 return_sequences=False)(b1)
        b2 = BatchNormalization()(l2)
        l4 = Dense(2, name='Dense_1')(b2)

        output = Lambda(wtte.output_lambda, arguments={
                        "init_alpha": init_alpha, "max_beta_value": 100,
                        "scalefactor": 0.5})(l4)
        loss = wtte.loss(kind='continuous', reduce_loss=False).loss_function
        np.random.seed(sd)
        tf.random.set_seed(sd)
        self.model = Model(inputs=[main_input], outputs=[output])
        self.model.compile(loss=loss, optimizer=Nadam(
            lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            schedule_decay=0.004, clipvalue=3))
        mg_train = self.__batch_gen_train(X_train, y_train)
        mg_val = self.__batch_gen_train(X_val, y_val)
        start = time.time()
        self.model.fit_generator(
            mg_train,
            epochs=500,
            steps_per_epoch=ceil(len(X_train) / self.batch_size),
            validation_data=(mg_val),
            validation_steps=ceil(len(X_val) / self.batch_size),
            verbose=vb,
            callbacks=[history, mc, es, csv_logger],
            shuffle=False
            )
        try:
            self.model.load_weights(out_path + 'best_model.h5')
            print('model loaded successfully')
        except Exception:
            self.model is None

        end = time.time()
        self.fit_time = np.round(end-start, 0)
        return

    def fit_regression(self, X_train, y_train, X_val, y_val, size, vb=True):

        out_path = "output/"
        check_dir(out_path)
        vb = vb
        if vb:
            print("\n")
        history = History()
        cri = 'va_loss'
        csv_logger = CSVLogger(out_path + 'training.log',
                               separator=',', append=False)
        es = EarlyStopping(monitor=cri, mode='min', verbose=vb,
                           patience=42, restore_best_weights=False)
        mc = ModelCheckpoint(out_path + 'best_model.h5', monitor=cri,
                             mode='min', verbose=vb, save_best_only=True,
                             save_weights_only=True)
        n_features = X_train.shape[-1]

        np.random.seed(sd)
        tf.random.set_seed(sd)
        main_input = Input(shape=(None, n_features), name='main_input')
        l1 = LSTM(size, activation='tanh', recurrent_dropout=0.2,
                  return_sequences=True)(main_input)
        b1 = BatchNormalization()(l1)
        l2 = LSTM(size/2, activation='tanh', recurrent_dropout=0.2,
                  return_sequences=False)(b1)
        b2 = BatchNormalization()(l2)
        output = Dense(1, name='output')(b2)
        np.random.seed(sd)
        tf.random.set_seed(sd)
        self.model = Model(inputs=[main_input], outputs=[output])
        self.model.compile(loss={'output': 'mae'}, optimizer=Nadam(
            lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            schedule_decay=0.004, clipvalue=3))
        mg_train = self.__batch_gen_train(X_train, y_train)
        mg_val = self.__batch_gen_train(X_val, y_val)
        start = time.time()
        self.model.fit_generator(mg_train,
                                 epochs=500,
                                 steps_per_epoch=ceil(
                                     len(X_train) / self.batch_size),
                                 validation_data=(mg_val),
                                 validation_steps=ceil(
                                     len(X_val) / self.batch_size),
                                 verbose=vb,
                                 callbacks=[history, mc, es, csv_logger],
                                 shuffle=False)
        try:
            self.model.load_weights(out_path + 'best_model.h5')
        except Exception:
            self.model is None

        end = time.time()
        self.fit_time = np.round(end-start, 0)
        return

    def predict(self, X):
        """ A method to predict the alpha and beta parameters defining the weibull pdf

        Args:
            X (object): Input array of [n_examples, prefix, n_features]

        Returns:
            y_pred (object): pandas dataframe with the shape [n_examples, 2]

        """
        if self.model is None:
            return None
        else:
            mg_test = self.__batch_gen_test(X)
            y_pred = self.model.predict_generator(
                mg_test, steps=ceil(len(X) / self.batch_size))
            y_pred_df = pd.DataFrame(y_pred, columns=['alpha', 'beta'])
            y_pred = y_pred_df[['alpha', 'beta']].apply(
                lambda row: weibull_mode(row[0], row[1]), axis=1)
            return y_pred

    def evaluate(self, X, y):
        """A method to evaluate the model prodiction with a test set

        Returns:
            mae (int): Mean absolute error of all predictions
            test_results_df: pandas dataframe with the following format

            +--------------+-----------+----------------------------------+
            | Column       | Data type | Content                          |
            +==============+===========+==================================+
            | T            | float     | True remaining time              |
            +--------------+-----------+----------------------------------+
            | U            | float     | Censored indicator               |
            +--------------+-----------+----------------------------------+
            | alpha        | float     | model prediction                 |
            +--------------+-----------+----------------------------------+
            | beta         | float     | model prediction                 |
            +--------------+-----------+----------------------------------+
            | T_pred       | float     | mode value of the generated pdf  |
            +--------------+-----------+----------------------------------+
            | error (days) | float     | Error in days                    |
            +--------------+-----------+----------------------------------+
            | MAE          | float     | Absolute error in days           |
            +--------------+-----------+----------------------------------+
            | Accurate     | boolean   | Based on predfined threshold     |
            +--------------+-----------+----------------------------------+

        """
        # Make some predictions and put them alongside the real TTE
        # and event indicator values
        if self.model is None:
            return np.nan, np.nan
        else:
            mg_test = self.__batch_gen_test(X)
            y_pred = self.model.predict_generator(
                mg_test, steps=ceil(len(X) / self.batch_size))
            test_result = np.concatenate((y, y_pred), axis=1)

            if self.regression is True:
                test_results_df = pd.DataFrame(
                    test_result, columns=['T', 'T_pred'])
                if self.resolution == 's':
                    test_results_df['error (days)'] = (
                        test_results_df['T'] - test_results_df['T_pred'])\
                            / 86400
                    test_results_df["MAE"] = np.absolute(
                        test_results_df["error (days)"])
                    mae = mean_absolute_error(
                        test_results_df['T'], test_results_df['T_pred'])\
                        / 86400
                elif self.resolution == 'h':
                    test_results_df['error (days)'] = (
                        test_results_df['T'] - test_results_df['T_pred']) / 24
                    test_results_df["MAE"] = np.absolute(
                        test_results_df["error (days)"])
                    mae = mean_absolute_error(
                        test_results_df['T'], test_results_df['T_pred']) / 24
                elif self.resolution == 'd':
                    test_results_df['error (days)'] = (
                        test_results_df['T'] - test_results_df['T_pred'])
                    test_results_df["MAE"] = np.absolute(
                        test_results_df["error (days)"])
                    mae = mean_absolute_error(
                        test_results_df['T'], test_results_df['T_pred'])
                test_results_df["Accurate"] = test_results_df["MAE"] <= 2
                # accuracy = round(test_results_df["Accurate"].mean()*100,3)

            if self.regression is False:
                test_results_df = pd.DataFrame(
                    test_result, columns=['T', 'U', 'alpha', 'beta'])
                test_results_df['T_pred'] = test_results_df[['alpha', 'beta']]\
                    .apply(lambda row: weibull_mode(row[0], row[1]), axis=1)
                if self.transform is True:
                    test_results_df['T_pred'] = test_results_df['T_pred']\
                        ** self.power
                    print("Y_label is restored")
                if self.resolution == 's':
                    test_results_df['error (days)'] = (
                        test_results_df['T'] - test_results_df['T_pred'])\
                            / 86400
                    test_results_df["MAE"] = np.absolute(
                        test_results_df["error (days)"])
                    mae = mean_absolute_error(
                        test_results_df['T'], test_results_df['T_pred'])\
                        / 86400
                elif self.resolution == 'h':
                    test_results_df['error (days)'] = (
                        test_results_df['T'] - test_results_df['T_pred']) / 24
                    test_results_df["MAE"] = np.absolute(
                        test_results_df["error (days)"])
                    mae = mean_absolute_error(
                        test_results_df['T'], test_results_df['T_pred']) / 24
                elif self.resolution == 'd':
                    test_results_df['error (days)'] = (
                        test_results_df['T'] - test_results_df['T_pred'])
                    test_results_df["MAE"] = np.absolute(
                        test_results_df["error (days)"])
                    mae = mean_absolute_error(
                        test_results_df['T'], test_results_df['T_pred'])
                test_results_df["Accurate"] =\
                    ((test_results_df["U"] == 1) &
                        (test_results_df["MAE"] <= 2))\
                    | ((test_results_df["U"] == 0) &
                        (test_results_df['T_pred'] >= test_results_df["T"]))
                # accuracy = round(test_results_df["Accurate"].mean()*100,3)
            return test_results_df, mae

    def get_cen_prc(self):
        try:
            return len(self.censored_cases)/len(self.all_cases)
        except Exception:
            return np.nan

    def __time_features(self, case):
        first_index = case.index[0]
        last_index = case.index[-1]
        endtime = case["CompleteTimestamp"][last_index]
        starttime = case["CompleteTimestamp"][first_index]
        case["fvt1"] = case["CompleteTimestamp"].diff(
            periods=1).dt.total_seconds()
        case["fvt2"] = case["CompleteTimestamp"].dt.hour
        case["fvt3"] = (case["CompleteTimestamp"] -
                        starttime).dt.total_seconds()
        case["T2E"] = endtime - case["CompleteTimestamp"]
        if self.resolution == 'd':
            case["D2E"] = case["T2E"].dt.days
        elif self.resolution == 's':
            case["S2E"] = case["T2E"].dt.total_seconds()
        elif self.resolution == 'h':
            case["H2E"] = case["T2E"].dt.total_seconds()/3600
        else:
            raise ValueError("d: day, h:hours, s:seconds are allowed as\
                time resolution")
        return case

    def __time_features_dev(self, case):
        case_length = len(case)
        first_index = case.index[0]
        last_index = case.index[-1]
        if case.iloc[0].loc["U"] == 1:
            endtime = case["CompleteTimestamp"][last_index]
        else:
            # possible cuts
            x = np.arange(1, case_length - self.prefix)
            # choose a cut
            random_censor = np.random.choice(x, 1)[0]
            for _ in range(random_censor):
                case.drop(last_index, axis=0, inplace=True)
                last_index = case.index[-1]
            endtime = case["CompleteTimestamp"][last_index]
        starttime = case["CompleteTimestamp"][first_index]
        case["fvt1"] = case["CompleteTimestamp"].diff(
            periods=1).dt.total_seconds()
        case["fvt2"] = case["CompleteTimestamp"].dt.hour
        case["fvt3"] = (case["CompleteTimestamp"] -
                        starttime).dt.total_seconds()
        case["T2E"] = endtime - case["CompleteTimestamp"]
        case["D2E"] = case["T2E"].dt.days
        case["S2E"] = case["T2E"].dt.total_seconds()
        case["H2E"] = case["S2E"]/3600
        return case

    def __xy_split(self, data):
        X = data.groupby(["CaseID"]).apply(lambda df: self.__extract_X(df))
        X = np.array(tf.convert_to_tensor(X))
        y = data.groupby(["CaseID"]).apply(lambda df: self.__extract_y(df))
        y = np.array(tf.convert_to_tensor(y))
        return X, y

    def __extract_X(self, df):
        feature_idx = np.concatenate(
            np.where(df.columns.str.contains('ActivityID_')) + \
            #  np.where(df.columns.str.contains('weekday_')) + \
            np.where(df.columns.str.contains('fvt'))
        )
        x = []
        if len(df) < self.prefix:
            x.append(
                np.concatenate(
                    (np.full((self.prefix - df.shape[0], len(feature_idx)),
                     fill_value=0), df.values[0:self.prefix, feature_idx]),
                    axis=0))
        else:
            x.append(df.values[0:self.prefix, feature_idx])

        x = np.hstack(np.array(x)).flatten()
        x = x.reshape((self.prefix, len(feature_idx)))

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

        if len(df) <= self.prefix:
            y.append(df.values[-1, time_idx])
        else:
            y.append(df.values[self.prefix-1, time_idx])
        y.append(df.loc[0, "U"])
        y = np.hstack(np.array(y)).flatten()
        y.reshape((1, 2))
        return y

    def __batch_gen_train(self, X, y):
        n_batches = math.ceil(len(X) / self.batch_size)
        while True:
            for i in range(n_batches):
                X_batch = X[i*self.batch_size:(i+1)*self.batch_size, :, :]
#                 if self.regression == True:
#                     y_batch = y[i*self.batch_size:(i+1)*self.batch_size]
#                 else:
                y_batch = y[i*self.batch_size:(i+1)*self.batch_size, :]
                yield X_batch, y_batch

    def __batch_gen_test(self, X):
        n_batches = math.ceil(len(X) / self.batch_size)
        while True:
            for i in range(n_batches):
                X_batch = X[i*self.batch_size:(i+1)*self.batch_size, :, :]
                yield X_batch
