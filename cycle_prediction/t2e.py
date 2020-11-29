import pandas as pd
import numpy as np
import time
import math
import logging
import random
from math import ceil
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
random.seed(sd)
np.random.seed(sd)
tf.random.set_seed(sd)

save_logs = False
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logger = logging.getLogger("log")
logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
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
        extra_censored=0,
        end_event_list=[],
        transform=False,
        fit_type='t2e',
        censored=True
    ):
        """Initializes the t2e object with the desired setup.

        Args:
            dataset (obj): Dataframe of the trace dataset in the form of:
            prefix (int): Number of history prefixes to train with.
            resolution (str): remaining time resolution
                {'s': 'seconds', 'h':'hours', 'd':'days'}
            extra_censored (float): Number of censored traces to create from
                complete traces.
            end_event_list (obj): list of (int) containing the process's
                possible end events
            transform (bool): Transform the output to a new space where
                it is less biased toward short traces.
            fit_type (str): t2e (default) => for furture development.
            censored (bool): Whether to use/ignore the censored traces
                (if found).

        """
        self.dataset = dataset
        self.prefix = prefix
        self.extra_censored = extra_censored
        self.censored = censored
        self.censored_cases = []
        self.observed_cases = []
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
        self.sc = None

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
            |                    |           | hours or days                 |
            +--------------------+-----------+-------------------------------+

        """
        # Safe condiftions
        if self.end_event_list == []:
            raise ValueError(
                "end_event_list should not be empty"
            )
        logger.info("Prefix = %d", self.prefix)
        # Set datatime colum
        self.dataset.CompleteTimestamp = pd.to_datetime(
            self.dataset.CompleteTimestamp)
        self.dataset.sort_values(
            ["CaseID", "CompleteTimestamp"], ascending=True)
        logger.info('Total    cases: %d', self.dataset["CaseID"].nunique())
        # Retrieve all sequences with their records count
        # Only keep sequences that has at least prefix + 1 records
        case_counts = self.dataset.groupby(["CaseID"]).count()
        below_prefix = list(
            case_counts.loc[case_counts["ActivityID"] <= self.prefix].index)
        self.dataset = self.dataset.loc[~self.dataset["CaseID"].isin(
            below_prefix)].reset_index(drop=True)

        # Store all cases length
        self.all_cases = self.dataset["CaseID"].unique()
        logger.info('Prefix   cases: %d', len(self.all_cases))

        # Create censorship dictionary based on end_event_list
        self.cen_dict = self.dataset.drop_duplicates('CaseID', keep='last')[
                                        ['CaseID', 'ActivityID']]\
                                    .set_index('CaseID')\
                                    .to_dict()['ActivityID']
        for k, v in self.cen_dict.items():
            if v in self.end_event_list:
                self.cen_dict[k] = 1
            else:
                self.cen_dict[k] = 0
        observed_cases = [k for k, v in self.cen_dict.items() if v == 1]
        random.shuffle(observed_cases)
        # Add extra censored cases
        i = 0
        for case in observed_cases:
            if i < self.extra_censored:
                tmp_df = self.dataset.loc[self.dataset['CaseID'] == case]
                if len(tmp_df) >= self.prefix + 2:
                    i += 1
                    last_index = tmp_df.index[-1]
                    # possible trims
                    x = np.arange(1, len(tmp_df) - self.prefix)
                    random_censor = np.random.choice(x, 1, replace=False)[0]
                    drop_idx = []
                    for j in range(random_censor):
                        drop_idx.append(last_index - j)
                    self.dataset['CaseID'] = self.dataset['CaseID']\
                                                 .drop(drop_idx, axis=0)
                    self.cen_dict[case] = 0
                else:
                    continue
            else:
                break

        # Assign the censorship state
        self.dataset['CaseID'].reset_index(drop=True, inplace=True)
        self.dataset['U'] = self.dataset['CaseID'].map(self.cen_dict)
        self.censored_cases = [k for k, v in self.cen_dict.items()
                               if int(v) == 0]
        self.observed_cases = [k for k, v in self.cen_dict.items()
                               if int(v) == 1]
        logger.info("Censored cases: %d", len(self.censored_cases))
        logger.info("Observed cases: %d", len(self.observed_cases))
        # process time features
        self.dataset = self.dataset.groupby("CaseID").apply(
                        lambda case: self.__time_features(case))
        self.dataset.fvt1.fillna(0, inplace=True)
        # process activity features
        dummy = pd.get_dummies(
            self.dataset["ActivityID"], prefix="ActivityID", drop_first=False)
        self.dataset = pd.concat([self.dataset, dummy], axis=1)

        # Remove last step
        last_step = self.dataset.drop_duplicates(
            subset=["CaseID"], keep='last')["ActivityID"].index
        self.dataset = self.dataset.drop(
            last_step, axis=0).reset_index(drop=True)
        # drop useless columns
        self.dataset = self.dataset.drop(
            ['CompleteTimestamp', 'T2E'], axis=1)
        end_list_cols = ['ActivityID_'+x for x in self.end_event_list]
        self.dataset = self.dataset.drop(
            end_list_cols, axis=1, errors='ignore')

    def xy_split(self, scaling):
        """Spliting the dataset into X_test [and y_test if available].

        Used when testing new traces

        Args:
            scaling (bool): To scale numerical feature.
            scaling_obj: fit object to scale the features.

        Returns:

            X_test (arr): tensor of shape [n_examples, prefix, n_features]
            y_test (arr): tensor of shape [n_examples, 2]
        """
        df_test = self.dataset
        if scaling:
            if self.sc is None:
                raise ValueError('scaling attribute is set to TRUE,\
                    while self.sc == None')
            df_test.loc[:, ["fvt1", "fvt2", "fvt3"]] = self.sc.transform(
                df_test[["fvt1", "fvt2", "fvt3"]])

        X_test, y_test = self.__xy_split(df_test)

        return X_test, y_test

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
        len_train = int(train_prc * len(self.observed_cases))
        len_val = int(len_train*val_prc)
        cases_train = self.observed_cases[0:len_train]
        cases_val = [cases_train.pop() for i in range(len_val)]
        cases_test = self.observed_cases[len_train:]

        if self.censored:
            cases_train = cases_train + self.censored_cases

        logger.info("Training   : %d", len(cases_train))
        logger.info("Validation : %d", len(cases_val))
        logger.info("Testing    : %d", len(cases_test))
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
            self.sc = sc

        X_train, y_train = self.__xy_split(df_train)
        X_val, y_val = self.__xy_split(df_val)
        X_test, y_test = self.__xy_split(df_test)
        X_train = np.float64(X_train)
        y_train = np.float64(y_train)
        X_val = np.float64(X_val)
        y_val = np.float64(y_val)
        X_test = np.float64(X_test)
        y_test = np.float64(y_test)
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
            logger.info('Y_label has been transformed')
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
            logger.info('model loaded successfully')
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
        """ A method to predict the alpha and beta parameters\
                defining the weibull pdf

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
                    logger.info("Y_label is restored")
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
        # case = case.sort_values('CompleteTimestamp')
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
            case["D2E"] = case["T2E"].dt.total_seconds()/86400
        elif self.resolution == 's':
            case["S2E"] = case["T2E"].dt.total_seconds()
        elif self.resolution == 'h':
            case["H2E"] = case["T2E"].dt.total_seconds()/3600
        else:
            raise ValueError("d: day, h:hours, s:seconds are allowed as\
                time resolution")
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
            logger.info("Defualt option chosen ==> daily")
            time_idx = np.where(df.columns.str.contains("D2E"))[0]

        if len(df) <= self.prefix:
            y.append(df.values[-1, time_idx])
        else:
            y.append(df.values[self.prefix-1, time_idx])

        # y.append(df.loc[0, "U"])
        y.append(self.cen_dict[df['CaseID'].unique()[0]])
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
