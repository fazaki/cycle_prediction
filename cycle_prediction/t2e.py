import pandas as pd
import numpy as np
import datetime
import time
import logging
import random
import itertools
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Concatenate  # LSTM
from tensorflow.keras.layers import BatchNormalization, Lambda, Input
from tensorflow.keras.optimizers import Nadam  # RMSprop, Adam
from tensorflow.keras.models import Model
from cycle_prediction.weibull_utils import weibull_mode, check_dir
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import wtte.wtte as wtte
from tensorflow.keras.callbacks import History, CSVLogger, TensorBoard
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
        process_id_col='CaseID',
        event_type_col='ActivityID',
        extra_censored=0,
        end_event_list=[],
        dynamic_features=[],
        static_features=[],
        transform='log',
        fit_type='t2e',
        censored=True
    ):
        """Initializes the t2e object with the desired setup.

        Args:
            dataset (obj): Dataframe of the trace dataset in the form of:
            prefix (int): Number of history prefixes to train with.
            resolution (str): remaining time resolution
                {'s': 'seconds', 'h':'hours', 'd':'days'}
            process_id_col (str): column name to be used as process ID.
                default: 'CaseID'
            event_type_col (str): column name to be used as event type.
                default: 'ActivityID'
            extra_censored (int): Number of censored traces to artificially
                create from complete traces, default 0.
            end_event_list (list): list of (int) containing the process's
                possible end events
            dynamic_features (list): list of time varying feature columns
                to include in the model.
            static_features (list): list of time invariant feature columns
                to include in the model.
            transform (str): Transform the output to a new space where
                it is less biased toward short traces.
                Accepted values (None, 'log', 'power').
                Default: 'log'
            fit_type (str): 't2e' (default) => for furture development.
            censored (bool): Whether to use/ignore the censored traces
                (if found).

        """
        self.dataset = dataset
        self.process_id_col = process_id_col
        self.event_type_col = event_type_col
        self.prefix = prefix
        self.dynamic_features = dynamic_features
        self.static_features = static_features
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
        self.model = None
        self.scaling=True

        if fit_type == 't2e':
            self.regression = False
        else:
            self.regression = True
        self.fit_time = 0
        self.sc = None

    def train_val_test_split(self,train_prc=0.7,val_prc=0.45):
        logger.info('========================================================')
        logger.info("Prefix = %d", self.prefix)
        # Store all cases length
        logger.info('Total cases: %d', self.dataset[self.process_id_col]
                    .nunique())
        # Safe condiftions
        if self.end_event_list == []:
            raise ValueError(
                "end_event_list should not be empty"
                )
        # Set datatime column
        self.dataset.CompleteTimestamp = pd.to_datetime(
            self.dataset.CompleteTimestamp)
        self.dataset.sort_values(
            [self.process_id_col, "CompleteTimestamp"], ascending=True)

        # Retrieve all sequences with their records count
        # Only keep sequences that has at least prefix + 1 records
        case_counts = self.dataset.groupby([self.process_id_col]).count()
        below_prefix = list(
            case_counts.loc[case_counts[self.event_type_col]
                            <= self.prefix].index)
        self.dataset = self.dataset.loc[~self.dataset[self.process_id_col]
                                        .isin(below_prefix)]\
            .reset_index(drop=True)

        self.all_cases = self.dataset[self.process_id_col].unique().tolist()
        logger.info('Prefix cases: %d', len(self.all_cases))


        len_train = int(train_prc * len(self.all_cases))
        len_val = int(len_train*val_prc)
        self.cases_train = self.all_cases[0:len_train]
        self.cases_val = [self.cases_train.pop() for i in range(len_val)]
        self.cases_test = self.all_cases[len_train:]

        self.train = self.dataset.loc[self.dataset[self.process_id_col].isin(
            self.cases_train)].reset_index(drop=True)
        self.val = self.dataset.loc[self.dataset[self.process_id_col].isin(
            self.cases_val)].reset_index(drop=True)
        self.test = self.dataset.loc[self.dataset[self.process_id_col].isin(
            self.cases_test)].reset_index(drop=True)
        logger.info('========================')


    def preprocess(self, extra_censored = 0):
        """a method responsible for:
            1. Removing traces longer than the desired prefix.
            2. Creating dynamic and static featires as per the initialization

        Args:
            None

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
            | ActivityID_0       | bool      | Activity in one-hot form      |
            +--------------------+-----------+-------------------------------+
            | ...                | ...       | ...                           |
            +--------------------+-----------+-------------------------------+
            | ActivityID_n-1     | ...       | ...                           |
            +--------------------+-----------+-------------------------------+
            | Static_feature_0   | bool      | static feature in one-hot form|
            +--------------------+-----------+-------------------------------+
            | ...                | ...       | ...                           |
            +--------------------+-----------+-------------------------------+
            | Static_feature_n-1 | ...       | ...                           |
            +--------------------+-----------+-------------------------------+
            | U                  | int       | 0/1 : censored/observed trace |
            +--------------------+-----------+-------------------------------+
            | T2E/D2E/S2E        | float     | Remaining time in seconds,    |
            |                    |           | hours or days                 |
            +--------------------+-----------+-------------------------------+
        """
        # reduced dataset based on prefix
        self.dataset = pd.concat([self.train, self.val, self.test], axis=0).reset_index(drop=True)

        # Create censorship dictionary based on end_event_list
        self.cen_dict = self.dataset.drop_duplicates(
            self.process_id_col, keep='last')[
                [self.process_id_col, self.event_type_col]]\
            .set_index(self.process_id_col)\
            .to_dict()[self.event_type_col]
        for k, v in self.cen_dict.items():
            if v in self.end_event_list:
                self.cen_dict[k] = 1
            else:
                self.cen_dict[k] = 0
        observed_cases = [k for k, v in self.cen_dict.items() if v == 1]
        random.shuffle(observed_cases)

        for data_idx, df in enumerate([self.train, self.val, self.test]):
            
            if (data_idx == 0) and (extra_censored > 0):
                # Add extra censored cases
                i = 0
                logger.info('Extra censored percentage %s', str(extra_censored))
                logger.info('len(df) %s', str(len(self.cases_train)))
                extra_censored = int(extra_censored*len(self.cases_train))
                logger.info('Extra censored examples %s', str(extra_censored))
                for case in observed_cases:
                    if i < extra_censored:
                        tmp_df = df.loc[df[self.process_id_col] == case]
                        if len(tmp_df) >= self.prefix + 2:
                            i += 1
                            last_index = tmp_df.index[-1]
                            # possible trims
                            x = np.arange(1, len(tmp_df) - self.prefix)
                            random_censor = np.random.choice(x, 1, replace=False)[0]
                            drop_idx = []
                            for j in range(random_censor):
                                drop_idx.append(last_index - j)
                            df[self.process_id_col] =\
                                df[self.process_id_col]\
                                    .drop(drop_idx, axis=0)
                            self.cen_dict[case] = 0
                        else:
                            continue
                    else:
                        break

            # Assign the censorship state
            df[self.process_id_col].reset_index(drop=True, inplace=True)
            df['U'] = df[self.process_id_col]\
                                    .map(self.cen_dict)
            self.censored_cases = [k for k, v in self.cen_dict.items()
                                if int(v) == 0]
            self.observed_cases = [k for k, v in self.cen_dict.items()
                                if int(v) == 1]

            if data_idx == 0:
                logger.info("TRAINING SET")
                logger.info("Censored cases: %d", df.loc[df['U']==0][self.process_id_col].nunique())
            elif data_idx == 1:
                logger.info("VALIDATION SET")
            elif data_idx == 2:
                logger.info("TEST SET")
            logger.info("Observed cases: %d", df.loc[df['U']==1][self.process_id_col].nunique())

            # process time features
            df = df.groupby(self.process_id_col).apply(
                            lambda case: self.__time_features(case))
            df.fvt1.fillna(0, inplace=True)
            if self.transform == 'log':
                logger.info('Y_label has been transformed to logarithmic scale')
                df['D2E'] = np.log(df['D2E'] + 1)
            elif self.transform == 'power':
                logger.info('Y_label has been transformed with (1/3) root')
                df['D2E'] = df['D2E']**3
            logger.info('========================')

            if data_idx == 0:
                ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
                cf = self.dynamic_features + self.static_features
                ohe.fit(df[cf])
                feature_labels = np.array(ohe.categories_)
                colnames = []
                for i, (fname, numbers) in enumerate(zip(cf,feature_labels)):
                    colnames.append([fname+'_'+str(j) for j in numbers])
                    colnames = list(itertools.chain(*colnames))
            feature_arr = ohe.transform(df[cf]).toarray()
            feature_arr = pd.DataFrame(feature_arr, columns=colnames)
            df = pd.concat([df, feature_arr], axis=1)

            # Remove last step
            last_step = df.drop_duplicates(
                subset=[self.process_id_col], keep='last')[self.event_type_col]\
                .index
            df = df.drop(
                last_step, axis=0).reset_index(drop=True)
            # drop useless columns
            df = df.drop(
                ['CompleteTimestamp', 'T2E'], axis=1)

            dyn_features_idx = np.where(df.columns.str
                                            .contains('fvt'))[0]
            sta_features_idx = np.array([])

            for i in range(len(self.dynamic_features)):
                dyn_features_idx = np.concatenate(
                    (dyn_features_idx,
                    [np.where(df.columns.str.contains(str(x) + '_'))
                    for x in self.dynamic_features][i][0]), axis=0)

            for i in range(len(self.static_features)):
                sta_features_idx = np.concatenate(
                    (sta_features_idx,
                    [np.where(df.columns.str.contains(str(x) + '_'))
                    for x in self.static_features][i][0]), axis=0)

            self.dyn_features_idx = sorted(dyn_features_idx)
            self.sta_features_idx = sorted(sta_features_idx)
            self.sta_features_idx = [int(x) for x in self.sta_features_idx]
            if data_idx == 0:
                self.train = df.copy()
            elif data_idx == 1:
                self.val = df.copy()
            elif data_idx == 2:
                self.test = df.copy()

        logger.info("Dynamic Features Idx: %s", self.dyn_features_idx)
        logger.info("Static  Features Idx: %s", self.sta_features_idx)


    # def xy_split(self, scaling):
    #     """Spliting the dataset into X_test [and y_test if available].

    #     Used when testing new traces

    #     Args:
    #         scaling (bool): To scale numerical feature.
    #         scaling_obj: fit object to scale the features.

    #     Returns:

    #         X_test (arr): tensor of shape [n_examples, prefix, n_features]
    #         y_test (arr): tensor of shape [n_examples, 2]
    #     """
    #     df_test = self.dataset
    #     if scaling:
    #         if self.sc is None:
    #             raise ValueError('scaling attribute is set to TRUE,\
    #                 while self.sc == None')
    #         df_test.loc[:, ["fvt1", "fvt2", "fvt3"]] = self.sc.transform(
    #             df_test[["fvt1", "fvt2", "fvt3"]])

    #     X_test, y_test = self.__xy_split(df_test)

    #     return X_test, y_test

    def xy_split(self):
        """Spliting the dataset to train, validation and test sets.

        The data nature requires a special function for this purpose

        Args:
            train_prc (float): Training percentage (include validation).
            val_prc (str): Validation percentage (% of the training set).
            scaling (bool): To scale numerical feature.

        Returns:
            X_train (object): tensor of shape [n_examples, prefix, n_features]

            X_val (object): tensor of shape [n_examples, prefix, n_features]

            X_test (object): tensor of shape [n_examples, prefix, n_features]

            y_train (object): tensor of shape [n_examples, 2]

            y_val (object): tensor of shape [n_examples, 2]

            y_test (object): tensor of shape [n_examples, 2]
        """
        # len_train = int(train_prc * len(self.observed_cases))
        # len_val = int(len_train*val_prc)
        # self.cases_train = self.observed_cases[0:len_train]
        # self.cases_val = [self.cases_train.pop() for i in range(len_val)]
        # self.cases_test = self.observed_cases[len_train:]

        # if self.censored:
        #     self.cases_train = self.cases_train + self.censored_cases

        # logger.info("Training   : %d \t (Obs:%d, Cen:%d)", len(self.cases_train),
        #             len_train-len_val, len(self.censored_cases))
        # logger.info("Validation : %d", len(self.cases_val))
        # logger.info("Testing    : %d", len(self.cases_test))
        # df_train = self.dataset.loc[self.dataset[self.process_id_col].isin(
        #     self.cases_train)].reset_index(drop=True)
        # df_val = self.dataset.loc[self.dataset[self.process_id_col].isin(
        #     self.self.cases_val)].reset_index(drop=True)
        # df_test = self.dataset.loc[self.dataset[self.process_id_col].isin(
        #     self.self.cases_test)].reset_index(drop=True)
        if self.scaling:
            sc = StandardScaler()
            self.train.loc[:, ["fvt1", "fvt2", "fvt3"]] = sc.fit_transform(
                self.train[["fvt1", "fvt2", "fvt3"]])
            self.val.loc[:, ["fvt1", "fvt2", "fvt3"]] = sc.transform(
                self.val[["fvt1", "fvt2", "fvt3"]])
            self.test.loc[:, ["fvt1", "fvt2", "fvt3"]] = sc.transform(
                self.test[["fvt1", "fvt2", "fvt3"]])
            self.sc = sc

        X_train, y_train = self.__xy_split(self.train)
        X_val,   y_val   = self.__xy_split(self.val)
        X_test,  y_test  = self.__xy_split(self.test)
        # X_train = np.float64(X_train)
        # y_train = np.float64(y_train)
        # X_val = np.float64(X_val)
        # y_val = np.float64(y_val)
        # X_test = np.float64(X_test)
        # y_test = np.float64(y_test)

        if self.regression is True:
            y_train = y_train[:, 0].reshape((-1, 1))
            y_val = y_val[:, 0].reshape((-1, 1))
            y_test = y_test[:, 0].reshape((-1, 1))

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, X_train, y_train, size_dyn, size_sta):
        """Build time to event model using a GRU network.

        Args:
            X_train (object): training set input features of shape
                [n_examples, prefix, n_features]
            y_train (object): training set labels of shape
                [n_examples, n_features]
            size_dyn (int): GRU units size.
            size_sta (int): Static branch hidden layer size (optional)

        Returns:
            initialize self.model

        """
        logger.info('Initializing time to event model ...')
        # check if we have static features
        static_flag = False
        if X_train[1].shape[2] != 0:
            static_flag = True
            X_train_static = X_train[1]
            n_features_static = X_train_static.shape[-1]
            # Static model
            static_input = Input(shape=(n_features_static),
                                 name='static_input')
            dense_static1 = Dense(size_sta,
                                  name='hidden_static1')(static_input)
            bs1 = BatchNormalization()(dense_static1)
            # dense_static2 = Dense(size_sta//2, name='hidden_static2')(bs1)
            # bs2 = BatchNormalization()(dense_static2)
            static_output = Dense(1,
                                  name='static_output',
                                  activation='sigmoid')(bs1)
        X_train = X_train[0]

        tte_mean_train = np.nanmean(y_train[:, 0].astype('float'))
        mean_u = np.nanmean(y_train[:, 1].astype('float'))
        init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0))
        init_alpha = init_alpha/mean_u
        n_features = X_train.shape[-1]

        # Fixing seeds
        np.random.seed(sd)
        tf.random.set_seed(sd)

        # Main model
        main_input = Input(shape=(None, n_features), name='main_input')
        l1 = GRU(size_dyn, activation='tanh', recurrent_dropout=0.25,
                 return_sequences=True)(main_input)
        b1 = BatchNormalization()(l1)
        l2 = GRU(size_dyn//2, activation='tanh', recurrent_dropout=0.25,
                 return_sequences=False)(b1)
        b2 = BatchNormalization()(l2)
        if static_flag:
            dynamic_output = Dense(2, name='Dense_main')(b2)
            merged = Concatenate()([dynamic_output, static_output])
            l4 = Dense(2, name='output')(merged)
        else:
            l4 = Dense(2, name='output')(b2)

        output = Lambda(wtte.output_lambda, name="lambda_layer", arguments={
                        "init_alpha": init_alpha, "max_beta_value": 100,
                        "scalefactor": 0.5})(l4)
        loss = wtte.loss(kind='continuous', reduce_loss=False).loss_function
        np.random.seed(sd)
        tf.random.set_seed(sd)
        if static_flag:
            self.model = Model(inputs=[main_input, static_input],
                               outputs=[output])
        else:
            self.model = Model(inputs=[main_input], outputs=[output])
        self.model.compile(loss=loss, optimizer=Nadam(
            lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            schedule_decay=0.004, clipvalue=3))
        # self.model.compile(loss=loss, optimizer=RMSprop(
        #     lr=0.001, clipvalue=3))
        return

    def fit(self, X_train, y_train, X_val, y_val, bs=64,
            exp_dir=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            vb=True):
        """Fitting a time to event model using a GRU network.

            Args:
                X_train (object): training set input features of shape
                    [n_examples, prefix, n_features]
                y_train (object): training set labels of shape
                    [n_examples, n_features]
                X_val (object): validation set input features of shape
                    [n_examples, prefix, n_features]
                y_val (object): validation set labels [n_examples, n_features]
                bs (int): batch size
                exp_dir (str): tensorboard path
                vb (bool): verbose (true/False)

            Returns:
                :obj:`self`: fit self.model weights

            """
        out_path = "../output/"
        check_dir(out_path)
        vb = vb
        history = History()
        csv_logger = CSVLogger(out_path + 'training.log',
                               separator=',', append=False)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=vb,
                           patience=100, restore_best_weights=False)
        mc = ModelCheckpoint(out_path + 'best_model.h5', monitor='val_loss',
                             mode='min', verbose=vb, save_best_only=True,
                             save_weights_only=True)
        log_dir = out_path + 'tensorboard/' + exp_dir + '/'
        tensorboard = TensorBoard(log_dir=log_dir,
                                  histogram_freq=1,
                                  profile_batch=100000000)

        X_train_dyn = X_train[0]
        X_val_dyn = X_val[0]
        X_train_sta = X_train[1].reshape((X_train[1].shape[0],
                                          X_train[1].shape[2]))
        X_val_sta = X_val[1].reshape((X_val[1].shape[0],
                                      X_val[1].shape[2]))

        if X_train[1].shape[2] != 0:
            input_list = [X_train_dyn, X_train_sta]
            val_list = [X_val_dyn, X_val_sta]
            logger.info("Static   Input Shape: %s", X_train_sta.shape)
        else:
            input_list = [X_train_dyn]
            val_list = [X_val_dyn]
            logger.info("No Static features added")
        logger.info("Variable Input Shape: %s", X_train_dyn.shape)
        logger.info("Output   Input Shape: %s", y_train.shape)

        logger.info('Fitting model ... Batch size: %d', bs)
        start = time.time()
        self.model.fit(
            input_list,
            [y_train],
            epochs=1500,
            # batch_size=self.batch_size,
            batch_size=bs,
            # steps_per_epoch=ceil(X_train_dyn.shape[0] / self.batch_size),
            validation_data=(
                    val_list,
                    [y_val]
                ),
            # validation_steps=ceil(X_val_dyn.shape[0] / self.batch_size),
            verbose=vb,
            callbacks=[tensorboard, history, mc, es, csv_logger],
            shuffle=False
            )
        try:
            self.model.load_weights(out_path + 'best_model.h5')
            logger.info('Model loaded successfully')
        except Exception:
            self.model is None

        end = time.time()
        self.fit_time = np.round(end - start, 0)
        return

    def predict(self, X):
        """A method to predict alpha & beta parameter for a given prefix of trace
            after using the fit method to train the self.model

        Args:
            X (tensor): Input array of size [n_examples, prefix, n_features]

        Returns:
            y_pred (object): pandas dataframe with the shape [n_examples, 2]

        """
        logger.info("Predicting test set ...")
        if self.model is None:
            return None
        else:
            X_dyn = X[0]
            X_sta = X[1].reshape(X[1].shape[0], X[1].shape[2])
            y_pred = self.model.predict([X_dyn, X_sta],
                                        batch_size=self.batch_size)
            y_pred_df = pd.DataFrame(y_pred, columns=['alpha', 'beta'])
            y_pred = y_pred_df[['alpha', 'beta']].apply(
                lambda row: weibull_mode(row[0], row[1]), axis=1)
            return y_pred

    def evaluate(self, X, y):
        """A method to predict and evaluate the self.model after using the fit
            method, given a test set with known ground truth

        Args:
            X (tensor): Input array of size [n_examples, prefix, n_features]
            y (tensor): Output array of size [nexample, 2]

        Returns:
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
            | Accurate     | boolean   | For development purpose          |
            +--------------+-----------+----------------------------------+

            mae (float): Mean absolute error of all predictions
        """
        # Make some predictions and put them alongside the real TTE
        # and event indicator values
        logger.info("Evaluating test set ...")
        if self.model is None:
            return np.nan, np.nan
        else:
            X_dyn = X[0]
            X_sta = X[1].reshape(X[1].shape[0], X[1].shape[2])
            y_pred = self.model.predict([X_dyn, X_sta],
                                        batch_size=self.batch_size)
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

            else:
                test_results_df = pd.DataFrame(
                    test_result, columns=['T', 'U', 'alpha', 'beta'])
                test_results_df['T_pred'] = test_results_df[['alpha', 'beta']]\
                    .apply(lambda row: weibull_mode(row[0], row[1]), axis=1)
                if self.transform == 'log':
                    test_results_df['T'] = np.exp(test_results_df['T']) - 1
                    test_results_df['T_pred'] =\
                        np.exp(test_results_df['T_pred']) - 1
                    logger.info("Y_label is restored")
                elif self.transform == 'power':
                    test_results_df['T'] = test_results_df['T'] ** 3
                    test_results_df['T_pred'] = test_results_df['T_pred'] ** 3
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
            logger.info("MAE: %f, unique redictions: %d ", mae,
                        len(test_results_df['T_pred'].unique()))

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
        X = data.groupby([self.process_id_col]).apply(
                            lambda df: self.__extract_X(df))
        X_dyn = [arr[0] for arr in X]
        X_sta = [arr[1] for arr in X]
        X_dyn = np.array(tf.convert_to_tensor(X_dyn))
        X_sta = np.array(tf.convert_to_tensor(X_sta))
        X_sta = X_sta[:, 0, :].reshape(X_sta.shape[0], 1, X_sta.shape[2])
        y = data.groupby([self.process_id_col]).apply(
                            lambda df: self.__extract_y(df))
        y = np.array(tf.convert_to_tensor(y))
        return((X_dyn, X_sta), y)

    def __extract_X(self, df):
        x_dyn = []
        x_sta = []
        if len(df) < self.prefix:
            logger.error("this should NOT happen !!")
            x_dyn.append(np.concatenate(
                (np.full((self.prefix - df.shape[0],
                          len(self.dyn_features_idx)),
                 fill_value=0),
                 df.values[0:self.prefix, self.dyn_features_idx]),
                axis=0))
        else:
            x_dyn.append(df.values[0:self.prefix, self.dyn_features_idx])
        x_dyn = np.hstack(np.array(x_dyn)).flatten()
        x_dyn = x_dyn.reshape((self.prefix, len(self.dyn_features_idx)))

        if len(df) < self.prefix:
            logger.info("this should NOT happen !!")
            x_sta.append(np.concatenate(
                (np.full((self.prefix - df.shape[0],
                          len(self.sta_features_idx)),
                 fill_value=0),
                 df.values[0:self.prefix, self.sta_features_idx]),
                axis=0))
        else:
            x_sta.append(df.values[0:self.prefix, self.sta_features_idx])
        x_sta = np.hstack(np.array(x_sta)).flatten()
        x_sta = x_sta.reshape((self.prefix, len(self.sta_features_idx)))

        return x_dyn, x_sta

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
        case_nb = df[self.process_id_col].unique()[0]
        y.append(self.cen_dict[case_nb])
        y = np.hstack(np.array(y)).flatten()
        y.reshape((1, 2))
        return y
