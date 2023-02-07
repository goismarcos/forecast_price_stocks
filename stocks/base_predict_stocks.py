import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_squared_error


class BasePredictStocks:
    """This class is base for predict values of stocks"""

    def __init__(self, name_stock):
        """Initialize features stocks.

        Param
        ======
            name_stock (str): name of stock
        """
        self.name_stock = name_stock
        self.order_columns = None

        # Dataframes used for predictions
        self.df_stock = None
        self.df_train = None
        self.df_test = None
        self.df_stock_predict = None

        # Numpy arrays for model prediction
        self.train = None
        self.test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # model of Class
        self.model = None

        # scale of normalization data
        self.scaler = MinMaxScaler()

        # numpy arrays for predict new values
        self.X_features_predict_train = np.zeros(1)
        self.y_stock_predict_train = None
        self.X_features_predict = np.zeros(1)
        self.y_stock_predict = None

        # initialize data
        yf.pdr_override()
        self.read_data()

    def load_data_model_train_test_complete(self, order_columns,):
        """Do clear data/model/train/test for predictions futures.

        Param
        ======
            order_columns (list): list of columns ordered for prediction
        Return
        ======
            stocks predicted and trained.
        """
        self.order_columns = order_columns
        self.normalization_separe_train_test()
        self.create_model()
        self.load_fit_model(fit=True)
        self.y_stock_predict = self.model.predict(self.X_test)
        self.transform_predict_data_values_original()
        self.plot_predict_stocks()
        self.rmse_model()
        return self.y_stock_predict

    def load_new_prediction(self, features, order_columns, create_and_fit_model=False):
        """Generation prediction of stock.

        Param
        ======
            features (numpy array): new features of stock for prediction
            order_columns (list): list of columns ordered for prediction
            create_and_fit_model (bool): create and fiting model (Default false)
        Return
        ======
            stock predicted.
        """
        self.X_features_predict = features
        self.order_columns = order_columns
        self.normalization_separe_train_test(percent_division=0)
        if create_and_fit_model:
            self.create_model()
            self.load_fit_model(fit=True)
        self.y_stock_predict = self.model.predict(self.X_features_predict)
        self.transform_predict_data_values_original()
        self.X_features_predict = np.zeros(1)
        return self.y_stock_predict

    def read_data(self):
        """Read data of stock for predict."""
        try:
            start = datetime.datetime(1900, 1, 1)
            self.df_stock = web.get_data_yahoo(self.name_stock, start).drop(columns=['Adj Close'])
        except Exception as e:
            print('Error data read, please check this name stock!' + str(e))

    def normalization_separe_train_test(self, percent_division=0.7):
        """Method for normalization data and separe between train and test to use in model LSTM.

        Param
        ======
            percent_division (float): percent for division train and test.
            features_predict (numpy array): predict one value (model in production, predict value future).
        """
        try:
            self.df_stock = self.df_stock[self.order_columns]
            series = self.df_stock.values

            # normalization (max 1 e min 0)
            series = self.scaler.fit_transform(series)

            if percent_division == 0:
                '''transform data for train'''
                # division for train
                self.train = series[:int(len(self.df_stock) - 1)]

                # split into input and outputs
                self.X_train, self.y_train = self.train[:, :-1], self.train[:, -1]

                # reshape input to be 3D [samples, timestamps, features]
                self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))

                '''transform new values for predict'''
                # normalization (max 1 e min 0)
                self.X_features_predict = self.scaler.transform(self.X_features_predict)
                # drop value nan
                self.X_features_predict = self.X_features_predict[~np.isnan(self.X_features_predict)].reshape(1, -1)

                # reshape input to be 3D [samples, timestamps, features]
                self.X_features_predict = self.X_features_predict.reshape((self.X_features_predict.shape[0],
                                                                           1,
                                                                           self.X_features_predict.shape[1]))
            else:
                # division between train and test
                n, p = len(self.df_stock), percent_division
                self.train = series[:int(n * p)]
                self.test = series[int(n * p):]

                # split into input and outputs
                self.X_train, self.y_train = self.train[:, :-1], self.train[:, -1]
                self.X_test, self.y_test = self.test[:, :-1], self.test[:, -1]

                # reshape input to be 3D [samples, timestamps, features]
                self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
                self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        except Exception as e:
            print('Error normalization data, please check your dataFrame or Method of normalization!' + str(e))

    def create_model(self, summary=False):
        """Create LSTM model for predict values.

        Param
        ======
            summary (bool): show result summary of model.
        """
        try:
            inputs = keras.layers.Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
            dense = keras.layers.Dense(32, activation='relu')(inputs)
            lstm_out = keras.layers.LSTM(50)(dense)

            outputs = keras.layers.Dense(1)(lstm_out)

            self.model = keras.Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
            if summary:
                self.model.summary()
        except Exception as e:
            print('Error create model, please check your creating of model!' + str(e))

    def load_fit_model(self, fit=False):
        """fit model LSTM.

        Param
        ======
            fit (bool): training or not model (new weights for model or not) .
        """
        try:
            if fit:
                my_callbacks = [keras.callbacks.ModelCheckpoint(filepath='weights_model.h5')]
                self.model.fit(x=self.X_train, y=self.y_train, batch_size=32, epochs=10, callbacks=my_callbacks)
            else:
                self.model = keras.models.load_model("./weights_model.h5")
        except Exception as e:
            print('Error fit model, please check your fiting of model!' + str(e))

    def transform_predict_data_values_original(self):
        """Transformation data to format original."""
        try:
            # if predict values futures unknown
            if self.X_features_predict[0] > 0:
                self.X_features_predict = np.append(self.X_features_predict, np.nan)
                self.df_stock_predict = pd.DataFrame(self.X_features_predict).T
                self.df_stock_predict[self.df_stock_predict.shape[1] - 1] = self.y_stock_predict
                self.df_stock_predict = pd.DataFrame(self.scaler.inverse_transform(self.df_stock_predict))
                self.y_stock_predict = self.df_stock_predict[self.df_stock_predict.shape[1] - 1]
            else:
                self.df_train = pd.DataFrame(self.scaler.inverse_transform(self.train))
                self.df_test = pd.DataFrame(self.scaler.inverse_transform(self.test))

                self.df_stock_predict = pd.DataFrame(self.test)
                self.df_stock_predict[self.df_stock_predict.shape[1] - 1] = self.y_stock_predict
                self.df_stock_predict = pd.DataFrame(self.scaler.inverse_transform(self.df_stock_predict))

        except Exception as e:
            print('Error transformation data to format original, please check your data endangered!' + str(e))

    def plot_predict_stocks(self):
        """Show graphic of prediction of resulting model."""
        try:
            n_train, n_test = len(self.df_train[self.df_train.shape[1] - 1]), len(
                self.df_test[self.df_test.shape[1] - 1])
            plt.figure(figsize=(20, 10))
            plt.plot(np.arange(n_train), self.df_train[self.df_train.shape[1] - 1], label='train')
            plt.plot(np.arange(n_train, n_train + n_test), self.df_test[self.df_test.shape[1] - 1], label='test')
            plt.plot(np.arange(n_train, n_train + n_test), self.df_stock_predict[self.df_stock_predict.shape[1] - 1],
                     label='predict')
            plt.title('Stock')
            plt.grid()
            plt.legend()
            plt.show()
        except Exception as e:
            print('Error in plot data stock, please check your method of plot data!' + str(e))

    def rmse_model(self):
        """Calculate R.M.S.E of model generated."""
        try:
            return print(f'R.M.S.E: {np.sqrt(mean_squared_error(self.df_test[self.df_test.shape[1] - 1], self.df_stock_predict[self.df_stock_predict.shape[1] - 1]))}')
        except Exception as e:
            print('Error in r.m.s.e calculate, please check your method of r.m.s.e calculate!' + str(e))
