import datetime
import numpy as np
import json
import investpy as inv
from workadays import workdays as wd
from stocks.base_predict_stocks import BasePredictStocks


class PredictStocks:
    """Class for predict values of open and close stocks"""

    def base_prediction_stock(self, stock, order_columns, open_stock, close_stock, invert_order=False):
        """Base for train and predict next day's open and close stock price based on current data

        Param
        ======
            stock (obj): instance of obj stock with values and methods for prediction
            order_columns (list): order of columns for predict value to open and close
            open_stock (float): last value of open
            close_stock (float): last value of close
            invert_order (str): argument to invert variables open and close for prediction correctly
        Return
        ======
            value to open or close(according to the invert_order order).
        """
        if invert_order:
            open = open_stock
            close = np.nan
            features = np.array([[open, close]])
        else:
            open = np.nan
            close = close_stock
            features = np.array([[close, open]])
        return stock.load_new_prediction(features, order_columns, create_and_fit_model=True)[0]

    def calculate_percent_between_values(self, open_a, open_b, close_a, close_b):
        """Calculate percent to between values open and predicted open the same for values to close

        Param
        ======
            open_a (float): value of open previous
            open_b (float): value of open predict
            close_a (float): value of open previous
            close_b (float): value of open predict
        Return
        ======
            value to percent open and close.
        """
        return round(((open_b - open_a) / open_a * 100), 2), round(((close_b - close_a) / close_a * 100), 2)

    def predict_day(self, stock, order_columns, open_stock, close_stock, date_prediction):
        """Predict next day to open and close stock in order of parameter(order_columns)

        Param
        ======
            stock (obj): instance of obj stock with values and methods for prediction
            order_columns (list): order of columns for predict value to open and close
            open_stock (float): last value of open
            close_stock (float): last value of close
            date_prediction (date): date of prediction done
        Return
        ======
            dictionary with prediction of day.
        """
        open_predict = self.base_prediction_stock(stock, [order_columns[0], order_columns[1]], open_stock, close_stock)
        close_predict = self.base_prediction_stock(stock, [order_columns[1], order_columns[0]], open_stock, close_stock,
                                                   True)

        return {'name_stock': stock.name_stock,
                'date_prediction:': date_prediction,
                'open': round(open_predict, 2),
                'close': round(close_predict, 2),
                'percent_open': 0,
                'percent_close': 0,
                'percent_open_accumulated': 0,
                'percent_close_accumulated': 0}

    def prediction_days(self, stock, order_columns, days, open_stock_init, close_stock_init, date_prediction):
        """Predict days to open and close stock in order of parameter(order_columns)

        Param
        ======
            stock (obj): instance of obj stock with values and methods for prediction
            order_columns (list): order of columns for predict value to open and close
            days (int): count of days for prediction stock
            open_stock_init (float): last value of open(not predicted)
            close_stock_init (float): last value of close(not predicted)
            date_prediction (date): next date for prediction
        Return
        ======
            list with predictions and percents of stock in days.
        """
        predictions = []
        open_stock = open_stock_init
        close_stock = close_stock_init
        i = 0
        percent_open_accumulated = 0
        percent_close_accumulated = 0
        while i < days:
            result_day = self.predict_day(stock, order_columns, open_stock, close_stock,
                                          wd.workdays(date_prediction, i).strftime("%d-%m-%Y"))
            percent_open, percent_close = self.calculate_percent_between_values(open_stock,
                                                                                round(result_day['open'], 2),
                                                                                close_stock,
                                                                                round(result_day['close'], 2))
            percent_open_accumulated += percent_open
            percent_close_accumulated += percent_close
            result_day['percent_open'] = round(percent_open, 2)
            result_day['percent_close'] = round(percent_close, 2)
            result_day['percent_open_accumulated'] = round(percent_open_accumulated, 2)
            result_day['percent_close_accumulated'] = round(percent_close_accumulated, 2)
            predictions.append(result_day)
            open_stock = round(result_day['open'], 2)
            close_stock = round(result_day['close'], 2)
            i += 1

        return predictions

    def predict_stocks(self, stock_name, days, plot=False):
        """Predict new values of open and close

        obs: using all stocks use (list_stock_name = inv.get_stocks_list("brazil"))
        Param
        ======
            stock_name (list): list stocks for predition
            days (int): count of days for prediction stock
            plot (bool): show results
        Return
        ======
            json with predictions and percents of stock in days.
        """
        result_predict_stocks = []

        for stock in stock_name:
            stock_obj = BasePredictStocks(name_stock=stock)
            order_columns = ['Close', 'Open']
            open_stock_init = round(stock_obj.df_stock[-1:]['Open'][0], 2)
            close_stock_init = round(stock_obj.df_stock[-1:]['Close'][0], 2)
            date_prediction = wd.workdays(datetime.datetime.today(), 1)
            result_predict_stocks.append(
                self.prediction_days(stock_obj, order_columns, days, open_stock_init, close_stock_init,
                                     date_prediction))

        if plot:
            for result_stocks in result_predict_stocks:
                for stock_predict in result_stocks:
                    print(stock_predict)

        return json.dumps(result_predict_stocks)

    def train_test_model(self, stock_name):
        """Train test model for example

        Param
        ======
            stock_name (list): list stocks for prediction
        """
        for stock in stock_name:
            stock_obj = BasePredictStocks(name_stock=stock)
            order_columns = ['Close', 'Open']
            stock_obj.load_data_model_train_test_complete(order_columns)
            order_columns = ['Open', 'Close']
            stock_obj.load_data_model_train_test_complete(order_columns)
