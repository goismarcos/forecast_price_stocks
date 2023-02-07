from stocks.predict_stocks import PredictStocks
import boto3
from datetime import datetime


def get_stocks_predict(list_stocks_names):
    predict_stocks_obj = PredictStocks()
    results_predicts_stocks = predict_stocks_obj.predict_stocks(stock_name=list_stocks_names, days=5, plot=False)
    return results_predicts_stocks


if __name__ == "__main__":
    list_stocks_names = ['MGLU3.SA']
    results = get_stocks_predict(list_stocks_names)

    # get date prediction
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    file_name = now.strftime("%d-%m-%Y %H:%M:%S") + '.json'

    # upload for bucket s3 aws
    s3 = boto3.resource('s3')
    s3_object = s3.Object('my-data-stocks', year + '/' + month + '/' + day + '/' + file_name)

