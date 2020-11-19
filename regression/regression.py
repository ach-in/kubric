import requests
import pandas
import scipy
import numpy
import sys
from math import sqrt
import io

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def mean(values):
	return sum(values) / float(len(values))

def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar


def variance(values, mean):
	return sum([(x-mean)**2 for x in values])


def coeff(data):
	x = [float(row) for row in data.keys()]
	y = [row for row in data]
	# print (x, y)
	# print(data)
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]



def predict_price(area) -> float:
		"""
		This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

		You can run this program from the command line using `python3 regression.py`.
		"""
		response = requests.get(TRAIN_DATA_URL).content
		# YOUR IMPLEMENTATION HERE
		raw_data = pandas.read_csv(io.StringIO(response.decode('utf-8')))

		raw_data =  raw_data.transpose()[0]

		raw_data = raw_data.iloc[1:]
		# raw_data.columns = ["area", "price"]

		predictions = list()

		# print (raw_data.keys())

		b0, b1 = coeff(raw_data)
		for row in area:
			yhat = b0 + b1 * row
			predictions.append(yhat)

		return predictions


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
