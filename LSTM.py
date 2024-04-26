#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:54:44 2024

@author: shahab-nasiri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose # For time series decomposition
from pmdarima import auto_arima


# For LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model  # Allows load a previously saved model.

# To evaluate the models
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# To enable interactive plots

## For Jupyter web (requires ipympl module)
#%matplotlib widget

## For IDEs, like PyCharm
import matplotlib

full_data = pd.read_csv('brent_daily_prices.csv', parse_dates=['DATE'], index_col='DATE', na_values='.')  # In the original time series, NA values are represented by a period (.)
data = full_data['DCOILBRENTEU']
data.rename_axis('date', inplace=True)
data.rename('brent_crude_oil', inplace=True)
data.fillna(method='ffill', inplace=True)  # Replace NaN values with the last valid observation.
data.index = pd.DatetimeIndex(data.index).to_period('B').to_timestamp()  # Sets the frequency for the time series.

# Basic EDA of the data
print(data.describe())
print('\n')
print(data.info())
print('Missing values: ', data.isna().sum())

# Plots the data
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.title('Brent Crude Oil Price since 1987')
plt.xlabel('Date')
plt.ylabel('Price per barrel (USD)')
plt.show()


seasonal_decompose(data).plot()
plt.xticks(rotation=45)
plt.show()

train_data = data.iloc[:len(data) - 30]
test_data = data.iloc[len(data) - 30:]
# Reshapes the data to feed the model
full_data_lstm = data.values.reshape(-1, 1)
train_data_lstm = train_data.values.reshape(-1, 1)
test_data_lstm = test_data.values.reshape(-1, 1)

# Defines train and test sets
X_train = []
y_train = []
ws = 30 # Window size: indicates the number of previous time steps. The more, may lead to higher accuracy, but increases complexity and training time.

for i in range(ws, len(train_data_lstm)):
    X_train.append(train_data_lstm[i - ws: i])
    y_train.append(train_data_lstm[i])

X_train, y_train = np.array(X_train), np.array(y_train)

model = Sequential()
model.add(LSTM(150, activation='relu', input_shape = (X_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')  
model.fit(X_train, y_train, epochs=100, batch_size=600)

plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()