# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:34:03 2020

@author: kv83821
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = train_dataset.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0 , 1))
scaled_training_set = scaler.fit_transform(train_set)

#We now use 80(4 months) timesteps backwards to predict the new stock price i.e. create data structure for this
X_train = []
Y_train = []
for i in range(80,1258):
    X_train.append(scaled_training_set[i-80:i , 0]) #80 prev. examples
    Y_train.append(scaled_training_set[i , 0]) #new prediction based on X_train
X_train ,Y_train = np.array(X_train),np.array(Y_train)

X_train = np.resize(X_train,(X_train.shape[0],X_train.shape[1],1))

# now building

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# first layer and dropout to reduce overfitting(20% neurons off)
regressor = Sequential()

regressor.add(LSTM(units = 50 , return_sequences = True , input_shape = (X_train.shape[1] , 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50 , return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50 , return_sequences = True))
regressor.add(Dropout(0.2))

#Now since this is the final layer we do not return the sequence 

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Now we got the four layer Rec. NN

#Now we have the output layer as dense NN layer where 1 corresponds to the output
regressor.add(Dense(units = 1))

#Now we add the optimizer and loss function

regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error')

#Now we fit the X_train and Y_train

regressor.fit(X_train, Y_train, epochs = 100 , batch_size = 32)

# Get the test dataset

test_dataset = pd.read_csv('Google_Stock_Price_Test.csv')
real_set = test_dataset.iloc[:,1:2].values

#Now we want the concatenated train and test set since we require 80 previous 
#records 

total_dataset = pd.concat((train_dataset['Open'], test_dataset['Open']), axis = 0) 
inputs = total_dataset[len(total_dataset) - len(test_dataset) - 80:].values
inputs = inputs.reshape(-1 ,1)

#Since the scaler object has already been fit we need fit it again only apply the transform method
inputs = scaler.transform(inputs)

#We now create the X_test
X_test = []
for i in range(80,100):
    X_test.append(inputs[i-80:i , 0]) #80 prev. examples
X_test = np.array(X_test)

X_test = np.resize(X_test,(X_test.shape[0],X_test.shape[1],1))

#predicting the results

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#plotting the results
plt.plot(real_set , color = 'red' , label = 'Real stock price for google')
plt.plot(predicted_stock_price , color= 'blue' , label = 'Predicted stock price for google') 
plt.title('Google stock price predictions')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()













