# -*- coding: utf-8 -*-
from __future__ import print_function
    
import tensorflow as tf
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential,load_model

'''-------- convert series to supervised learning --------'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


'''------------- load total wind data ------------------'''
def load_total_wind_data(files_name, time_steps, features):
	scaled_data = range(len(files_name))
	#supervised_data = range(len(files_name))

	years = -1
	for file in files_name:
		years += 1
		raw_data = pd.read_csv(file)
		#print(raw_data.head(100))
		values = np.array(raw_data)
		#print('the shape of values',values.shape)

		temp_data = np.zeros((24,values.shape[0]/25))
		for i in range(values.shape[0]/25):
			#print(i*25+1)
			temp_data[:,i] = values[(i*25+1):(i*25+25),0] 

		reframed_data = np.transpose(temp_data)
		# print('the shape of reframed_data: ',reframed_data.shape,'\nreframed_data: \n',reframed_data)

		supervised_data = series_to_supervised(reframed_data,1,1)	
		#supervised_data = supervised_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,
		#										  12,13,14,15,16,17,18,19,20,21,22,23,24]]
		#print('the shape of supervised_data: ',supervised_data.shape,'\nsupervised_data: \n',supervised_data)

		scaler = MinMaxScaler(feature_range = (0,1)) 
		scaled_data[years] = scaler.fit_transform(supervised_data) 
		#print('the shape of scaled_data',scaled_data[years].shape)	

	# get train data
	X_train_total = np.empty((1, time_steps, features)) # 7 years
	Y_train_total = np.empty((1, 24))
	X_test_total = np.empty((1, time_steps, features))  # 2 years
	Y_test_total = np.empty((1, 24))

	for f_index in range(len(files_name)):  
		X_2D, Y_2D = scaled_data[f_index][:,:-24], scaled_data[f_index][:,-24:] 
		X_3D = X_2D.reshape(X_2D.shape[0], time_steps, features)

		print(X_3D.shape,'  ' ,Y_2D.shape)

		# trian and test
		if f_index < 8:
			X_train_total = concatenate((X_train_total,X_3D),axis=0)
			Y_train_total = concatenate((Y_train_total,Y_2D),axis=0)
		else:
			X_test_total = concatenate((X_test_total,X_3D),axis=0)
			Y_test_total = concatenate((Y_test_total,Y_2D),axis=0)
	
	X_train_total = X_train_total[1:,:,:] # because of empty
	Y_train_total = Y_train_total[1:,:] # because of empty
	X_test_total = X_test_total[1:,:,:] # because of empty
	Y_test_total = Y_test_total[1:,:] # because of empty
	#print('the shape of training data',X_train_total.shape,Y_train_total.shape,X_test_total.shape,Y_test_total.shape)

	print(X_train_total)
	print(Y_train_total)
	return X_train_total,Y_train_total,X_test_total,Y_test_total	
	'''
	# wind_total_line
	wind_total_line = np.zeros((1,(wind_total_day.shape[0]*wind_total_day.shape[1])))
	for i in range(wind_total_day.shape[0]):
		wind_total_line[0,24*i:(24*i+24)] = wind_total_day[i,:]

	print('the shape of wind_total_line',wind_total_line.shape)
	'''


'''--------------- load wind data ----------------'''
def load_wind_data(filename, time_steps, features):
	raw_data = pd.read_csv(filename)
	#print(raw_data.head(100))
	values = np.array(raw_data)
	#print('the shape of values',values.shape)

	temp_data = np.zeros((24,values.shape[0]/25))
	for i in range(values.shape[0]/25):
		#print(i*25+1)
		temp_data[:,i] = values[(i*25+1):(i*25+25),0] 

	reframed_data = np.transpose(temp_data)
	# print('the shape of reframed_data: ',reframed_data[0].shape,'\nreframed_data: \n',reframed_data[0])

	supervised_data = series_to_supervised(reframed_data,1,1)	
	# print('the shape of supervised_data: ',supervised_data.shape,'\nsupervised_data: \n',supervised_data)

	scaler = MinMaxScaler(feature_range = (0,1)) 
	scaled_data = scaler.fit_transform(supervised_data) 
	print('the shape of scaled_data',scaled_data.shape)	

	# get test data
	X_2D, Y_2D = scaled_data[:,:-24], scaled_data[:,-24:] 
	X_3D = X_2D.reshape(X_2D.shape[0], time_steps, features)
	print('the shape of test dataset',X_3D.shape,Y_2D.shape)

	return X_3D,Y_2D,scaler



	
'''---------- model -----------'''
def build_train_model(X_train, Y_train, X_test, Y_test, epochs, batch_size):
	model = Sequential()

	time_steps = X_train.shape[1]
	n_features = X_train.shape[2]
	neurons = 100
	ahead = Y_test.shape[1]
	model.add(LSTM(units=neurons,input_shape=(time_steps,n_features),return_sequences=True))
	model.add(Dropout(0.2))

	neurons = 200
	model.add(LSTM(units=neurons))
	model.add(Dropout(0.2))

	model.add(Dense(units=ahead,input_dim=neurons))

	model.add(Activation('linear'))

	model.compile(loss='mae',optimizer='adam')

	#fit network
	history = model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_train,Y_train),verbose=2,shuffle=False)#
	
	#plot history
	plt.title("loss and val_loss")
	plt.plot(history.history['loss'],label='train_loss')
	plt.plot(history.history['val_loss'],label='test_val_loss')
	plt.legend()	
	
	return model


'''-------- prediction -------------'''
def pos_predict(model, scaler, X_test, Y_test):
	day = 1
	Y_hat = model.predict(X_test)
	
	print('Y_hat: ',Y_hat[day,:])
	print('Y_test: ',Y_test[day,:])

	plt.plot(Y_hat[day,:]);
	plt.plot(Y_test[day,:]);
	
	'''
	#invert scaling for forecast
	inv_Y_hat = concatenate((X_test[:,0,:],Y_hat),axis=1) # axis=1 (addition on second dim) 
	#print('inv_Y_hat shape',inv_Y_hat.shape)
	inv_Y_hat = scaler.inverse_transform(inv_Y_hat)
	inv_Y_hat = inv_Y_hat[:,-2:]
	#invert scaling for actual
	inv_Y = concatenate((X_test[:,0,:],Y_test),axis=1)
	inv_Y = scaler.inverse_transform(inv_Y)	
	#print('inv_Y shape',inv_Y.shape)
	inv_Y = inv_Y[:,-2:]
	#print('predited Y',inv_Y_hat,'true Y',inv_Y)

	#calculate RMSE
	rmse_x = sqrt(mean_squared_error(inv_Y_hat[:,0],inv_Y[:,0]))
	rmse_y = sqrt(mean_squared_error(inv_Y_hat[:,1],inv_Y[:,1]))
	print('Test RMSE_X:%.3f'%rmse_x,'Test RMSE_Y:%.3f'%rmse_y)
	
	#predict data and true data
	return inv_Y_hat,inv_Y
	'''


if __name__ == '__main__':
	fig = plt.figure(1)

	'''load training data'''
	print('>>>>> Loading data...')
	time_steps = 24 # 24 hours
	features = 1 # wind output

	files = ['CaliforniaISO_2010.csv','CaliforniaISO_2011.csv','CaliforniaISO_2012.csv',
			 'CaliforniaISO_2013.csv','CaliforniaISO_2014.csv','CaliforniaISO_2015.csv',
			 'CaliforniaISO_2016.csv','CaliforniaISO_2017.csv','CaliforniaISO_2018.csv']

	#load_total_wind_data(files, time_steps, features)
	X_train,Y_train,X_test,Y_test = load_total_wind_data(files, time_steps, features)
	
	# print('the shape of training data',X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
	print('>>>>> Data loaded. Compiling...\n')

	
	### build and train model
	epochs = 100
	batch_size = 12 # int(round(X_train.shape[0]/200))
	#model = build_train_model(X_train,Y_train,X_test,Y_test, epochs, batch_size)
	#model.save('./result_day_hour/model_day_hour.h5')
	my_model = load_model('./result_day_hour/model_day_hour.h5')

	# test dataset 
	X_test,Y_test,scaler = load_wind_data('CaliforniaISO_2018.csv', time_steps, features)
	print('The shape of test dataset',X_test.shape,Y_test.shape)
	#predict next position 
	pos_predict(my_model,scaler,X_test,Y_test) # scaler (,744)
	#predict_result,true_result = pos_predict(my_model,scaler,X_test,Y_test)
	# plot_result(predict_result,true_result,0)

	plt.show()
	