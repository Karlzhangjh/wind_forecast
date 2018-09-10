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

# convert series to supervised learning
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

# load data
def load_data(filename,time_steps,features,split,time_range):
	#load_data
	raw_data = pd.read_csv(filename)
	raw_data = raw_data.iloc[:,[4,3,5,6]]
	print('the shape of raw_data: ',raw_data.shape)#(500,8)
	print(raw_data.head(12))

	values = np.array(raw_data).astype(float)

	#frame as supervised learning
	reframed_data = series_to_supervised(raw_data,1,time_range)#from 10 to predict next 1 
	#print(reframed_data)

	#drop columns we don't want to predict
	reframed_data = reframed_data.iloc[:,[0,1,2,3,time_range*features,time_range*features+1]]  
	print('the shape of reframed_data: ',reframed_data.shape)#(490,52)
	print(reframed_data.head(12))

	#normalize features  (second axis should be the same)
	scaler = MinMaxScaler(feature_range = (0,1)) #it will be better if split to several section
	scaled_data = scaler.fit_transform(reframed_data) #(,6)
	#print('the shape of scaled_data: ',scaled_data.shape)
	#print('scaled_data: ',scaled_data)

	#[Samples,timesteps,features]
	scaled_data = np.array(scaled_data)
	train_row = int(round(split * (scaled_data.shape[0])))
	train_data = scaled_data[:train_row,:]
	#np.random.shuffle(train_data)
	X_train, Y_train = train_data[:,:-2], train_data[:,-2:]
	X_test, Y_test = scaled_data[train_row:,:-2], scaled_data[train_row:,-2:]

	#change from 2D to 3D  :  (,1,5) to (,10,5)
	X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
	X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

	X_train_3D = np.zeros((X_train.shape[0]-time_range-1, time_steps, X_train.shape[2]), dtype=np.float)#9 numbers to front
	X_test_3D = np.zeros((X_test.shape[0]-time_range-1, time_steps, X_train.shape[2]), dtype=np.float)#9 numbers to front
	Y_train = Y_train[:X_train.shape[0]-time_range-1, :]
	Y_test = Y_test[:X_test.shape[0]-time_range-1, :]

	for i in range(X_train.shape[0]-time_range-1):
			for j in range(time_steps):
				X_train_3D[i,j,:] = X_train[i+j,0,:]

	for i in range(X_test.shape[0]-time_range-1):
			for j in range(time_steps):
				X_test_3D[i,j,:] = X_test[i+j,0,:]

	#print('the shape of training data',X_train_3D.shape,Y_train.shape,X_test_3D.shape,Y_test.shape)
	return X_train_3D,Y_train,X_test_3D,Y_test,scaler 
	
#model
def build_train_model(X_train,Y_train,X_test,Y_test,epochs,batch_size):
	model = Sequential()

	time_steps = X_train.shape[1]
	n_features = X_train.shape[2]
	neurons = 50
	ahead = Y_test.shape[1]
	model.add(LSTM(units=neurons,input_shape=(time_steps,n_features),return_sequences=True))
	model.add(Dropout(0.2))

	neurons = 100
	model.add(LSTM(units=neurons))
	model.add(Dropout(0.2))

	model.add(Dense(units=ahead,input_dim=neurons))

	model.add(Activation('linear'))

	model.compile(loss='mae',optimizer='adam')

	#fit network
	history = model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_train,Y_train),verbose=2,shuffle=False)#
	
	#plot history
	'''
	plt.subplot(5,1,5)
	plt.title("loss and val_loss")
	plt.plot(history.history['loss'],label='train_loss')
	plt.plot(history.history['val_loss'],label='test_val_loss')
	plt.legend()	#plt.show()
	'''	
	return model

#prediction
def pos_predict(model,scaler,X_test,Y_test):
	Y_hat = model.predict(X_test)
	
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

def plot_result(predict_result,true_result,figs):
	fig_m = 4
	fig_n = 2
	Rmse_X,Rmse_Y = sqrt(mean_squared_error(predict_result[:,0],true_result[:,0])), sqrt(mean_squared_error(predict_result[:,1],true_result[:,1]))
	Max_X,Max_Y   = max(abs(predict_result - true_result)[:,0]), max(abs(predict_result - true_result)[:,1])

	if figs == 0 :
		# X
		plt.subplot(fig_m,fig_n,1)
		plt.title('2018-8-3-11:13 Train   '+'RMSE_X:%.2f '%Rmse_X+'RMSE_Y:%.2f     '%Rmse_Y+'Max_X:%.2f '%Max_X+'Max_Y:%.2f '%Max_Y)
		plt.plot(true_result[:,0],'k-',label='True X')
		plt.plot(predict_result[:,0],'r--',label='Pred X')
		plt.legend(loc="upper left")
		# Y
		plt.subplot(fig_m,fig_n,3)
		plt.plot(true_result[:,1],'k-',label='True Y')
		plt.plot(predict_result[:,1],'r--',label='Pred Y')
		plt.plot(np.zeros(true_result.shape[0]),'k-')
		plt.legend(loc="upper left")
	elif figs == 1 :
		# X
		plt.subplot(fig_m,fig_n,2)
		plt.title('2018-8-3-11:18   '+'RMSE_X:%.2f '%Rmse_X+'RMSE_Y:%.2f     '%Rmse_Y+'Max_X:%.2f '%Max_X+'Max_Y:%.2f '%Max_Y)
		plt.plot(true_result[:,0],'k-',label='True X')
		plt.plot(predict_result[:,0],'r--',label='Pred X')
		plt.legend(loc="upper left")
		# Y
		plt.subplot(fig_m,fig_n,4)
		plt.plot(true_result[:,1],'k-',label='True Y')
		plt.plot(predict_result[:,1],'r--',label='Pred Y')
		plt.plot(np.zeros(true_result.shape[0]),'k-')
		plt.legend(loc="upper left")
	elif figs == 2 :
		# X
		plt.subplot(fig_m,fig_n,5)
		plt.title('2018-8-3-11:22   '+'RMSE_X:%.2f '%Rmse_X+'RMSE_Y:%.2f     '%Rmse_Y+'Max_X:%.2f '%Max_X+'Max_Y:%.2f '%Max_Y)
		plt.plot(true_result[:,0],'k-',label='True X')
		plt.plot(predict_result[:,0],'r--',label='Pred X')
		plt.legend(loc="upper left")
		# Y
		plt.subplot(fig_m,fig_n,7)
		plt.plot(true_result[:,1],'k-',label='True Y')
		plt.plot(predict_result[:,1],'r--',label='Pred Y')
		plt.plot(np.zeros(true_result.shape[0]),'k-')
		plt.legend(loc="upper left")
	elif figs == 3 :
		# X
		plt.subplot(fig_m,fig_n,6)
		plt.title('2018-8-3-11:27   '+'RMSE_X:%.2f '%Rmse_X+'RMSE_Y:%.2f     '%Rmse_Y+'Max_X:%.2f '%Max_X+'Max_Y:%.2f '%Max_Y)
		plt.plot(true_result[:,0],'k-',label='True X')
		plt.plot(predict_result[:,0],'r--',label='Pred X')
		plt.legend(loc="upper left")
		# Y
		plt.subplot(fig_m,fig_n,8)
		plt.plot(true_result[:,1],'k-',label='True Y')
		plt.plot(predict_result[:,1],'r--',label='Pred Y')
		plt.plot(np.zeros(true_result.shape[0]),'k-')
		plt.legend(loc="upper left")


if __name__ == '__main__':
	
	print('>>>>> Loading data...')
	#filename1,filename2,time_steps,split
	time_steps = 20
	forecast_steps = [10]#[0,5,10,15,20]

	for fore_step in forecast_steps:

		fig = plt.figure(fore_step)

		time_range = time_steps + fore_step
		features = 4
		split = 0.9
		X_train,Y_train,X_test,Y_test,scaler = load_data('X_Y_dX_dY_11_13.csv', time_steps, features, split , time_range)
		print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
		print('>>>>> Data Loaded. Compiling...')

		#build  and  train   model
		epochs = 50
		batch_size = 50#int(round(X_train.shape[0]/200))
		model = build_train_model(X_train,Y_train,X_test,Y_test, epochs, batch_size)

		#save model  HDF5 file
		#model.save('my_model.h5')
		
		my_model = model#load_model('my_model.h5') # model_four_two_0.1s.h5   from 0.5s to predict 0.1s

		#predict next position 
		predict_result,true_result = pos_predict(my_model,scaler,X_test,Y_test)
		plot_result(predict_result,true_result,0)

		#other dataset  1
		X_train,Y_train,X_test,Y_test,scaler = load_data('X_Y_dX_dY_11_18.csv', time_steps, features, 0.1, time_range)
		print('The shape of other dataset 1',X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
		predict_result,true_result = pos_predict(my_model,scaler,X_test,Y_test)
		plot_result(predict_result,true_result,1)

		#other dataset  2
		X_train,Y_train,X_test,Y_test,scaler = load_data('X_Y_dX_dY_11_22.csv', time_steps, features, 0.1, time_range)
		print('The shape of other dataset 2',X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
		predict_result,true_result = pos_predict(my_model,scaler,X_test,Y_test)
		plot_result(predict_result,true_result,2)

		#other dataset  3
		X_train,Y_train,X_test,Y_test,scaler = load_data('X_Y_dX_dY_11_27.csv', time_steps, features, 0.1, time_range)
		print('The shape of other dataset 3',X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
		predict_result,true_result = pos_predict(my_model,scaler,X_test,Y_test)	
		plot_result(predict_result,true_result,3)

	#draw the result
	plt.show()
	#'''
	