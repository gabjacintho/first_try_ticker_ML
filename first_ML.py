'''
simple Multilayer perceptron to predict ticker prices

also, thanks SentDex for all of the great tutorials! https://pythonprogramming.net/
'''
from alpha_vantage.timeseries import TimeSeries
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np
import time

ALPHA_VANT_API_KEY = '' #put your Alpha Vantage API key here, if you want to dowload for there
TICKERS = ['AAPL', 'ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL','AMG', 'A', 'GAS', 'ARE', 'APD', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ']
#these are just the first 45 companies in the SP500... use whatever you want here.
HIDDEN_LAYERS = [2, 4, 8]
HIDDEN_NODES = [16, 64, 128, 256]
LRS= [0.001, 0.0001]
BATCH_SIZES = [4, 8 , 32]
EPOCHS = 1000


def get_tickers_daily_data(tickers, file_name, api_key):
	if os.path.exists(f'{file_name}.pkl'):
		print('File already exists. Loading pickle...')
		main_df = pd.read_pickle(f'{file_name}.pkl')
		return main_df
	else:
		print('File does not exists. Downloading from Alpha Vantage...')
		#tickers = ['ABEV', 'AZUL', 'ESTC', 'GOAU', 'JBSS', 'RAIL', 'VALE']: ESTC começa em mai/2018
		ts = TimeSeries(key=api_key, output_format='pandas', indexing_type='date')
		counter = 1
		main_df = pd.DataFrame()
		for ticker in tickers:
			try:
				data, meta = ts.get_daily(symbol=ticker, outputsize='full')
				main_df[f'{ticker}'] = data['4. close']
				print(f'{counter} of {len(tickers)} companies downloaded...')
				time.sleep(15)
			except: 
				print(f'Could not download "{ticker}" data')
			
			counter+=1
			

		main_df.dropna(inplace=True)
		main_df.to_pickle(f'{file_name}.pkl')
		return main_df

def classify(current, future):
	if float(future) > float(current):
		return 1
	else:
		return 0

def normalize(df):
	norm = []
	max_val = np.nanmax(df)
	min_val = np.nanmin(df)
	for i in df:
		i = (i - min_val)/(max_val-min_val)	
		norm.append(i)
		
	return norm

def balance_data(data):
	pos = pd.DataFrame(columns=data.columns)
	neg = pd.DataFrame(columns=data.columns)
	for i, label in enumerate(data['label']):
		if label == 0:
			neg = neg.append(other=data.iloc[i,:], ignore_index=True)
		elif label == 1:
			pos = pos.append(other=data.iloc[i,:], ignore_index=True)

	lower = min(len(neg), len(pos))
	neg = neg[:lower]
	pos = pos[:lower]
	balanced_data = pos.append(other=neg,ignore_index=True).sample(frac=1).reset_index(drop=True)
	return balanced_data

def preprocess_data(data, ticker, test_pct):
	data['future'] = data[ticker].shift(-1)
	data['label'] = list(map(classify, data[ticker], data['future']))
	data.dropna(inplace=True)
	data.drop(columns=['future'], inplace=True)
	for col in data.columns:
		if col != 'label': 
			data[col] = data[col].pct_change()
			data[col] = normalize(data[col].values)
	data.dropna(inplace=True)
	test_portion = sorted(data.index.values)[-int(test_pct*len(sorted(data.index.values)))]
	test_data = data[(data.index >= test_portion)]
	train_data = data[(data.index < test_portion)]
	train_data = balance_data(train_data)
	test_data = balance_data(test_data)
	test_y = test_data.filter(items=['label'])
	test_X = test_data.drop(columns=['label'])
	train_y = train_data.filter(items=['label'])
	train_X = train_data.drop(columns=['label'])

	return train_X.values, train_y.values, test_X.values, test_y.values

def build_model(input_sample,n_hidden_layers,n_hidden_nodes, lr):
	model = Sequential()
	model.add(Dense(n_hidden_nodes, input_shape=(input_sample.shape[1:])))
	model.add(Dropout(0.1))
	for n in range(0,n_hidden_layers+1):
		model.add(Dense(n_hidden_nodes, activation='sigmoid'))
		model.add(Dropout(0.1))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(
    	loss='mean_squared_error',
    	optimizer=tf.keras.optimizers.Adam(lr=lr, decay=1e-6),
    	metrics=['accuracy'])
	return model


main_df = get_tickers_daily_data(TICKERS, 'ticker_data', ALPHA_VANT_API_KEY)

scores = []
for ticker in TICKERS[]:
	for nodes in HIDDEN_NODES:
		for lays in HIDDEN_LAYERS:
			for batch in BATCH_SIZES:
				for lr in LRS:
					df = main_df.copy()
					print(f'{ticker}....')
					train_X, train_y, test_X, test_y = preprocess_data(df, ticker, 0.1)
					NAME = f'{ticker}-PRED-{batch}-BATCH-{lays}-HIDLAYERS-{nodes}-HIDNODES-{lr}LR-{int(time.time())}'
					model = build_model(train_X,lays,nodes,lr)
					tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
					history = model.fit(
						train_X, train_y,
						batch_size=batch,
						epochs=EPOCHS,
						validation_data=(test_X, test_y),
						callbacks=[tensorboard],
						)
					acc = model.evaluate(test_X, test_y, verbose=0)[0]
					scores.append(acc)

print(max(scores))