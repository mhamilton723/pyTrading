from __future__ import print_function
from keras.models import Sequential  
from keras.layers.core import TimeDistributedDense, Activation, Dropout  

from six.moves import cPickle
import pickle
import os
#from keras.layers.recurrent import GRU
import numpy as np
import pandas.io.data as web
from datetime import datetime
from keras.models import Sequential  
from keras.models import model_from_json
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

save_model = True
load_model = False
arch_fname = 'results/my_model_architecture.json'
weigths_fname = 'results/my_model_weights.h5'

def _load_data(data, n_prev = 30):  
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY
	
def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

def get_data(tickers, start="2009-1-1", end="2015-11-02"):
    start_time = datetime.strptime(start, "%Y-%m-%d")
    end_time = datetime.strptime(end, "%Y-%m-%d")
    df = web.DataReader(tickers, 'yahoo', start_time, end_time)
    df = df['Adj Close']
    #df = df.diff()
    #df = df.iloc[1:len(df),:]
    return df
	
np.random.seed(0)  # For reproducability
tickers=['AAPL', 'QQQ','VZ','NKE','KMI']
data = get_data(tickers)
(X_train, y_train), (X_test, y_test) = train_test_split(data)
print("Data loaded.")

if not load_model:
	in_out_neurons = len(tickers)  
	hidden_neurons = 300
	model = Sequential()  
	model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))  
	model.add(Dense(hidden_neurons, in_out_neurons))  
	model.add(Activation("linear"))  
	model.compile(loss="mean_squared_error", optimizer="rmsprop")  
	print('model compiled')

	# and now train the model. 
	model.fit(X_train, y_train, batch_size=30, nb_epoch=200, validation_split=0.1)	
else:
	print('Load model...')
	model = model_from_json(open(arch_fname).read())
	model.load_weights(weights_fname)	

if save_model:
	print("Saving model...")
	#f not os.path.exists(model_file):
	#	os.makedirs(sav)
	#pickle.dump(model, open(model_fname, "wb"))	
	
	json_string = model.to_json()
	open(arch_fname, 'w').write(json_string)
	model.save_weights(weigths_fname,overwrite=True)




predicted = model.predict(X_test)  


fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(predicted[:,0],color='r')
ax.plot(X_test[:,0,0],color='b')
ax = fig.add_subplot(222)
ax.plot(predicted[:,1],color='r')
ax.plot(X_test[:,0,1],color='b')
ax = fig.add_subplot(223)
ax.plot(predicted[:,2],color='r')
ax.plot(X_test[:,0,2],color='b')
ax = fig.add_subplot(224)
ax.plot(predicted[:,3],color='r')
ax.plot(X_test[:,0,3],color='b')

fig.savefig('results/stock_rnn.png')

print(np.sqrt(((predicted - y_test) ** 2).mean(axis=0)).mean())  # Printing RMSE 