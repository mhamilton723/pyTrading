from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from six.moves import cPickle
import pickle
import os
import random
import numpy as np
import pandas.io.data as web
import pandas as pd
from datetime import datetime
# from keras.layers.recurrent import GRU
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.callbacks import EarlyStopping
from utils import load_s_and_p_data, train_test_split, forecast, window_dataset


###################### Functions, Try to develop these and move into the utils folder ####################


#######################CONTROL FLOW PARAMS#################################

load_data = False
save_data = True
load_model = False
run_model = True
save_model = True
load_results = False
save_results = True
plot_results = True

if (not load_results and not run_model) and save_results:
    raise ValueError("Cannot save what has not been loaded or run ")

base_path = "~/machine_learning/stock_sandbox/"
model_prefix = 'simple_RNN'
data_fname = base_path + "s_and_p_500_data.pkl"
data_fname = os.path.expanduser(data_fname)
arch_fname = base_path + 'results/' + model_prefix + '_model_architecture.json'
arch_fname = os.path.expanduser(arch_fname)
weights_fname = base_path + 'results/' + model_prefix + '_model_weights.h5'
weights_fname = os.path.expanduser(weights_fname)
plot_fname = base_path + 'results/' + model_prefix + '_results.png'
plot_fname = os.path.expanduser(plot_fname)
results_fname = base_path + 'results/' + model_prefix + '_results.pkl'
results_fname = os.path.expanduser(results_fname)


#########################BEGIN CODE#######################################
# tickers = ['AAPL','VZ','NKE','KMI','M','MS','WMT','DOW','MPC']
tickers = None

if not load_results:

    if load_data:
        print('Loading data...')
        data = pickle.load(open(data_fname, 'r'))
        if tickers:
            data.loc(tickers)
    else:
        ##### Real Stock Data
        #print('Using Stock data')
        # data = load_s_and_p_data(start="2014-1-1",tickers=tickers)

        ##### Synthetic data for testing purposes
        #print('Using Synthetic data')
        #values = 10000
        #s = pd.Series(range(values))
        #noise = pd.Series(np.random.randn(values))
        #s = s / 1000 #+ noise / 100
        #d = {'one': s * s * 100/values,
        #     'two': np.sin(s * 10.),
        #     'three': np.cos(s * 10),
        #     'four': np.sin(s * s / 10) * np.sqrt(s) }
        #data = pd.DataFrame(d)

        ##### Easy synthetic data for testing purposes
        print('Using Easy synthetic data')
        flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
        pdata = pd.DataFrame({"a": flow, "b": flow})
        pdata.b = pdata.b.shift(9)
        data = pdata.iloc[10:] * random.random()  # some noise



    if save_data:
        print('Saving data...')
        pickle.dump(data, open(data_fname, 'wb+'))

    (X_train, y_train), (X_test, y_test) = train_test_split(data, n_prev=100)

    if not load_model:
        print('compiling model')
        in_out_neurons = len(data.columns)

        model = Sequential()
        hidden_neurons = 300
        #model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))
        model.add(SimpleRNN(in_out_neurons, hidden_neurons, return_sequences=False))
        model.add(Dense(hidden_neurons, in_out_neurons))
        model.add(Activation("linear"))

        '''
        model = Sequential()
        model.add(LSTM(in_out_neurons, 300, return_sequences=True))
        model.add(LSTM(300, 500, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(500, 200, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(200, in_out_neurons))
        model.add(Activation("linear"))
        '''
        model.compile(loss="mean_squared_error", optimizer="rmsprop")

        print('Training model...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
        model.fit(X_train, y_train, batch_size=30, nb_epoch=10, validation_split=0.1, callbacks=[early_stopping])
        #model.fit(X_train, y_train, batch_size=450, nb_epoch=10, validation_split=0.05)
    else:
        print('Loading model...')
        model = model_from_json(open(arch_fname).read())
        model.load_weights(weights_fname)

    if save_model:
        print("Saving model...")
        json_string = model.to_json()
        open(arch_fname, 'w+').write(json_string)
        model.save_weights(weights_fname, overwrite=True)

    if run_model:
        print('Running forecast...')
        window = 1
        predicted = forecast(model, X_test[0, :, :], n_points=len(X_test))
        wrong_predicted = model.predict(X_test)

        rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0)).mean()
        print("RMSE:", rmse)

    if save_results:
        print('Saving results...')
        pickle.dump((predicted, y_test), open(results_fname, 'wb+'))
else:
    print('Loading results...')
    predicted, y_test = pickle.load(open(results_fname, 'r'))

if plot_results:
    print('Plotting results...')
    fig = plt.figure()
    for i in range(min(4, predicted.shape[1])):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.plot(predicted[:100, i], color='r')
        ax.plot(wrong_predicted[:100, i], color='r', marker='+')
        ax.plot(y_test[:100, i], color='b')
        if tickers:
            ax.set_title(tickers[i])

    fig.savefig(plot_fname)
