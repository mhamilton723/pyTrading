#PyTrading
Python classes for automated stock trading.

-Classes for portfolios and strategies
-Buy/hold, momentum, and linear/nonlinear regression strategy classes
-Backtesting functions defined for any strategy object
-example omega cluster scripts
-GPU enabled Recurrent Neural Nets, LSTMs for stock prediction


##Install:
The current dependencies are mostly found in anaconda with the exception of theano and keras.
 These are packages for running Neural Nets on GPUs and are not completely necessary for most of the files. 
 Theano comes with linux/mac anaconda but is a royal pain on windows.  If you have trouble installing any of these feel free to shoot me an email.

## Usage

PyTrading is a python package designed to facilitate the creation and testing of automated trading strategies. 
The main object of the package is the strategy object.

'''
from Strategies import BuyAndHoldStrategy, backtest

tickers = ['AAPL','VZ']
bs = BuyAndHoldStrategy(10000, tickers)
percent_gains = backtest(bs,start="2014-1-1", end="2015-11-02",correct=False)
print('Buy and hold strategy made a %{} return on investment'.format(round(percent_gains,1)) ) 
'''
'''
Buy and hold strategy made a %31.1 return on investment
'''

See the ipython notebook demo for more examples and usage! 

##Needed:
-Unit tests
-Documentation
-More Stock Strategies
-tick data 


# TimeSeriesRegressor
A wrapper estimator that transforms any sklearn regressor into a time series predictor or sequence to sequence mapper.
The TSR internally transforms a regular dataset where the rows correspond to terms of a sequence into a sequence prediction dataset and
learns a sequence to sequence predictor.

##Standalone
See https://github.com/mhamilton723/TimeSeriesRegressor for standalone files

## Requires
Numpy, Pandas, SciKit-Learn,pickle

## Usage

To make a predictor of the stock market that maps the previous two days of the s&p500 stock prices and 
predicts the next day's price of AAPL stock try the following:
```
from TimeSeriesEstimator import TimeSeriesRegressor, time_series_split
from sklearn.linear_model import LinearRegression,Lasso
from utils import datasets


X = datasets('sp500')
y = X['AAPL']
X_train, X_test = time_series_split(X)
y_train, y_test = time_series_split(y)


n_prev=2
tsr = TimeSeriesRegressor(Lasso(), n_prev=n_prev)
tsr.fit(X_train, y_train)
pred_train = tsr.predict(X_train) #outputs a numpy array of length: len(X_train)-n_prev
pred_test = tsr.predict(X_test)
```

To forecast all stocks in the s&p500 100 days into the future use the forecast method:

```
tsr = TimeSeriesRegressor(LinearRegression(), n_prev=2)
tsr.fit(X_train)
fc = tsr.forecast(X_train, 100)
```
See the ipython notebook for a longer interactive example!

## Install
Clone this repo and call directly as a module. Have not added automatic install support yet.

##Mechanics

The TSR works by taking in a single (X) or two datasets (X,Y) of equal length. 
In the single dataset case, the TSR assumes you would like to predict the next element in the dataset using the previous elements.
In either case it forms a dataset by taking the previous n timesteps and flattening them into a vector. 

<table>
 <caption>Dataset X</caption>
<tr>
<th>Feature 1</th>
<th>Feature 2</th>
</tr>
<tr>
<td> 1</td>
<td> 1.5</td>
</tr>
<tr>
<td>2</td>
<td>2.5</td>
</tr>
<tr>
<td>3</td>
<td>3.5</td>
</tr>
<tr>
<td>4</td>
<td>4.5</td>
</tr>
<tr>
<td>5</td>
<td>5.5</td>
</tr>
</table>


<table>
<table style="float: left;">
 <caption>New X with n_prev = 2</caption>
<tr>
<th>Feature 1</th>
<th>Feature 2</th>
<th>Feature 3</th>
<th>Feature 4</th>
</tr>
<tr>
<td> 1</td>
<td> 1.5</td>
<td>2</td>
<td>2.5</td>
</tr>
<tr>
<td>2</td>
<td>2.5</td>
<td>3</td>
<td>3.5</td>
</tr>
<tr>
<td>3</td>
<td>3.5</td>
<td>4</td>
<td>4.5</td>
</tr>
</table>



<table>
<table style="float: middle-left;">
 <caption>New Y with n_prev = 2</caption>
<tr>
<th>Feature 1</th>
<th>Feature 2</th>
</tr>
<tr>
<td>3</td>
<td>3.5</td>
</tr>
<tr>
<td>4</td>
<td>4.5</td>
</tr>
<tr>
<td>5</td>
<td>5.5</td>
</tr>
</table>



Now the X and Y datasets can be fit by any regression technique in sklearn.
If the technique cannot handle vectors as outputs, use the "parallel_models" input to predict
each feature sequence with its own multi to single dim regressor.

