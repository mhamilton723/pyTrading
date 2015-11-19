PyTrading

Python classes for automated stock trading.

-Classes for portfolios and strategies
-Buy/hold and momentum strategy classes
-Backtesting functions defined for any strategy object
-Helper functions for sci-kit learn regressors to turn them into time series predictors
-example omega cluster scripts



Install:


The current dependencies are mostly found in anaconda with the exception of theano and keras.
 These are packages for running Neural Nets on GPUs and are not completely necessary for most of the files. Theano comes with linux/mac anaconda but is a royal pain on windows.  If you have trouble installing any of these feel free to shoot me an email.


Under development:

-A sci-kit learn time series estimator class

-GPU enabled Recurrent Neural Nets, LSTMs for stock prediction


Needed:

-Unit tests

-Documentation

-More Stock Strategies

-command line interface for running multiple jobs with differing parameters

-tick data 