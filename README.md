# Forex trading script based on a XGBoost model

This is a XGBoost model i made to test how well it could handle the noisy forex dataset from my reinforcement learning project. It actually did a very decent job. It showed a drawdown in the long run, but not that bad.

To tune hyperparameters I used GridsearchCV from Sklearn, but the cross validation mad it pretty slow to do just a quick test of parameters. Instead I made a function which did it without cv.

I tested it as classification model. 4 actions:
 
 0. Do nothing
 1. Buy
 2. Sell
 3. close order
 
The actions was handled by my program and not the model.

But I had the most success with the regression version of the model, in which it had to predict how far the forexpair would rise or fall. If it predicted 30, price would go up 30 points (pips actually) and if it predicted -30 it would go down 30 points. Thus if it was 30 my program shuld execute a buy order and hold it until price reached a +30 points.

In the Jupyter notebook file I have the model creation, search and tuning.
In the "live" folder you habe the final program I used to test the model live in a FXCM demo account.

DISCLAIMER: DO NOT USE THIS FOR LIVE TRADING ON REAL MONEY.
