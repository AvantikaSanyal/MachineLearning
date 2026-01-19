\# Retail1 - Sales Prediction/Forecasting



\## This is a simple forecasting model using the following python libraries :

1\. pandas - data manipulation and analysis

2\. matplotlib.pyplot - basic plotting and visualisation

3\. seaborn - statistical data visualisation (built on top of matplotlib)

4\. xgboost - extreme gradient boosting (ML algorithm)

5\. train\_test\_split (from sklearn.model\_selection) - splitting the available data into test set and training set

6\. mean\_squared\_error (from sklearn.metrics) - evaluating the accuracy for the regression model

7\. numpy - numerical computing library



This project builds a time-series forecasting model to predict daily sales.  

It uses lagged sales features and rolling window statistics to capture trends and seasonality in the data.



\## Key techniques used:  

\- Feature engineering: lag features and rolling statistics for time series trends.  

\- Modeling: XGBoost regressor to predict sales.  

\- Evaluation: RMSE metric on test data.



The model achieved an \*\*RMSE of 668.49\*\*, showing improved accuracy compared to naive forecasts.

