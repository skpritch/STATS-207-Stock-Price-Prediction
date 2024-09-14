# A Time Series Analysis of Patent Filings and Stock Prices: A Proxy for Modeling the Relationship between Innovation and Profitability

**Authors**:  
Teddy Ganea (Stanford University, Department of Mathematics)  
Ky Friedman (Stanford University, Doerr School of Sustainability)  
Simon Pritchard (Stanford University, Department of Biology)  

## Abstract
This study presents a novel use for patent filing data in predicting stock price as a proxy for innovation within a leading Japanese semiconductor company. After extracting feature variables from patent filings, we fit linear models and SARIMAX models to the time series of stock price, following both a simple log transform and a more complex decorrelation from a semiconductor stock market.

In fitting the SARIMAX models, we employed an extensive hyperparameter grid search, testing over 14,000 configurations of order and seasonal order. We then compared each SARIMAX model to a SARIMA fit (without patent data) of the same order and seasonal order. In every model fitting, appropriate cross-validation techniques were used to compute test MSE.

Our results demonstrated that decorrelation from the stock market vastly improved the performance of the models and that SARIMAX models fit on decorrelated data perform the best among the models we evaluated. Additionally, we found that, for identical order and seasonal order configurations, SARIMAX with patent features as exogenous variables yielded lower test MSE in about 96% of cases compared to their pure SARIMA counterpart.

While none of the models achieved predictive capacities great enough for market deployment, this study suggests a powerful opportunity to utilize company patent filings as a regressive variable in stock prediction models.

## Keywords
Time series, linear regression, ARIMA, SARIMA, SARIMAX, stock predictions, patents, innovation
