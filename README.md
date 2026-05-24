# Uniform Singapore Energy Price 2025

<img width="637" height="288" alt="Screenshot 2026-05-23 223532" src="https://github.com/user-attachments/assets/89b7912b-be54-4583-b4e5-e66882d580eb"/>

## 1.0 Introduction
Time-Series forecasting is widely used in financial and energy markets to capture temporal dependencies and price fluctuations. In this project, the USEP dataset was analyzed using a hybrid modelling approach. First, ARIMA (1, 1, 1) was applied to capture linear patterns and trends in the time series. Next, GARCH (1, 1) was used to model volatility clustering present in the ARIMA residuals. Finally, engineered features derived from these models were used to train a Random Forest Regressor and BiLSTM. 

The workflow includes:
1. Data preprocessing and exploratory analysis.
2. ARIMA modeling
3. GARCH modeling
4. Feature engineering using lagged values and residual statistics.
5. Random Forest and BiLSTM training and hyperparameter optimization
6. Model Evaluation using MAE, RMSE, and R2 metrics. 


## 2.0 Problem Statement 
The objective of this study is to develop a predictive framework for forecasting the Uniform Singapore Energy Price (USEP) using statistical and machine learning approaches. Electricity prices are highly volatile and influenced by both historical trends and market uncertainty. TO address this challenge, it combines ARIMA for time-series trend modeling, GARCH for volatility estimation, and Random Forest Regression/ BiLSTM for nonlinear predictive learning. The study aims to evaluate whether integrating econometric models with machine learning and deep learning techniques can improve forecasting accuracy. 


## 3.0 ARIMA (1, 1, 1)
An ARIMA (1, 1, 1) model was selected to forecast the USEP time series after examining the stationarity and autocorrelation structure of the data. The ARIMA model contains three parameters:
- p = 1: autoregressive (AR) order
- d = 1: first differencing
- q = 1: moving-average (MA) order

### 3.1 Determination of Parameters
The original USEP series exhibited non-stationary behaviour, where the mean and variance changed over time. Since ARIMA requires a stationary series, first-order differencing was applied. 

This transformation removes long-term trends by subtracting consecutive observations. After one differencing step, the series appeared more stable around a constant mean, d = 1 was selected.  

The differenced series was then analysed using the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.
- The PACF plot showed a strong spike at lag 1 followed by a sharp decline, indicating that only one significant autoregressive term was needed. Therefore, p = 1.
- The ACF plot also displayed a significant spike at lag 1 with gradual decay afterward, suggesting a first-order moving-average process. Therefore, q = 1. 

### 3.2 ARIMA Model Representation
The fitted ARIMA model can be expressed as: 


### 3.3 Interpretation of ARIMA Summary Outputs
The ARIMA summary output provides statistical information about the fitted model:
- AR coefficient (ar.L1) measures the relationship between the current differenced value and its previous value.
- MA coefficient (ma.L1) measures the relationship between the current value and the previous forecast error.
- Standard Error indicates the uncertainty associated with each estimated coefficient.
- z-statistic and p-value determine whether the coefficients are statistically significant. (i.e. Small p-values (typically less than 0.05) indicate that the coefficient contributes meaningfully to the model.)

### 3.4 Residual Diagnostics 
After fitting the ARIMA model, residual diagnostics were analyzed to determine whether the remaining errors behaved like white noise. 
- Residual Plot: Residuals fluctuated randomly around zero without obvious patterns, suggesting that most linear dependencies were removed.
- Histogram/ Density Plot: The residual distribution appeared approximately centered around zero, although some heavy tails remained due to volatility in electricity prices.
- ACF of residuals: The residuals autocorrelations were relatively small and mostly within confidence intervals, indicating limited remaining serial correlation.
- Ljung-Box Test: This test evaluates whether residual autocorrelation remains significant. Insignificant p-values suggest that the residuals behave similarly to white noise.

### 3.5 Intermediate Conclusion
Despite removing most linear structure, the residuals still displayed periods of changing variance, known as heteroscedasticity. This justified the use of GARCH (1, 1) model in the next stage to capture volatility clustering. 


## 4.0 GARCH (1, 1)


## 5.0 Feature Engineering 
Several features were engineered to improve predictive performance:
- Lagged USEP values to capture short-term temporal dependency.
- GARCH residuals to incorporate volatility-related information.
- Standardized GARCH residuals to normalize volatility effects.

The dataset was then divided into training and testing subsets using a time-series split approach, ensuring that temporal ordering was preserved. 

This feature engineering process allowed the Random Forest and BiLSTM models to learn both trend and volatility characteristics from the time series. 

## 6.0 Random Forest Regressor 
A Random Forest Regressor was trained using the engineered features. Hyperparameter tuning was performed using GridSearchCV with TimeSeriesSplit cross-validation. Parameters such as the number of estimators, maximum tree depth, minimum samples per split, and minimum samples per leaf were optimized. 

The Random Forest model was selected because of its ability to model non-linear relationships and interaction between features without requiring strong statistical assumptions. 


## 7.0 BiLSTM Regressor 




## 8.0 Model Comparison 

### 8.1 Random Forest Regressor
#### 8.1.1 Training Performance 
- MAE: 3.7303
- RMSE: 10.8280
- R2 Score: 0.9344

#### 8.1.2 Testing Performance
- MAE: 6.1488
- RMSE: 7.6781
- R2 Score: 0.8992


## 9.0 Future Work 



## 10.0 Conclusion 


## 11.0 Libraries
- Sklearn
- Keras
- StatsModels

## 12.0 Datasets
https://www.nems.emcsg.com/nems-prices

## References















