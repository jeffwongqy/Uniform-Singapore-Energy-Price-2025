# Uniform Singapore Energy Price 2025

<img width="950" height="288" alt="Screenshot 2026-05-23 223532" src="https://github.com/user-attachments/assets/89b7912b-be54-4583-b4e5-e66882d580eb"/>

## 1.0 Introduction
Time-series forecasting plays an important role in understanding temporal dynamics in financial and energy markets, particularly in capturing evolving price behaviour and volatility. In this study, the USEP dataset is examined using a hybrid modelling framework that integrates both statistical and machine learning approaches. An ARIMA(1,1,1) model is first employed to characterise the underlying linear structure and trend components in the series. To account for time-varying volatility observed in the residuals, a GARCH(1,1) model is subsequently introduced. Features derived from both models are then used as inputs for two predictive models—a Random Forest Regressor and a BiLSTM network—to evaluate their effectiveness in capturing complex, non-linear patterns in the data.

The workflow includes:
1. Data preprocessing and exploratory analysis.
2. ARIMA modelling
3. GARCH modelling
4. Feature engineering using lagged values and residual statistics.
5. Random Forest and BiLSTM training and hyperparameter optimisation
6. Model Evaluation using MAE, RMSE, and R2 metrics. 


## 2.0 Problem Statement 
The objective of this study is to develop a predictive framework for forecasting the Uniform Singapore Energy Price (USEP) using statistical and machine learning approaches. Electricity prices are highly volatile and influenced by both historical trends and market uncertainty. To address this challenge, it combines ARIMA for time-series trend modeling, GARCH for volatility estimation, and Random Forest Regression/ BiLSTM for nonlinear predictive learning. Furthermore, it also evaluates whether integrating econometric models with machine learning and deep learning techniques can improve forecasting accuracy. 


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

<img width="500" height="350" alt="pacf" src="https://github.com/user-attachments/assets/29868441-91b5-4e72-ad60-8ead5dabdfa5" />

- The ACF plot also displayed a significant spike at lag 1 with gradual decay afterward, suggesting a first-order moving-average process. Therefore, q = 1.

<img width="500" height="350" alt="acf" src="https://github.com/user-attachments/assets/27024e13-e4b6-4757-97fb-2ef0283c9bcb" />


### 3.2 ARIMA Model Representation
The fitted ARIMA model can be expressed as: 


### 3.3 Summary Output Interpretation
The ARIMA summary output provides statistical information about the fitted model:

<img width="500" height="350" alt="arima" src="https://github.com/user-attachments/assets/93203cbf-730c-41a7-a71d-c75339a833e1" />

- AR coefficient (ar.L1) measures the relationship between the current differenced value and its previous value.
- MA coefficient (ma.L1) measures the relationship between the current value and the previous forecast error.
- Standard Error indicates the uncertainty associated with each estimated coefficient.
- z-statistic and p-value determine whether the coefficients are statistically significant. (i.e. Small p-values (typically less than 0.05) indicate that the coefficient contributes meaningfully to the model.)

### 3.4 Residual Diagnostics 
After fitting the ARIMA model, residual diagnostics were examined to assess whether the residuals behaved like white noise. 
- Residual Plot: Residuals fluctuated randomly around zero without obvious patterns, suggesting that most linear dependencies were removed.
- Histogram/ Density Plot: The residual distribution appeared approximately centered around zero, although some heavy tails remained due to volatility in electricity prices.
- ACF of residuals: The residuals autocorrelations were relatively small and mostly within confidence intervals, indicating limited remaining serial correlation.
- Ljung-Box Test: This test evaluates whether residual autocorrelation remains significant. Insignificant p-values suggest that the residuals behave similarly to white noise.


<img width="500" height="350" alt="download" src="https://github.com/user-attachments/assets/7c2b3ec0-4a47-4c90-a01f-5033fd0be078" />

### 3.5 Intermediate Conclusion
Despite removing most linear structure, the residuals still displayed periods of changing variance, known as heteroscedasticity. This justified the use of GARCH (1, 1) model in the next stage to capture volatility clustering. 


## 4.0 GARCH (1, 1)
After fitting the ARIMA model, the residuals were modeled using a GARCH(1,1) process with a Student's t-distribution to capture volatility clustering and time-varying conditional variance in the USEP series. 

The ARIMA residuals still showed heteroscedasticity, where periods of high volatility were followed by periods of low volatility. The squared residual ACF showed significant correlation at lag 1. 

The GARCH model contains:
- p = 1: one lag of squared residual (ARCH term)
- q = 1: one lag of conditional variance (GARCH term)

### 4.1 Summary Output Interpretation
The GARCH summary output consists of the mean model and volatility model coefficients:

<img width="500" height="350" alt="Screenshot 2026-05-25 103737" src="https://github.com/user-attachments/assets/000bf703-266e-419e-b225-83086df8654b" />

- omega: represents the baseline or long-run variance - a significant p-value indicates stable background volatility.
- alpha1: measures how strongly recent shocks affect current volatility - a significant p-value below 0.05 indicates that sudden price changes have a meaningful impact on volatility.
- beta1: measures how persistent volatility remains over time - a significant p-value confirms persistent conditional variance.

### 4.2 Intermediate Conclusion
GARCH (1, 1) was sufficient to capture short-term volatility persistence without making the model unnecessarily complex, due to the first-order ARCH and GARCH structure. 


## 5.0 Feature Engineering 
Several features were engineered to improve predictive performance:
- Lagged USEP values to capture short-term temporal dependency.
- GARCH residuals to incorporate volatility-related information.
- Standardized GARCH residuals to normalize volatility effects.

The dataset was then divided into training and testing subsets using a time-series split approach, ensuring that temporal ordering was preserved. 

This feature engineering process allowed the Random Forest and BiLSTM models to learn both trend and volatility characteristics from the time series. 

## 6.0 Random Forest Regressor 
A Random Forest Regressor was trained using the engineered features. Hyperparameter tuning was performed using GridSearchCV with TimeSeriesSplit cross-validation. Parameters such as the number of estimators, maximum tree depth, minimum samples per split, and minimum samples per leaf were optimized. 

The Random Forest model was selected because of its ability to model non-linear relationships and interactions between features without requiring strong statistical assumptions. 


## 7.0 BiLSTM
A Bidirectional Long Short-Term Memory (BiLSTM) model was implemented to capture nonlinear temporal dependencies in the USEP time series. Compared to a Random Forest regressor, the BiLSTM processes information in both forward and backward directions, allowing the model to learn richer sequential patterns from historical electricity price movements. 

### 7.1 Hyperparameter Tuning 
To improve model performance, hyperparameter tuning was performed using TimeSeriesSplit cross-validation, which preserves the chronological order of the dataset and prevents future data leakage. Different numbers of neurons were evaluated in the dense layers to optimize feature refinement after the BiLSTM layers extracted complex temporal patterns from the time-series data. The dense layers transform these learned sequential features into final prediction outputs, where too few neurons may lead to underfitting, while too many may increase overfitting and computational complexity. Therefore, different dense neuron configurations were evaluated to achieve a balance between model complexity, learning capability, and generalization performance.

### 7.2 Architecture 
The final optimized BiLSTM architecture consisted of three main components. 
1. A bidirectional LSTM layer with tanh activation was used to learn both forward and backward temporal relationships in the USEP time series, allowing the model to capture long-term sequential dependencies and contextual information from past and future observations simultaneously.
2. A second LSTM layer with tanh activation, which further refined the high-level temporal features extracted from the BiLSTM layer and improved sequence representation learning.
3. The model included a Dense hidden layer with 16 neurons and ReLU activation, which transformed the extracted sequential features into more meaningful nonlinear representations while improving training efficiency and reducing vanishing gradient issues.
4. A final Dense(1) output layer was then used to generate the final USEP price prediction.
5. The model was trained using the Adam optimizer and Mean Squared Error (MSE) loss function. 

### 7.3 Learning Behaviour 
The training and validation loss curves showed a steady decrease throughout training, indicating that the model successfully learned the temporal structure of the USEP data. 

The MSE curve also decreased consistently over epochs, suggesting improved prediction accuracy and stable convergence during training. 

No major divergence between training and validation loss was observed, indicating limited overfitting and good generalization performance. 

## 8.0 Model Comparison 

### 8.1 Random Forest Regressor
#### 8.1.1 Training Performance 
- MAE: 3.7303
- RMSE: 10.8280
- R2 Score: 0.9344

<img width="850" height="547" alt="rf_train" src="https://github.com/user-attachments/assets/880cf6f3-7480-4671-9488-62a2e42a2dc7" />



#### 8.1.2 Testing Performance
- MAE: 6.1488
- RMSE: 7.6781
- R2 Score: 0.8992

<img width="850" height="547" alt="rf_test" src="https://github.com/user-attachments/assets/5f7e6d5a-0973-46a8-b974-93377d70fe16" />


### 8.2 BilSTM


#### 8.2.1 Training Performance


#### 8.2.2 Testing Performance

## 9.0 Future Work 



## 10.0 Conclusion 
The tuned BiLSTM model outperformed the Random Forest Regressor across all evaluation metrics. Its ability to learn sequential dependencies directly from time-series data allowed it to achieve better forecasting accuracy and stronger generalization performance on unseen data. 


## 11.0 Libraries
- Sklearn
- Keras
- StatsModels

## 12.0 Datasets
https://www.nems.emcsg.com/nems-prices

## References
[1] Montgomery, D. C., Jennings, C. L., & Kulahci, M. (2016). Introduction to time series analysis and forecasting. Wiley.

‌[2] Ruppert, D. (2016). Statistics and data analysis for financial engineering. Springer-Verlag New York.

‌[3] James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An Introduction to Statistical Learning. Springer Nature. https://www.statlearning.com/

‌[4] Aurelien Geron. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow : concepts, tools, and techniques to build intelligent systems. O’reilly.

‌















