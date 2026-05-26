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
The Random Forest Regressor was selected because it is effective in modeling complex nonlinear relationships and interactions between engineered features without requiring strong statistical assumptions. Its ensemble structure combines multiple decision trees, which improves prediction stability and reduces overfitting compared to a single decision tree model. 

### 6.1 Hyperparameter Tuning
To improve model performance, hyperparameter tuning was performed using GridSearchCV with TimeSeriesSplit cross-validation, which preserves the chronological order of time-series and prevents future data leakage. The tuning process evaluated different combinations of key parameters, including the number of trees (n_estimators = 50 to 100), maximum tree depth (max_depth = 8 to 10), minimum samples required at leaf nodes (min_samples_leaf = 1 to 5), and minimum samples required for node splitting (min_samples_split = 2 to 5). Model performance was evaluated using negative mean squared error, and the parameter combination with the best validation performance was selected as the final optimized Random Forest model. 

````
# initialize the base random forest regressor
rfr = RandomForestRegressor(random_state = 42)

# define the hyperparameter grid
param_grid = {'n_estimators': [50, 60, 70, 80, 90, 100],
              'max_depth': [8, 9, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5],
              'min_samples_split': [2, 3, 4, 5]}

# set up time series split
tscv = TimeSeriesSplit(n_splits = 5)

# set up the grid search cross-validation configuration
grid_search = GridSearchCV(estimator = rfr,
                           param_grid = param_grid,
                           cv = tscv,
                           n_jobs = 1,
                           verbose = 3,
                           scoring = 'neg_mean_squared_error',
                           return_train_score = True,
                           refit = True)

# execute the search over all parameter combination
grid_search.fit(X_train, y_train)

````


## 7.0 BiLSTM
A Bidirectional Long Short-Term Memory (BiLSTM) model was implemented to capture nonlinear temporal dependencies in the USEP time series. Compared to a Random Forest regressor, the BiLSTM processes information in both forward and backward directions, allowing the model to learn richer sequential patterns from historical electricity price movements. 

### 7.1 Hyperparameter Tuning 
To improve model performance, hyperparameter tuning was performed using TimeSeriesSplit cross-validation, which preserves the dataset's chronological order and prevents future data leakage. Different numbers of neurons were evaluated in the dense layers to optimize feature refinement after the BiLSTM layers extracted complex temporal patterns from the time-series data. The dense layers transform these learned sequential features into final prediction outputs, where too few neurons may lead to underfitting, while too many may increase overfitting and computational complexity. Therefore, various dense neural network configurations were evaluated to balance model complexity, learning capability, and generalization performance.

````
# hyperparameters tuning
dense_neurons_layer1 = [8, 16, 32]
dense_neurons_layer2 = [8, 16, 32]

best_rmse = float("inf")
best_params = None

for dense_neuron_layer1 in dense_neurons_layer1:
  for dense_neuron_layer2 in dense_neurons_layer2:
    rmses = []

    for train_idx, val_idx in tscv.split(X_train_reshaped):
      X_train_fold, X_val_fold = X_train_reshaped[train_idx], X_train_reshaped[val_idx]
      y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

      model = Sequential()
      model.add(Bidirectional(LSTM(128, kernel_regularizer = "l2", activation = "tanh", return_sequences = True), input_shape = (1, 4)))
      model.add(LSTM(32, kernel_regularizer = "l2", activation = "tanh"))
      model.add(Dense(dense_neuron_layer1, activation = "relu"))
      model.add(Dense(dense_neuron_layer2, activation = "relu"))
      model.add(Dense(1))
      model.compile(optimizer = "adam", loss = "mse", metrics = ['mse'])

      model.fit(X_train_fold, y_train_fold,
                epochs = 100,
                batch_size = 32,
                validation_split = 0.1,
                verbose = 1)

      y_pred_val = model.predict(X_val_fold)
      rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
      rmses.append(rmse)

    avg_rmse = np.mean(rmses)

    print("dense_neuron_layer1 = {}, dense_neuron_layer2 = {}, RMSE = {:.4f}".format(dense_neuron_layer1, dense_neuron_layer2, avg_rmse))
    if avg_rmse < best_rmse:
      best_rmse = avg_rmse
      best_params = {'dense_neuron_layer1': dense_neuron_layer1,
                     'dense_neuron_layer2': dense_neuron_layer2}

````

The best parameters and RMSE were found to be {'dense_neuron_layer1': 32, 'dense_neuron_layer2': 32} and 21.80629527366256, respectively. 


### 7.2 Architecture 
The final optimized BiLSTM architecture consisted of three main components. 

````
op_model = Sequential()
op_model.add(Bidirectional(LSTM(128,
                                kernel_regularizer = "l2",
                                activation = "tanh",
                                return_sequences = True),
                           input_shape = (1, 4)))
op_model.add(LSTM(32, kernel_regularizer = "l2", activation = "tanh"))
op_model.add(Dense(32, activation = "relu"))
op_model.add(Dense(32, activation = "relu"))
op_model.add(Dense(1))
op_model.compile(optimizer = "adam", loss = "mse", metrics = ['mse'])

op_model.summary()

history = op_model.fit(X_train_reshaped, y_train,
                    epochs = 100,
                    validation_split = 0.1,
                    batch_size = 32,
                    verbose = 1)
````

1. Bidirectional LSTM Layer (128 units) - Captures forward and backward temporal dependencies in the USEP time series to learn long-term sequential patterns.
2. LSTM Layer (32 units) - Refines the temporal features extracted from the Bidirectional LSTM layer for improved sequence learning.
3. First Dense Layer (32 neurons, ReLU) - Transforms learned temporal features into non-linear representations for better prediction learning.
4. Second Dense Layer (32 neurons, ReLU) - Further fine-tune feature representations to improve model accuracy and learning capability.
5. Output Layer (1 neuron) - Produces the final predicted USEP value for the regression task.
6. Adam Optimizer - Optimizes model weights efficiently using adaptive gradient learning.
7. MSE Loss Function - Measures prediction error by calculating the average squared difference between actual and predicted values.
8. Validation Split (0.1) - Uses 10% of the training data to evaluate model generalization during training.
9. Batch Size (32) - Processes training samples in batches of 32 to improve computational efficiency and training stability.
10. Epochs (100) - Allows the model to iteratively learn patterns over 100 training cycle. 


### 7.3 Learning Behaviour 
The training and validation loss curves showed a steady decrease throughout training, indicating that the model successfully learned the temporal structure of the USEP data. 

<img width="500" height="350" alt="loss curve" src="https://github.com/user-attachments/assets/68931adc-d5eb-47e2-be8e-8f7dd0343372" />

The MSE curve also decreased consistently over epochs, suggesting improved prediction accuracy and stable convergence during training. 

<img width="500" height="350" alt="mse curve" src="https://github.com/user-attachments/assets/bbcc2293-a173-47c2-876f-ba98b61ceb11" />

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
- MAE: 3.6843
- RMSE: 10.7931
- R2 Score: 0.9348

<img width="850" height="547" alt="train_bilstm" src="https://github.com/user-attachments/assets/47b95149-1fc7-40b9-925c-04ea161a77b4" />

#### 8.2.2 Testing Performance
- MAE: 5.0070
- RMSE: 7.1711
- R2 Score: 0.9121

<img width="850" height="547" alt="test_bilstm" src="https://github.com/user-attachments/assets/ddf7a998-0b32-4fa9-8144-52d8c61729c7" />


## 9.0 Future Work 



## 10.0 Conclusion 
The tuned BiLSTM model outperformed the Random Forest Regressor across all evaluation metrics. Its ability to learn sequential dependencies directly from time-series data allowed it to achieve better forecasting accuracy and stronger generalization performance on unseen data. 


## 11.0 Libraries
- Sklearn
- Tensorflow
- Keras
- StatsModels
- Matplotlib
- Seaborn 

## 12.0 Datasets
https://www.nems.emcsg.com/nems-prices

## References
[1] Montgomery, D. C., Jennings, C. L., & Kulahci, M. (2016). Introduction to time series analysis and forecasting. Wiley.

‌[2] Ruppert, D. (2016). Statistics and data analysis for financial engineering. Springer-Verlag New York.

‌[3] James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An Introduction to Statistical Learning. Springer Nature. https://www.statlearning.com/

‌[4] Aurelien Geron. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow : concepts, tools, and techniques to build intelligent systems. O’reilly.

‌















