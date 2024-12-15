# Weather Predictor Project

## Overview

This project implements a weather prediction model using machine learning. The model uses historical weather data (daily temperatures from 1970 to 2022) to predict future temperatures. The model is based on **Ridge Regression**, a type of linear regression that applies regularization to improve the model's generalization to unseen data.

The project uses the `scikit-learn` library to build and train the model, while `pandas` is used for data preprocessing and handling.

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `scikit-learn`

You can install the necessary libraries using `pip`:

```bash
pip install pandas scikit-learn
```

## Dataset

The dataset used for this project contains historical daily weather conditions from 1970 to 2022. The data includes columns such as:

- **DATE**: The date of the observation.
- **TMAX**: Maximum temperature in Fahrenheit.
- **TMIN**: Minimum temperature in Fahrenheit.
- **PRCP**: Precipitation.
- **SNWD**: Snow depth in inches.

Data can be obtained from this [raw GitHub URL](https://raw.githubusercontent.com/dataquestio/project-walkthroughs/master/temperature_prediction/weather.csv).

## Data Cleaning

The dataset undergoes several preprocessing steps, including:

1. **Null Value Handling**: Columns with more than 5% missing values are removed.
2. **Forward Filling**: Missing values are filled using the previous valid value.
3. **Feature Engineering**:
   - Rolling averages for `tmax`, `tmin`, and `prcp` are calculated over 3 and 14 days.
   - Expanding means are calculated for each month and day of the year.

## Machine Learning Model

### Ridge Regression

The model uses **Ridge Regression** to predict the maximum temperature (`tmax`) for the following day based on historical data. The model is trained using past weather data, and we use time series cross-validation to evaluate the model.

### Cross-Validation

To evaluate the model, a custom **backtest** function is used that simulates predicting the weather using a sliding window approach. The model is trained on data up to a specific point, and predictions are made for the following 90 days.

## Results

The model is evaluated using **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**. The backtest function simulates predictions on a rolling window, allowing for the calculation of prediction accuracy over time.

### Example Output

```python
predicted_temp = predict_temperature("2023-08-01", rr, weather, predictors)
print(f"Predicted temperature for 2023-08-01: {predicted_temp:.2f} °C")
```

## Improvements

- The model can be further enhanced by exploring other regression models such as **Lasso Regression** or **ElasticNet**.
- Hyperparameter tuning can be applied to improve the model's performance.
- Time series forecasting models such as **ARIMA** or **LSTM (Long Short-Term Memory)** networks could be explored for better accuracy in forecasting.
