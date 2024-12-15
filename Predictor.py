# Abdelrahman Ahmed
# Weather Predictor using scikit-learn to predict from the weather.csv sheet containing all daily weather conditions from 1970 to 2022

import pandas as pd

weather = pd.read_csv('weather.csv', index_col="DATE")

# Date Cleaning

null_percentage = weather.apply(pd.isnull).sum()/weather.shape[0]
valid_columns = weather.columns[null_percentage < .05]

weather = weather[valid_columns].copy()

weather.columns = weather.columns.str.lower()

weather = weather.ffill()

weather.apply(pd.isnull).sum()

weather.apply(lambda x: (x == 9999).sum())

# Type Checks and Conversions

weather.index = pd.to_datetime(weather.index)

weather.index.year.value_counts().sort_index()

weather["target"] = weather.shift(-1)["tmax"]  # Target Column for ML prediction
weather = weather.ffill()

# ML Ridge Regression Model

from sklearn.linear_model import Ridge

rr = Ridge(alpha=.1)
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]

# Time Series Cross Validation
# generate prediction for data set except first ten years
def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []
    
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)

predictions = backtest(weather, rr, predictors)

# Mean Accuracy Error test
from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(predictions["actual"], predictions["prediction"])

predictions.sort_values("diff", ascending=False)

pd.Series(rr.coef_, index=predictors)

# Improvments

def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()  # Calculate the rolling mean for the days (horizon)
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])  # Calculate the difference percentage
    return weather
    
rolling_horizons = [3, 14]
for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)

def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)

weather = weather.iloc[14:,:]
weather = weather.fillna(0)

predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]

predictions = backtest(weather, rr, predictors)

mean_absolute_error(predictions["actual"], predictions["prediction"])
mean_squared_error(predictions["actual"], predictions["prediction"])

predictions.sort_values("diff", ascending=False)

def predict_temperature(date, model, weather, predictors):
    date = pd.to_datetime(date)
    historical_data = weather[weather.index <= date]
    
    if len(historical_data) < 14:
        raise ValueError(f"Insufficient data to predict for {date}. Need at least 14 days of data.")
    
    row = historical_data.iloc[-1].copy()
    
    # Feature generation for the future date
    for horizon in [3, 14]:
        for col in ["tmax", "tmin", "prcp"]:
            row[f"rolling_{horizon}_{col}"] = historical_data[col].rolling(horizon).mean().iloc[-1]
            row[f"rolling_{horizon}_{col}_pct"] = (row[f"rolling_{horizon}_{col}"] - row[col]) / row[col]
    
    for col in ["tmax", "tmin", "prcp"]:
        row[f"month_avg_{col}"] = historical_data[col].groupby(historical_data.index.month, group_keys=False).apply(lambda df: df.expanding(1).mean()).iloc[-1]
        row[f"day_avg_{col}"] = historical_data[col].groupby(historical_data.index.day_of_year, group_keys=False).apply(lambda df: df.expanding(1).mean()).iloc[-1]

    row = row.fillna(0)
    
    input_data = row[predictors].values.reshape(1, -1)
    predicted_temp = model.predict(input_data)
    
    return predicted_temp[0]






