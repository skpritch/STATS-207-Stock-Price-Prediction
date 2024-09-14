import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

# suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# global file constants
INFOLDER_RADAR = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Patent Features"
INFILE_TIMESERIES = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Decorrelated Timeseries/LogRawWithSox.csv"
OUTFILE = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Results/LinearOnDecorrelated.csv"

# collects all csvs from the radar features folder and reads the single timeseries file
def import_data():
    feature_files = [file for file in os.listdir(INFOLDER_RADAR) if file.endswith('.csv')]
    feature_dfs = [(file, pd.read_csv(os.path.join(INFOLDER_RADAR, file), index_col=0, parse_dates=True)) for file in feature_files]
    
    # Read the single time series file
    timeseries_df = pd.read_csv(INFILE_TIMESERIES, index_col=0, parse_dates=True)
    
    return feature_dfs, timeseries_df

# calculates AIC for linear regression / OLS
def calculate_linear_aic(n, sse, k):
    return n * np.log(sse / n) + 2 * k

# generate AIC, train error, and test error to evaluate a given linear model,
# performing timeseries crossvalidation for the test error
def evaluate_linear_model_ts_cv(radar_data, toel, sox, time_lag):

    # ensure time_lag is integer
    if not isinstance(time_lag, int):
        raise TypeError("time_lag for year-by-year analysis must be an integer!")
    
    # shift radar data by time_lag and removes null rows
    radar_data.index = pd.PeriodIndex(radar_data.index, freq="M")
    toel.index = pd.PeriodIndex(toel.index, freq="M")
    sox.index = pd.PeriodIndex(sox.index, freq="M")

    radar_data = radar_data.shift(periods=time_lag, freq="M")
    radar_data = radar_data.dropna()
    aligned_data, aligned_toel = radar_data.align(toel, join="inner")
    x = radar_data.loc[aligned_data.index]
    y = toel.loc[aligned_toel.index]
    aligned_sox = sox.loc[aligned_toel.index]

    train_errors = []
    test_errors = []
    aic_values = []

    # timeseries cross-validation paradigm: start with 20% of the years for the first run
    # predict each subsequent year; evaluate error
    # then feed correct data for that year and predict the next year with linear model
    unique_years = sorted(aligned_data.index.year.unique())
    start_year = unique_years[int(len(unique_years) * 0.2)]
    for year in unique_years[unique_years.index(start_year):]:
        train_idx = x.index.year < year
        test_idx = x.index.year == year

        x_train, x_test = x[train_idx], x[test_idx]
        toel_train, toel_test = y[train_idx], y[test_idx]
        sox_train, sox_test = aligned_sox[train_idx], aligned_sox[test_idx]

        # Scale the data
        scaler_toel = StandardScaler()
        scaler_sox = StandardScaler()
        
        scaled_toel_train = scaler_toel.fit_transform(toel_train.values.reshape(-1, 1)).flatten()
        scaled_sox_train = scaler_sox.fit_transform(sox_train.values.reshape(-1, 1)).flatten()

        scaled_toel_test = scaler_toel.transform(toel_test.values.reshape(-1, 1)).flatten()
        scaled_sox_test = scaler_sox.transform(sox_test.values.reshape(-1, 1)).flatten()

        # Subtract the scaled SOX data from the corresponding TOEL values
        diff_toel_train = scaled_toel_train - scaled_sox_train
        diff_toel_test = scaled_toel_test - scaled_sox_test

        model = LinearRegression()
        model.fit(x_train, diff_toel_train)
        diff_train_predictions = model.predict(x_train)
        diff_test_predictions = model.predict(x_test)

        #Add back market data to reverse original differencing
        train_predictions = diff_train_predictions + scaled_sox_train
        test_predictions = diff_test_predictions + scaled_sox_test

        #Turn TS objects into numpy arrays for reshaping
        train_predictions = train_predictions.to_numpy()
        test_predictions = train_predictions.to_numpy()
        scaled_toel_train = train_predictions.to_numpy()
        scaled_toel_test = train_predictions.to_numpy()

        # Revert the predictions to the original scale
        pred_train_original = scaler_toel.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
        pred_test_original = scaler_toel.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
        toel_train_original = scaler_toel.inverse_transform(scaled_toel_train.reshape(-1, 1)).flatten()
        toel_test_original = scaler_toel.inverse_transform(scaled_toel_test.reshape(-1, 1)).flatten()

        # Calculate errors
        train_errors.extend(((np.exp(pred_train_original) - np.exp(toel_train_original)) ** 2).tolist())
        test_errors.extend(((np.exp(pred_test_original) - np.exp(toel_test_original)) ** 2).tolist())
 
        # Calculate AIC
        n = len(toel_train)
        k = x_train.shape[1] + 1 # number of features + intercept
        sse = np.sum((pred_train_original - toel_train_original) ** 2)
        aic = calculate_linear_aic(n, sse, k)
        aic_values.append(aic)
    
    avg_train_error = np.mean(train_errors)
    avg_test_error = np.mean(test_errors)
    avg_aic = np.mean(aic_values)

    return avg_train_error, avg_test_error, avg_aic

# produces linear models for all radars & time lags, evaluating their predictions against real-world Tokyo Electron stock prices
def linear_model(feature_dfs, timeseries_df) -> list[dict]:

    all_rows = []

    # Split timeseries_df into TOEL and SOX data
    toel = timeseries_df[['Log_Close_toel']]
    sox = timeseries_df[['Log_Close_sox']]

    # looping through all pairs of radar features & time lags
    for feature_file, df1 in tqdm(feature_dfs):
        for time_lag in tqdm(range(0, 11)):
            time_lag *= 12
            train_MSE, test_MSE, aic = evaluate_linear_model_ts_cv(df1, toel, sox, time_lag)
            all_rows.append({
                "feature_file": feature_file,
                "model_type": "linear",
                "time_lag": time_lag,
                "AIC": aic,
                "train_MSE": train_MSE,
                "test_MSE": test_MSE
            })
    
    return pd.DataFrame(all_rows)

feature_dfs, timeseries_df = import_data()

linear_results = linear_model(feature_dfs, timeseries_df)

linear_results.to_csv(OUTFILE, index=False)
print(f"Results successfully saved to {OUTFILE}")
