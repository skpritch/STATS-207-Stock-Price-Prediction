import os
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# suppress FutureWarning and ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

# global file constants
INFOLDER_RADAR = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Patent Features"
INFILE_TIMESERIES = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Decorrelated Timeseries/LogRawWithSox.csv"
OUTFILE = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Results/SARIMAOnDecorrelated.csv"
TEMPOUTFILE = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Temp Results/SARIMA_results_temp_"

# collects all csvs from the radar features folder and reads the single timeseries file
def import_data():
    feature_files = [file for file in os.listdir(INFOLDER_RADAR) if file.endswith('.csv')]
    feature_dfs = [(file, pd.read_csv(os.path.join(INFOLDER_RADAR, file), index_col=0, parse_dates=True)) for file in feature_files]
    
    # Read the single time series file
    timeseries_df = pd.read_csv(INFILE_TIMESERIES, index_col=0, parse_dates=True)
    
    return feature_dfs, timeseries_df

# calculates AIC for SARIMAX
def calculate_sarimax_aic(n, sse, k):
    return n * np.log(sse / n) + 2 * k

# generate AIC, train error, and test error to evaluate a given SARIMAX model,
# performing timeseries crossvalidation for the test error
def evaluate_sarimax_model_ts_cv(radar_data, toel, sox, time_lag, order, seasonal_order):
    
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
    # then feed correct data for that year and predict the next year with SARIMAX model
    unique_years = sorted(aligned_data.index.year.unique())
    start_year = unique_years[int(len(unique_years) * 0.2)]
    for year in unique_years[unique_years.index(start_year):]:
        train_idx = x.index.year < year
        test_idx = x.index.year == year
        x_train, x_test = x[train_idx], x[test_idx]
        toel_train, toel_test = y[train_idx], y[test_idx]
        sox_train, sox_test = aligned_sox[train_idx], aligned_sox[test_idx]

        # Ensure the data is in the correct shape for StandardScaler
        scaler_toel = StandardScaler()
        scaler_sox = StandardScaler()

        # Ensure the data is a NumPy array and reshape it
        scaled_toel_train = scaler_toel.fit_transform(toel_train.values.reshape(-1, 1)).flatten()  # Changed to to_numpy().reshape(-1, 1)
        scaled_sox_train = scaler_sox.fit_transform(sox_train.values.reshape(-1, 1)).flatten()  # Changed to to_numpy().reshape(-1, 1)
        scaled_toel_test = scaler_toel.transform(toel_test.values.reshape(-1, 1)).flatten()  # Changed to to_numpy().reshape(-1, 1)
        scaled_sox_test = scaler_sox.transform(sox_test.values.reshape(-1, 1)).flatten()  # Changed to to_numpy().reshape(-1, 1)

        # Subtract the scaled SOX data from the corresponding TOEL values
        diff_toel_train = scaled_toel_train - scaled_sox_train
        diff_toel_test = scaled_toel_test - scaled_sox_test

        # Fit SARIMAX model
        model = SARIMAX(diff_toel_train, exog=x_train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        diff_train_predictions = model_fit.fittedvalues
        diff_test_predictions = model_fit.predict(start=len(diff_toel_train), end=len(diff_toel_train) + len(diff_toel_test) - 1, exog=x_test)

        # Revert the predictions to the original scale
        train_predictions = diff_train_predictions.to_numpy() + scaled_sox_train
        test_predictions = diff_test_predictions.to_numpy() + scaled_sox_test

        pred_train_original = scaler_toel.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
        pred_test_original = scaler_toel.inverse_transform(test_predictions.reshape(-1, 1)).flatten()

        toel_train_original = scaler_toel.inverse_transform(scaled_toel_train.reshape(-1, 1)).flatten()
        toel_test_original = scaler_toel.inverse_transform(scaled_toel_test.reshape(-1, 1)).flatten()

        # Calculate errors
        train_errors.extend(((np.exp(pred_train_original) - np.exp(toel_train_original)) ** 2).tolist())
        test_errors.extend(((np.exp(pred_test_original) - np.exp(toel_test_original)) ** 2).tolist())

        # Calculate AIC
        n = len(toel_train)
        k = x_train.shape[1] + 1  # number of features + intercept
        sse = np.sum((pred_train_original - toel_train_original) ** 2)
        aic = calculate_sarimax_aic(n, sse, k)
        aic_values.append(aic)

    avg_train_error = np.mean(train_errors)
    avg_test_error = np.mean(test_errors)
    avg_aic = np.mean(aic_values)

    return avg_train_error, avg_test_error, avg_aic

# produces SARIMAX models for all radars & time lags, evaluating their predictions against real-world Tokyo Electron stock prices
def sarimax_model(feature_dfs, timeseries_df, p_values, d_values, q_values, P_values, D_values, Q_values, s_values) -> list[dict]:

    all_rows = []
    iteration = 0
    # Split timeseries_df into TOEL and SOX data
    toel = timeseries_df[['Log_Close_toel']]
    sox = timeseries_df[['Log_Close_sox']]

    # looping through all pairs of radar features & time lags
    for feature_file, df1 in tqdm(feature_dfs):
        for time_lag in tqdm(range(0, 11)):
            time_lag *= 12
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        for P in P_values:
                            for D in D_values:
                                for Q in Q_values:
                                    for s in s_values:
                                        try:
                                            with warnings.catch_warnings():
                                                warnings.simplefilter("error", UserWarning)
                                            print(f"p: {p}, d: {d}, q: {q}, P: {P}, D: {D}, Q: {Q}, s: {s}")
                                            train_MSE, test_MSE, aic = evaluate_sarimax_model_ts_cv(df1, toel, sox, time_lag, (p, d, q), (P, D, Q, s))
                                            all_rows.append({
                                                "feature_file": feature_file,
                                                "model_type": "SARIMAX",
                                                "time_lag": time_lag,
                                                "order": (p, d, q),
                                                "seasonal_order": (P, D, Q, s),
                                                "AIC": aic,
                                                "train_MSE": train_MSE,
                                                "test_MSE": test_MSE
                                            })
                                        except Exception as e:
                                            print(f"Skipping p: {p}, d: {d}, q: {q}, P: {P}, D: {D}, Q: {Q}, s: {s}, e: {e}")
                                            continue
            filename = TEMPOUTFILE + str(iteration) + ".csv"
            iteration += 1
            pd.DataFrame(all_rows).to_csv(filename, index=False)
    return pd.DataFrame(all_rows)

# Tunable parameters for SARIMAX
p_values = [0, 1, 2]
d_values = [0, 1, 2]
q_values = [0, 1, 2]
P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]
s_values = [3, 12]  # Monthly seasonality

feature_dfs, timeseries_df = import_data()

sarimax_results = sarimax_model(feature_dfs, timeseries_df, p_values, d_values, q_values, P_values, D_values, Q_values, s_values)

sarimax_results.to_csv(OUTFILE, index=False)
print(f"Results successfully saved to {OUTFILE}")
