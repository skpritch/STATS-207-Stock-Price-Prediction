import os
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

# global file constants
INFOLDER_RADAR = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Patent Features"
INFILE_TIMESERIES = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Decorrelated Timeseries/LogRaw.csv"
OUTFILE = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Results/SARIMAOnRaw.csv"


# collects all csvs from folders and packages them as a list of dataframes for radar features and a single dataframe for timeseries
def import_data():
    feature_files = [file for file in os.listdir(INFOLDER_RADAR) if file.endswith('.csv')]

    feature_dfs = [(file, pd.read_csv(os.path.join(INFOLDER_RADAR, file), index_col=0, parse_dates=True)) for file in feature_files]
    timeseries_df = pd.read_csv(INFILE_TIMESERIES, index_col=0, parse_dates=True)

    return feature_dfs, timeseries_df

# calculates AIC for SARIMAX
def calculate_sarimax_aic(n, sse, k):
    return n * np.log(sse / n) + 2 * k

# generate AIC, train error, and test error to evaluate a given SARIMAX model,
# performing timeseries crossvalidation for the test error
def evaluate_sarimax_model_ts_cv(radar_data, toel, time_lag, p, d, q, P, D, Q, s):
    
    # ensure time_lag is integer
    if not isinstance(time_lag, int):
        raise TypeError("time_lag for year-by-year analysis must be an integer!")
    
    # shift radar data by time_lag and removes null rows
    # shifting by year_months, not by number of observations, to handle gaps in monthly data
    # ensure year_month in datetime format
    radar_data.index = pd.PeriodIndex(radar_data.index, freq="M")
    toel.index = pd.PeriodIndex(toel.index, freq="M")
    # shift data by time_lag months
    radar_data = radar_data.shift(periods=time_lag, freq="M")
    # drop nulls & align datasets
    radar_data = radar_data.dropna()
    aligned_data, aligned_toel = radar_data.align(toel, join="inner")
    x = radar_data.loc[aligned_data.index]
    y = toel.loc[aligned_toel.index]

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
        y_train, y_test = y[train_idx], y[test_idx]

        model = SARIMAX(y_train, exog=x_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)

        y_train_predictions = model_fit.fittedvalues
        y_test_predictions = model_fit.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=x_test)

        # Handle negative or zero values for exponential transformation
        y_train_exp = np.exp(y_train)
        y_train_predictions_exp = np.exp(y_train_predictions)
        y_test_exp = np.exp(y_test)
        y_test_predictions_exp = np.exp(y_test_predictions)

        y_train_exp = np.where(y_train_exp <= 0, np.nan, y_train_exp)
        y_train_predictions_exp = np.where(y_train_predictions_exp <= 0, np.nan, y_train_predictions_exp)
        y_test_exp = np.where(y_test_exp <= 0, np.nan, y_test_exp)
        y_test_predictions_exp = np.where(y_test_predictions_exp <= 0, np.nan, y_test_predictions_exp)

        train_errors.extend(((y_train_predictions_exp - y_train_exp) ** 2).flatten().tolist())
        test_errors.extend(((y_test_predictions_exp - y_test_exp) ** 2).flatten().tolist())

        n = len(y_train)
        k = x_train.shape[1] + 1  # number of features + intercept
        sse = np.sum((y_train_predictions_exp - y_train_exp) ** 2)

        # Calculate AIC
        n = len(y_train)
        k = x_train.shape[1] + 1  # number of features + intercept
        aic = calculate_sarimax_aic(n, sse, k)
        aic_values.append(aic)

    avg_train_error = np.mean(train_errors)
    avg_test_error = np.mean(test_errors)
    avg_aic = np.mean(aic_values)

    return avg_train_error, avg_test_error, avg_aic

# produces SARIMAX models for all radars & time lags, evaluating their predictions against real-world Tokyo Electron stock prices
def sarimax_model(feature_dfs, timeseries_df, p_values, d_values, q_values, P_values, D_values, Q_values, s_values) -> list[dict]:

    all_rows = []

    # Define TOEL from timeseries_df
    toel = timeseries_df

    # looping through all pairs of radar features & preprocessing methods
    for feature_file, df1 in feature_dfs:
        for time_lag in tqdm(range(0, 1)):
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
                                            train_MSE, test_MSE, aic = evaluate_sarimax_model_ts_cv(df1, toel, time_lag, p, d, q, P, D, Q, s)
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
