import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings

# suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# global file constants
OUTFILE = r"/Users/simonpritchard/Documents/Academics/Sophomore Year/VS Code Projects/Stats 207/Decorrelated Timeseries/LogRawWithSox.csv"

# loads PHLX Semiconductor Index and Tokyo Electron prices from Yahoo finance
# returns daily closing prices for these indices
def load_data() -> (pd.DataFrame, pd.DataFrame):
    # PHLX Semiconductor Index
    sox = yf.download("^SOX", start="2000-01-01", end="2023-09-01", interval="1d")

    # Tokyo Electron stock
    toel = yf.download("8035.T", start="2000-01-01", end="2023-09-01", interval="1d")

    # keep only the closing stock prices
    sox, toel = sox[["Close"]], toel[["Close"]]

    # remove nulls & align indices
    sox, toel = sox.dropna(), toel.dropna()
    sox, toel = sox.align(toel, join="inner")
    return sox, toel

# smooths the financial timeseries
def smooth_df(df, method="monthly_mean"):
    # average daily closing prices for each month, and return monthly prices
    if method == "monthly_mean":
        df = df.resample("MS").mean()
        
    # add daily changes for each month
    if method == "monthly_add":
        df = df.resample("MS").sum()

    df.index = df.index.to_period('M')
    return df

# Adds a log transform to account for heteroscedasticity
def log_data(df, column_name):
    df[column_name] = np.log(df[column_name])
    return df

# Main script
sox, toel = load_data()

toel_raw = smooth_df(toel, method="monthly_mean")
sox_raw = smooth_df(sox, method="monthly_mean")

# Log-transform the data
toel_log = log_data(toel_raw.copy(), "Close")
sox_log = log_data(sox_raw.copy(), "Close")

# Rename columns to avoid conflicts during merge
toel_log.rename(columns={"Close": "Log_Close_toel"}, inplace=True)
sox_log.rename(columns={"Close": "Log_Close_sox"})

# Merge the two datasets
merged_data = pd.merge(toel_log, sox_log, how='inner', left_index=True, right_index=True)

# Save the merged data to a CSV file
merged_data.to_csv(OUTFILE, encoding='utf-8')

print(f"Data successfully saved to {OUTFILE}")


"""# decorrelates Tokyo Electron prices from the SOX
# computes relative changes in Tokyo Electron closing price compared to SOX's closing price
# i.e., examines daily changes in Tokyo Electron price, then subtracts off SOX's daily changes
# standard scaling implemented after first differencing to control for SOX's larger swings and raw value 
def decorrelate(sox, toel, method="regression") -> pd.DataFrame:

    # compute Tokyo Electron & SOX first differences
    toel_diff = toel["Close"].diff().dropna().rename("price_diff")
    sox_diff = sox["Close"].diff().dropna().rename("price_diff")

    # align indices after differencing
    aligned_sox, aligned_toel = sox_diff.align(toel_diff, join="inner")

    # standard scale the differenced data
    toel_scaler = StandardScaler()
    sox_scaler = StandardScaler()
    toel_scaled = toel_scaler.fit_transform(aligned_toel.values.reshape(-1, 1))
    sox_scaled = sox_scaler.fit_transform(aligned_sox.values.reshape(-1, 1))

    print("Differenced and scaled TOEL:\n", toel_scaled)
    print("Differenced and scaled SOX:\n", sox_scaled)

    # inverse transform to the scale of Tokyo Electron changes
    double_diff_scaled = toel_scaled - sox_scaled
    return pd.DataFrame(toel_scaler.inverse_transform(double_diff_scaled), columns=["price_diff"], index=aligned_toel.index)
"""