"""
This file loads in radar data for Tokyo Electron patents.
From the radar data, it extracts novelty, volume, breadth, and depth features.

TO USE:
(1) Specify your INFILE and OUTFILE paths at the top of the program.
(2) Make sure your OUTFILE path does not already exist; otherwise, the existing file will be overwritten!
Then run the file! It will automatically write all the feature data to a csv.

File written by Teddy Ganea.

"""

# TO DO:
# (1) dynamic KDE - parameter optimization
# (2) upgrade breadth & depth statistics

# system libraries
import os

# data libraries
import numpy as np
import pandas as pd

# feature extraction & engineering
import sklearn
from sklearn.neighbors import KernelDensity

# global file constants
INFILE = r"C:\Users\Administrator\Downloads\stock_price_prediction\TokyoElectron-fullwabstract-2024.csv"
OUTFILE = r"C:\Users\Administrator\Downloads\stock_price_prediction\TokyoElectron-fullwabstract-2024-featuredata.csv"

# computes novelty by calculating the center of mass for each year,
# then finding the length of the trendline vector between these centers of mass
def novelty(df) -> pd.DataFrame:

    # calculate center of mass for years
    centers_of_mass_df = df.groupby("year")[["x", "y"]].mean().reset_index()

    # calculate Euclidean distance between consecutive centers
    trendline_length = np.sqrt(centers_of_mass_df["x"].diff() ** 2 + centers_of_mass_df["y"].diff() ** 2)

    # prepare years index, excluding the first year
    # (since indices represent the SECOND year in the difference, and by definition there are no years after the last)
    shifted_years = centers_of_mass_df["year"].iloc[1:].reset_index(drop=True)
    
    # package results into dataframe for output, excluding first null entry in trendline length
    # (it's null because for the first year, there is no previous year from which to compute a difference)
    results_df = pd.DataFrame({
        "trendline_length": trendline_length.iloc[1:].reset_index(drop=True)
    })
    results_df.index = shifted_years
    results_df.index.name = "year"

    return results_df

# computes volume by calculating change in number of new documents for each year
def volume(df) -> pd.DataFrame:

    # count number of docs for each year
    number_of_docs_df = df.groupby("year").size().reset_index(name="number_of_docs")
    
    # calculate change in number of docs
    change_in_docs = number_of_docs_df["number_of_docs"].diff().shift(-1).iloc[:-1].astype(int)

    # prepare index
    shifted_years = number_of_docs_df["year"].iloc[1:].reset_index(drop=True)
    
    # package results into dataframe
    results_df = pd.DataFrame({
        "change_in_docs": change_in_docs.reset_index(drop=True)
    })
    results_df.index = shifted_years
    results_df.index.name = "year"
    
    return results_df

# computes change in breadth
# breadth is calculated by averaging distance of patents to the center of mass for each year
def breadth(df) -> pd.DataFrame:

    # deepcopy to preserve original radar dataframe
    breadth_df = df.copy()

    # calculate distance to center of mass for each patent point
    breadth_df["dist_to_center"] = np.sqrt((df["x"] - df.groupby("year")["x"].transform("mean")) ** 2
                                    + (df["y"] - df.groupby("year")["y"].transform("mean")) ** 2)
    
    # average distances for each year
    breadth_df = breadth_df.groupby("year")["dist_to_center"].mean().reset_index(name="avg_dist_center")
    
    # prepare index
    shifted_years = breadth_df["year"].iloc[1:].reset_index(drop=True)

    # calculate change in breadth
    change_in_breadth = breadth_df["avg_dist_center"].diff().shift(-1).iloc[:-1]
    
    # package results into dataframe
    results_df = pd.DataFrame({
        "change_in_breadth": change_in_breadth.reset_index(drop=True)
    })
    results_df.index = shifted_years
    results_df.index.name = "year"

    return results_df

# calculates depth, which measures how concentrated/clustered patents are
# uses kernel density estimation (KDE) to estimate density values for each patent point,
# then averages them to get overall density for each year
def depth(df) -> pd.DataFrame:

    depth_data = []

    # use kernel density estimation (https://scikit-learn.org/stable/modules/density.html)
    # to impute densities of all patents for each year

    # loop over each year and its corresponding patent data points
    for year, yearly_df in df.groupby("year"):

        xy = yearly_df[["x", "y"]].to_numpy()

        # for now, manually set bandwidth (main KDE parameter) to 1.0
        best_bandwidth = 1.0

        # fit KDE to entire data and obtain log of density values
        log_densities = KernelDensity(
            bandwidth=best_bandwidth).fit(xy).score_samples(xy)
            
        # convert log densities to average density for year
        avg_density = np.exp(log_densities).mean()

        # append year and density to depth_data list
        depth_data.append([year, avg_density])

    # convert depth_data list to dataframe
    depth_df = pd.DataFrame(depth_data, columns=["year", "density"])

    # calculate year-over-year change in density
    shifted_years = depth_df["year"].iloc[1:].reset_index(drop=True)
    change_in_density = depth_df["density"].diff().shift(-1).iloc[:-1]

    # package results into dataframe
    results_df = pd.DataFrame({
        "change_in_density": change_in_density.reset_index(drop=True)
    })
    results_df.index = shifted_years
    results_df.index.name = "year"

    return results_df

# our data pipeline! reads radar csvs and returns dataframe of features over years
def prepare_radar_data() -> pd.DataFrame:

    # the csv's format from the Radar should be:
    # cl# (cluster number); x; y; counts; ID; year; title; other patent-specific data ...; publication_date
    df = pd.read_csv(INFILE).iloc[:, :7]

    data = pd.merge(novelty(df), volume(df), left_index=True, right_index=True, how="inner")
    data = pd.merge(data, breadth(df), left_index=True, right_index=True, how="inner")
    data = pd.merge(data, depth(df), left_index=True, right_index=True, how="inner")

    data.to_csv(OUTFILE, encoding='utf-8')

prepare_radar_data()