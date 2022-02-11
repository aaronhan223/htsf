import eia
import pandas as pd
import numpy as np
import pdb


def retrieve_time_series(api, series_ID):
    """
    Return the time series dataframe, based on API and unique Series ID
    """
    #Retrieve Data By Series ID 
    series_search = api.data_by_series(series=series_ID)
    ##Create a pandas dataframe from the retrieved time series
    df = pd.DataFrame(series_search)
    return df


def get_ts_eia():
    #Create EIA API using your specific API key
    api_key = "eef610118bd997016bf11535f33dc3d8"
    api = eia.API(api_key)
    #Declare desired series ID
    series_ID = 'PET.RWTC.D'
    price_df = retrieve_time_series(api, series_ID)
    price_df.reset_index(level=0, inplace=True)
    #Rename the columns for easier analysis
    price_df.rename(columns={'index':'Date', price_df.columns[1]:'WTI_Price'}, inplace=True)
    #Format the 'Date' column 
    price_df['Date'] = price_df['Date'].astype(str).str[:-3]
    #Convert the Date column into a date object
    price_df['Date'] = pd.to_datetime(price_df['Date'], format='%Y %m%d')
    #Subset to only include data going back to 2014
    # price_df = price_df[(price_df['Date']>='2014-01-01')]

    #Convert the time series values to a numpy 1D array
    points = np.array(price_df['WTI_Price'])
    return points