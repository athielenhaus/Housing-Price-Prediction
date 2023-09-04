import pandas as pd
import numpy as np


# Get data from CSV
def import_data():
    df = pd.read_csv('C:/Users/Arne/Documents/GitHub/Housing-Price-Prediction/data/regression_data.csv', na_values=np.nan)  # encodes missing values so that ML model can handle them
    return df

# Drop unneccesary columns
def preprocess_df(df):
    df_new = df.copy()
    df_new = df_new.drop(['id'], axis= 1)
    df_new["date"] = pd.to_datetime(df_new["date"])
    return df_new


# basic data prep function
def prep_data():
    df = import_data().pipe(preprocess_df)
    return df

# function to drop lat and long columns
def remove_lat_long(df):
    df_new = df.copy()
    df_new.drop(['lat', 'long'], axis=1, inplace=True)
    return df_new

# function to add date features and drop date column
def prep_data_with_date_feats():
    df = prep_data()
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_of_week + 1   # +1 --> Monday is 1 and not 0
    df = df.drop('date', axis=1)
    return df

def prep_data_long():
    df = import_data().pipe(preprocess_df).pipe(remove_lat_long)
    return df

# removes statistical and geographical outliers and drops redundant columns
def prep_data_ext():
    from outlier_handling import remove_outliers
    df_sans_outliers = remove_outliers(prep_data())
    df = df_sans_outliers.drop(['sqft_living15', 'sqft_lot15', 'sqft_above'], axis=1)
    return df