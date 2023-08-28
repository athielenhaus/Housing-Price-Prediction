import pandas as pd
import numpy as np


# function to convert year built to age (keeping in mind that the dataset is from 2015)
def convert_yr_built_to_age(df):
    df_age = df.copy()
    df_age['property_age'] = 2015 - df_age['yr_built']
    df_age.drop('yr_built', axis=1, inplace=True)
    return df_age


# function to bin 'yr_built'
def bin_yr_built(df):
    conditions = [
#         (df['yr_built'] <= 1910),
#         (df['yr_built'] > 1910) & (df['yr_built'] <= 1920),
#         (df['yr_built'] > 1920) & (df['yr_built'] <= 1930),
#         (df['yr_built'] > 1930) & (df['yr_built'] <= 1940),
        (df['yr_built'] <= 1940),
        (df['yr_built'] > 1940) & (df['yr_built'] <= 1950),
        (df['yr_built'] > 1950) & (df['yr_built'] <= 1960),
        (df['yr_built'] > 1960) & (df['yr_built'] <= 1970),
        (df['yr_built'] > 1970) & (df['yr_built'] <= 1980),
        (df['yr_built'] > 1980) & (df['yr_built'] <= 1990),
        (df['yr_built'] > 1990) & (df['yr_built'] <= 2000),
        (df['yr_built'] > 2000) & (df['yr_built'] <= 2010),
        (df['yr_built'] > 2010)
        ]

    # values to assign for each condition
    values = ['pre_40', 'post_40','post_50','post_60', 'post_70','post_80','post_90', 'post_2000', 'post_2010']
#     values = ['pre_10', 'post_10','post_20','post_30', 'post_40','post_50','post_60', 'post_70','post_80','post_90', 'post_2000', 'post_2010']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['yr_built_bin'] = np.select(conditions, values)
    df.drop('yr_built', axis = 1, inplace=True)
    return df


# function to bin 'yr_renovated'
def bin_yr_renovated(df):
    conditions = [
#         (df['yr_renovated'] <= 1970),
#         (df['yr_renovated'] > 1970) & (df['yr_renovated'] <=2000),
#         (df['yr_renovated'] > 2000) & (df['yr_renovated'] <= 2010),
#         (df['yr_renovated'] > 2010)
        
        (df['yr_renovated'] == 0),
        (df['yr_renovated'] > 0) & (df['yr_renovated'] <=1970),
        (df['yr_renovated'] > 1970) & (df['yr_renovated'] <=1980),
        (df['yr_renovated'] > 1980) & (df['yr_renovated'] <=1990),
        (df['yr_renovated'] > 1990) & (df['yr_renovated'] <=2000),
        (df['yr_renovated'] > 2000) & (df['yr_renovated'] <= 2010),
        (df['yr_renovated'] > 2010)
        ]

    # create a list of the values we want to assign for each condition
    values = ['no renovation', 'pre_1970', 'post_70', 'post_80', 'post_90', 'post_2000','post_2010']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['renovation_cat'] = np.select(conditions, values)
    df.drop('yr_renovated', axis = 1, inplace=True)
    
    return df


# function to combine yr_built and yr_renovated bins into new categorical variable
def combine_yr_built_renov(df):
    df_new = df.copy()
    df_bins = df_new.pipe(bin_yr_built).pipe(bin_yr_renovated)
    df_bins['built_renov_bin'] = 'yr built bin: '+ df_bins['yr_built_bin'] + " renovation category:" + df_bins['renovation_cat']
    df_bins.drop(['yr_built_bin', 'renovation_cat'], axis=1, inplace=True)
    return df_bins