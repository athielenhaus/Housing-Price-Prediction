'''you’d first have a pipeline for ground truth creation (this is getting the data, cleaning the outliers, 
doing the data splits, and having in the end the kind of data that you’d like to use in train)  
'''
import numpy as np
import pandas as pd
from collections import Counter
from basic_data_prep import prep_data, prep_data_long


# function for detecting number of outliers
def detect_outlier(df, feature):
    outliers = []
    data = df[feature]
    mean = np.mean(data)
    std = np.std(data)
    
    for y in data:
        z_score = (y-mean)/std
        if np.abs(z_score)>3:
            outliers.append(y)
    
    quantile_95 = data.quantile(.95)
    quantile_99 = data.quantile(.99)
    above_3_stds = mean + 3 * std
    print(f'Outlier caps for {feature}')
    print(f'Cap 95th percentile: {quantile_95}  Nr of data points exceeding cap: {len([i for i in data if i > quantile_95 ])}')
    print(f'Cap 99th percentile: {quantile_99}  Nr of data points exceeding cap: {len([i for i in data if i > quantile_99 ])}')
    print(f'Cap 3 standard deviations above mean: {above_3_stds}  Nr of data points exceeding cap: {round(len([i for i in data if i > above_3_stds]), 2)}\n')


# get overview of number of outliers per feature
# feats = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'price']
# for f in feats:
#     detect_outlier(prep_data(), f)  


# function for identifying outliers above 99th percentile
def get_outliers(df, 
                columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
                     ):
    outlier_indices = []
    
    for col in columns:
        data = df[col]
        quantile_99 = data.quantile(.99)
        outlier_list_col = df[df[col] > quantile_99].index
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 0 )
    
    return multiple_outliers


# function for removing statistical outliers
def remove_stat_outliers(df, features_list):
    print('Nr. of samples:', len(df))
    outlier_indices = get_outliers(df, features_list)
    df_sans_outliers = df.drop(outlier_indices, axis = 0).reset_index(drop=True)
    print('Nr. of samples after statistical outlier removal:', len(df_sans_outliers))
    return df_sans_outliers
    
# feats = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
# df = prep_data()
# df_sans_outliers = remove_stat_outliers(df, feats)


# function to remove geo-outliers
def remove_geo_outliers(df):
    df_geo = df.copy()
    df_geo.drop(df_geo[df.long > -121.7].index, inplace=True)
    df_geo.reset_index(drop=True, inplace=True)
    print('Nr. of samples after geo-outlier removal:', len(df_geo))
    return df_geo


# function to remove both stat and geo-outliers
def remove_outliers(df,
                   features= ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
                   ):
    df_sans_outliers = df.copy().pipe(remove_stat_outliers, features).pipe(remove_geo_outliers)
    return df_sans_outliers

df = prep_data()
print(df.head())
# df_sans_outliers= remove_outliers(df)