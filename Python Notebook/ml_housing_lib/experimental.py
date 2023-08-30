from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

import itertools

from outlier_handling import remove_outliers
from pipeline import test_pipeline
from basic_data_prep import prep_data



# function to add geo-clusters with k-means to df
def add_kmeans_clusters(df, clusters=500, drop_cols=True):
    df_geo = df.copy()
    kmeans = KMeans(clusters, random_state=100)
    clusters = kmeans.fit_predict(df_geo[['lat','long']])
    df_geo['geo_cluster'] = kmeans.predict(df_geo[['lat','long']])
    if drop_cols:
        df_geo.drop(['zipcode', 'lat', 'long'], axis=1, inplace=True)

    return df_geo


# create function to test different numbers of clusters:
def test_number_of_clusters(models=[LinearRegression()], 
                            start_point=10, 
                            end_point = 20, 
                            increment = 5):
    
    for num in range(start_point, end_point+1, increment):

        num_features = ['bedrooms', 'bathrooms','sqft_living','sqft_lot','floors','view',
                'condition','grade',
                'sqft_above','sqft_basement','yr_built','sqft_living15','sqft_lot15']

        # in this test we encode zipcode
        cat_features = ['geo_cluster']
        
        df = remove_outliers(prep_data())
        
        
        print('Nr. of clusters: ', num)
        test_pipeline(df=df, scale=True, cluster=True, test_eval=False, 
                      cat_features= cat_features, 
                      num_features= num_features,
                      n_clusters = num)
        
        
# function for generating all possible combinations of items in a list
def get_combinations(options):
    combinations= []
    for r in range(1, len(options) + 1):  # starting at 1 rather than 0 avoids creating an 'empty' subset
        for subset in itertools.combinations(options, r):
            combinations.append(list(subset))
    return combinations


# function to add test results to dict
def add_results(data, results, cat_subset):
    data['categorical_features'].append(cat_subset)
#     data['model'].append(list(results.keys()))  # if key does not exist, append 'N/A'
    data['cv_mean_mae'].append(results[0]['cross_validate_scores'].get('mean_mae', float('nan')))  # if key does not exist, append NaN
    data['cv_mean_rmse'].append(results[0]['cross_validate_scores'].get('mean_rmse', float('nan')))  # if key does not exist, append NaN
    data['cv_mean_rsquare'].append(results[0]['cross_validate_scores'].get('mean_rsquare', float('nan')))  # if key does not exist, append NaN
    data['cv_mean_adj_rsquare'].append(results[0]['cross_validate_scores'].get('mean_adj_rsquare', float('nan')))  # if key does not exist, append NaN

    return data


# main function for testing different combinations
def test_combinations(opt_features_list, fixed_features_list, model=LinearRegression()):
    
    data = {
        'categorical_features': [],
        'cv_mean_mae': [], 
        'cv_mean_rmse': [],
        'cv_mean_rsquare': [],
        'cv_mean_adj_rsquare': [],
    }
    
    categorical_combinations = get_combinations(opt_features_list)
        
    # loop through categorical feature combinations
    for cat_subset in categorical_combinations:
        
        categorical_features = cat_subset + fixed_features_list   # we add optional features to fixed features
        print(categorical_features)
        
        results = test_pipeline(df=prep_data_ext_bins(), model=model, cluster=True, n_clusters=500, scale=False, cat_features=categorical_features)
        
        # FIT MODEL AND GET RESULTS
        data = add_results(data, results, categorical_features)
    
    data_df = pd.DataFrame.from_dict(data)
    return data_df



# check feature importance
def get_perm_importance(df):
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    model = LinearRegression().fit(X_train, y_train)
    feature_names= X_train.columns.tolist()

    r = permutation_importance(model, X, y,
                               n_repeats=30,
                               random_state=0)
    
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]:<8} "
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")


