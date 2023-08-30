from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from model_evaluation import execute_cross_validate, eval_on_set, get_split
from basic_data_prep import prep_data_long


# create custom transformer class for Kmeans clustering of 'lat' and 'long'
class KMeansTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=100)

    def fit(self, X, y=None):
        self.kmeans.fit(X[['lat', 'long']])
        return self

    def transform(self, X):
        X = X.copy()
        X['geo_cluster'] = self.kmeans.predict(X[['lat', 'long']])
        X.drop(['lat', 'long'], axis=1, inplace=True)
        return X

    
# function for getting df with desired numerical variables
def prep_numerical():
    numerical_df = prep_data_long().drop(['zipcode', 'waterfront', 'yr_renovated', 'price'],  axis=1)
    return numerical_df

    
# function for creating and testing pipeline
def test_pipeline(df,
                  cat_features, 
                  num_features, 
                  model= LinearRegression(), 
                  cluster=True,
                  n_clusters=250,
                  num_scaler=StandardScaler(),
                  test_eval=False):
    
    '''NOTE: due to geo_cluster column being created WITHIN the pipeline, the df entering the pipeline does not contain a geo_cluster column
    Howevever, list of categorical features which is to be one-hot encoded needs to include 'geo_cluster' 
    
    'lat' and 'long' are ALWAYS dropped either before entering pipeline as part of if-statement OR as part of first pipeline step in
    the KMeans-Transformer
    '''
    
    print('cat_features before if statement:', cat_features)
    # if we use Kmeans clusters, we make some adjustments to df and cat_features
    if cluster:
        if 'zipcode' in df:
            df.drop('zipcode', axis=1, inplace=True)
        if 'zipcode' in cat_features:
            cat_features.remove('zipcode')    
        if 'geo_cluster' not in cat_features:
            cat_features.append('geo_cluster')
    else:
        if 'lat' in df and 'long' in df:
            df.drop(['lat', 'long'], axis=1, inplace=True)
            
            
    # TRANSFORMERS      
    cat_transformer = ('cat', 
                       OneHotEncoder(handle_unknown='ignore'), 
                       cat_features)                           # cat_features may be modified by if-statement above
    
    num_transformer = ('num', 
                       num_scaler, 
                       num_features)
     
    preprocessor = ColumnTransformer(transformers= [cat_transformer, num_transformer], 
                                     remainder='passthrough')   # 'remainder' allows columns that are not explicitly mentioned to be passed through
    
    # PIPELINE STEPS - determined by arguments passed to function
    if cluster:
        steps=[('kmeans', KMeansTransformer(n_clusters=n_clusters)),
               ('preprocessor', preprocessor),
               ('model', model)]
    else:
        steps = [('preprocessor', preprocessor), 
                 ('model', model)]

    
    pipeline = Pipeline(steps=steps)
    
    # SPLIT DATA
    X = df.drop('price', axis = 1)
    print('columns of X:', list(X.columns))
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # keep in case we want to use validation set instead of cross-validate
#     train_size = 0.7
#     valid_size = 0.2      
#     test_size = 0.1
#     X_train, y_train, X_val, y_val, X_test, y_test = get_split(df, 
#                                                                train_size=train_size, 
#                                                                valid_size=valid_size, 
#                                                                test_size=test_size)

    print('shapes of X_train, X_test:', X_train.shape, X_test.shape)
    

    # CROSS-VALIDATE (only uses training data)
    cross_val_scores = execute_cross_validate(pipeline, X_train, y_train, folds=5)

    # EVALUATE ON VALIDATION SET - keep in case we want to use validation set instead of cross-validate
#     val_set_scores = eval_on_set(pipeline, X_train, y_train, X_val, y_val)  

    # EVALUATE ON TEST SET
    if test_eval==True:
        
        # keep in case we want to use validation set instead of cross-validate
        # X_train_full = pd.concat([X_train, X_val])
        # y_train_full = pd.concat([y_train, y_val])
        
        test_set_scores = eval_on_set(pipeline, X_train, y_train, X_test, y_test, set_type='test')    
        return cross_val_scores, test_set_scores, pipeline     # , val_set_scores
    else:
        return cross_val_scores, pipeline    # , val_set_scores