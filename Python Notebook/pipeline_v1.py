from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


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
                  model= LinearRegression(), 
                  cat_features=['zipcode'], 
                  num_features=prep_numerical().columns.to_list(),   # should maybe not be passed as default
                  scale=True,
                  cluster=True,
                  n_clusters=250,
                  preprocess=True,
                  num_scaler=StandardScaler(),
                  test_eval=False):
    
   
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
    cat_transformer = ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features) # cat_features affected by Kmeans clusters
    transformers = [cat_transformer]
    
    if scale:                                                 # 'if' allows us to test the impact of scaling
        num_transformer = ('num', num_scaler, num_features)
        transformers.append(num_transformer)    
    preprocessor = ColumnTransformer(transformers= transformers, remainder='passthrough')   # 'remainder' allows columns that are not explicitly mentioned to be passed through
    
    # PIPELINE STEPS - determined by arguments passed to function
    if preprocess and cluster:
        steps=[('kmeans', KMeansTransformer(n_clusters=n_clusters)),
               ('preprocessor', preprocessor),
               ('model', model)]
    elif preprocess:
        steps = [('preprocessor', preprocessor), 
                 ('model', model)]
    else:
        steps = [('model', model)]
    
    pipeline = Pipeline(steps=steps)

    # SPLIT DATA
    train_size = 0.7
    valid_size = 0.2
    test_size = 1-train_size-valid_size
    X_train, y_train, X_val, y_val, X_test, y_test = get_split(df, train_size=train_size, 
                                                               valid_size=valid_size, test_size=test_size)
    
    # CROSS-VALIDATE (only uses training data)
    cross_val_scores = execute_cross_validate(pipeline, X_train, y_train, folds=5)

    # EVALUATE ON VALIDATION SET
    val_set_scores = eval_on_set(pipeline, X_train, y_train, X_val, y_val)  

    # EVALUATE ON TEST SET
    if test_eval==True:
        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])
        test_set_scores = eval_on_set(pipeline, X_train_full, y_train_full, X_test, y_test, set_type='test')    
        return cross_val_scores, val_set_scores, test_set_scores, pipeline
    else:
        return cross_val_scores, val_set_scores, pipeline