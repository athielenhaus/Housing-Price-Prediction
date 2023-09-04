#### PIPELINE SCRIPT - GRIDSEARCH AND TRAINING


# Setup GridSearch

# define categorical and numerical features
cat_features = ['geo_cluster', 'yr_built_bin', 'renovation_cat', 'grade', 'bedrooms', 'bathrooms','view', 'floors']
num_features = [ 'condition', 'sqft_basement','sqft_living','sqft_lot']

# Setup transformers
cat_transformer = ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
num_transformer = ('num', MinMaxScaler(), num_features)
           
# Create a preprocessor
preprocessor = ColumnTransformer(transformers= [cat_transformer, 
                                                num_transformer], 
                                 remainder='passthrough')

# Setup pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('model', Lasso())])

# Gridsearch params
param_grid = [
#     {'preprocessor__num': [None, StandardScaler(), MinMaxScaler(), RobustScaler()],
#      'model': [LinearRegression()]},
    {
#     'preprocessor__num': [None, StandardScaler(), MinMaxScaler(), RobustScaler()],
#     'model': [Lasso()],
    'model__alpha': [7, 7.5, 8, 8.5, 9]
    },
#     {
#     'preprocessor__num': [None, StandardScaler(), MinMaxScaler(), RobustScaler()],
#     'model': [ElasticNet()],
#     'model__alpha': [0.01, 0.1, 1, 2, 5, 10],
#     'model__l1_ratio': [0.1, 0.5, 0.9]
#     },
]

# Set up grid search 
grid_search = GridSearchCV(pipeline, param_grid, scoring='r2', cv=5)

# Fit model
df = prep_data_ext_bins()

train_size = 0.7
valid_size = 0.2
test_size = 1-train_size-valid_size

X_train, y_train, X_val, y_val, X_test, y_test = get_split(df, train_size=train_size, valid_size=valid_size, test_size=test_size)

grid_search.fit(X_train, y_train)

# Get best model, params, score
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best params: {best_params}\nBest score: {best_score}\nBest model: {best_model}")