from fast_ml.model_development import train_valid_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression


# function to get split
def get_split(df,
              train_size = 0.7,
              valid_size = 0.2,
              test_size = 0.1):
    
    X_train, y_train, X_val, y_val, X_test, y_test = train_valid_test_split(df, 
                                                                            target = 'price', 
                                                                            train_size=train_size, 
                                                                            valid_size=valid_size, 
                                                                            test_size=test_size,  
                                                                            random_state=100)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# create custom scorer for adjusted r-square
def adjusted_r2_scorer(y_true, y_pred, estimator, X):
    r2 = r2_score(y_true, y_pred)
    n = X.shape[0]
    p = X.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2



# function for cross-validation which returns mean scores across folds
def execute_cross_validate(model_or_pipeline, X_train, y_train, folds=5):

    adjusted_r2 = make_scorer(adjusted_r2_scorer, greater_is_better=True, estimator=model_or_pipeline, X=X_train) # in case estimator causes problems, switch back to LinearRegression()

    scoring = {                                  # specify the metrics
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
        'adjusted_r2': adjusted_r2
    } 
    scores = cross_validate(model_or_pipeline, X_train, y_train, cv=folds, scoring=scoring, error_score='raise')
    mean_mae = round(-1 * scores['test_mae'].mean(), 3)
    mean_rmse = round(-1 * scores['test_rmse'].mean(), 3)
    mean_rsquare = round(scores['test_r2'].mean(), 3)
    mean_adj_rsquare = round(scores['test_adjusted_r2'].mean(), 3)
    mean_scores = {'mean_mae': mean_mae, 'mean_rmse': mean_rmse, 'mean_rsquare':mean_rsquare, 'mean_adj_rsquare':mean_adj_rsquare}
    print('5-fold Cross Validation using only training data:\n', mean_scores)
    
    return {'cross_validate_scores': mean_scores}


# function for calculating several indicators of model performance
def calc_scores(y, y_pred, X):
    mae =  round(mean_absolute_error(y, y_pred), 3)
    rmse =  round(mean_squared_error(y, y_pred, squared=False), 3)
    r2 = round(r2_score(y, y_pred), 3)
    n = X.shape[0]
    p = X.shape[1]
    adj_r2 = round((1 - (1 - r2) * (n - 1) / (n - p - 1)), 3)                    # custom calculation for adjusted R2
    scores = {'mae': mae, 'rmse': rmse, 'rsquare': r2, 'adj_rsquare': adj_r2}
#     print(f'score: {scores}')
    return scores


# function for evaluating a model or pipeline on a test set
def eval_on_set(model_or_pipeline, X_train, y_train, X_test, y_test, set_type='test'):   # takes instance of a model or pipeline as argument
    model_or_pipeline.fit(X_train, y_train)
    y_test_pred = model_or_pipeline.predict(X_test)
    print(f'Performance on {set_type} set:')
    scores = calc_scores(y_test, y_test_pred, X_test) 
    print(scores)
    return {f'{set_type} set scores': scores}


# combines cross validation with evaluation - POSSIBLY REDUNDANT - MAYBE DELETE!!!
def cross_val_and_eval(model_or_pipeline, X_train, y_train, X_test, y_test, set_type='validation', folds=5):
    cross_val_scores = execute_cross_validate(model_or_pipeline, X_train, y_train, folds=5)
    set_scores = eval_on_set(model_or_pipeline, X_train, y_train, X_test, y_test, set_type)
    return cross_val_scores, set_scores