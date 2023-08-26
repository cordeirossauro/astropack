from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

def create_model(model_type, hp_combination):
    if model_type == 'RF':
        n_features, n_trees, min_samples_leaf, bootstrap, max_features = hp_combination
        
        # Initialize the feature selector using the given hyperparameters
        FeatureSelector = RFE(estimator = DecisionTreeRegressor(), 
                              n_features_to_select = n_features, 
                              verbose = 0, step = 100)

        # Initialize the random forest using the given hyperparameters
        RF = RandomForestRegressor(n_estimators = n_trees, 
                                   bootstrap = bootstrap, 
                                   min_samples_leaf = min_samples_leaf,
                                   max_features = max_features)

        # Create a pipeline with the feature selector and the random forest
        pipeline = Pipeline(steps = [('Feature Selector', FeatureSelector),('Model', RF)])
    else:
        raise('Modelo não suportado...')
    
    return pipeline