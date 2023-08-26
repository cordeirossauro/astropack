from sklearn.model_selection import RepeatedKFold


import time
import numpy as np
import pandas as pd
from astropack.models import (create_model) 

class Tuner:
    def __init__(self,
                 model_type,
                 n_splits,
                 n_repeats,
                 tuning_id,
                 mag_type,
                 param_name):
        
        self.model_type = model_type
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.tuning_id = tuning_id
        self.mag_type = mag_type
        self.param_name = param_name
    
    def evaluate_combination(self, hp_combination):
        '''
        Evaluate a STEPE-RF model (Feature Selector + Random Forest) initialized with a
        certain combination of hyperparameters by using a k-fold n-repeat cross validation 
        and return the average values and standard deviations of a set of 
        metrics (MAE, RMSE, MaxError, R2 Score and elapsed time).

        Keyword arguments:
        Hyperparams - A list containing the combination of hyperpameters to test, in the 
                    format [n_features, n_trees, min_samples_leaf, max_features, criterion]

        X - Dataframe containing the input values of the development sample 
            (will be split into training and validation samples)

        y - Dataframe containing the target values of the development sample 
            (will be split into training and validation samples)

        n_splits - Number of samples that the data will be split into during the k-fold, 
                   n-repeat cross-validation (corresponds to k)

        n_repeats - Number of times that the data will be shuffled and split again during 
                    the k-fold, n-repeat cross-validation (corresponds to n)

        verbose - Indicates whether the function will print information on the screen or not
        '''
        times = []
        mads = []

        n_features, n_trees, min_samples_leaf, bootstrap, max_features = hp_combination

        filename = f'results/rf_tuning_{self.mag_type}_{self.param_name}_id{self.tuning_id}.csv'

        x_dev = pd.read_csv(f'temp_dataframes/x_dev_{self.mag_type}_{self.tuning_id}.csv', index_col = 0)
        y_dev = pd.read_csv(f'temp_dataframes/y_dev_{self.mag_type}_{self.tuning_id}.csv', index_col = 0)[self.param_name]

        # Initialize the cross-validation function
        kf_splitter = RepeatedKFold(n_splits = self.n_splits, n_repeats = self.n_repeats)

        # Loop through all the training/validation combinations given by the cross-validation function
        for train_index, validation_index in kf_splitter.split(x_dev):
            # Get the training and validation samples
            x_train, x_validation = x_dev.iloc[train_index], x_dev.iloc[validation_index]
            y_train, y_validation = y_dev.iloc[train_index], y_dev.iloc[validation_index]

            # Initialize the model
            pipeline = create_model(self.model_type,
                                    hp_combination)
            # Start a timer to time the process
            start_time = time.time()

            # Fit the pipeline to the training data
            pipeline = pipeline.fit(x_train, y_train.values.reshape(len(y_train)))

            # Predict the target values of the validation sample
            predictions = pipeline.predict(x_validation)
            # Stop the timer
            end_time = time.time()

            # Calculate the median and mad for the errors of the model
            errors = y_validation - predictions

            mad = np.median(np.abs(errors))

            times.append(end_time - start_time)
            mads.append(mad)

        results = pd.DataFrame()
        results[['n_features', 'n_trees', 'min_samples_leaf', 'bootstrap', 'max_features']] = [hp_combination]

        results['mad'] = np.array(mads).mean()    
        results['time'] = np.array(times).mean()

        results.to_csv(filename, index = False, header = False, mode = 'a')
        
        return 'Success!'