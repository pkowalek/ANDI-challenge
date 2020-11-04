import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def search_hyperparameters(simulation_folder):
    """
    Function for searching best hyperparameters for gradient boosting algorythm
    """

    Start = datetime.now()
    project_directory = os.getcwd()
    path_to_data = os.path.join(project_directory, "Data", "Synthetic data", simulation_folder)
    path_to_hyperparameters = os.path.join(project_directory, "Models", "GB", simulation_folder)
    if not os.path.exists(path_to_hyperparameters):
        os.makedirs(path_to_hyperparameters)
    X_train = np.load(os.path.join(path_to_data, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path_to_data, "y_train.npy"), allow_pickle=True)

    random_grid_forest = {'n_estimators': [int(x) for x in range(100, 1001, 100)],
                          'max_features': ['log2', 'sqrt', None],
                          'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 2, 4]
                          }

    gb = GradientBoostingClassifier()
    # Random search of parameters, using 10 fold cross validation,
    # search across 100 different combinations, and use all available cores
    gb_random = RandomizedSearchCV(estimator=gb, param_distributions=random_grid_forest, n_iter=100, cv=10,
                                   verbose=2, random_state=42, n_jobs=-1)
    # Fit the random search model
    gb_random.fit(X_train, y_train)

    print(gb_random.best_params_)
    with open(os.path.join(path_to_hyperparameters, "hyperparameters.json"), 'w') as fp:
        json.dump(gb_random.best_params_, fp)
    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_hyperparameters, "time_for_searching.csv"))
    print(ExecutedTime)


if __name__ == "__main__":
    search_hyperparameters(simulation_folder="Base_corr_Andi_1D")
    search_hyperparameters(simulation_folder="Base_corr_Andi_2D")
    search_hyperparameters(simulation_folder="Base_corr_Andi_3D")
