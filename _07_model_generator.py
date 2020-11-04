import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

np.set_printoptions(precision=2)
rs = 42


def generate_model(simulation_folder, featured_model, test_version=""):
    """
    Function for generating model for given scenario and feature based model
    :param featured_model: "RF", "GB" or other id defined
    :return: model
    """
    print(simulation_folder, test_version, featured_model)
    Start = datetime.now()
    project_directory = os.getcwd()
    path_to_data = os.path.join(project_directory, "Data", "Synthetic data", simulation_folder)
    path_to_model = os.path.join(project_directory, "Models", featured_model, simulation_folder)
    path_to_hyperparameters = os.path.join(path_to_model, "hyperparameters.json")
    X_train = np.load(os.path.join(path_to_data, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path_to_data, "y_train.npy"), allow_pickle=True)

    with open(path_to_hyperparameters, 'r') as f:
        param_data = json.load(f)
    if featured_model == "RF":
        model = RandomForestClassifier()
    elif featured_model == "GB":
        model = GradientBoostingClassifier()
    model.set_params(**param_data)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(path_to_model, 'model.sav'))
    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_model, "time_for_modelling.csv"))
    print(ExecutedTime)


if __name__ == "__main__":
    generate_model(simulation_folder="Base_corr_Andi_1D", featured_model="GB")
    generate_model(simulation_folder="Base_corr_Andi_2D", featured_model="GB")
    generate_model(simulation_folder="Base_corr_Andi_3D", featured_model="GB")

