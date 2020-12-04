import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(characteristics_file_name, simulation_folder, scenario):
    """
    Function for spliting data into test and train set
    """
    project_directory = os.getcwd()
    path_to_characteristics = os.path.join(project_directory, "Data", "characteristics")
    file_with_characteristics = os.path.join(path_to_characteristics, characteristics_file_name)
    path_to_data = os.path.join(project_directory, "Data", "Synthetic data")
    path_to_save = os.path.join(path_to_data, simulation_folder)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    print(file_with_characteristics)
    characteristics_data = pd.read_csv(file_with_characteristics)

    try:
        characteristics_data = characteristics_data.drop(["diff_type", "file", "index", "level_0"], axis=1)
    except:
        characteristics_data = characteristics_data.drop(["diff_type", "file", "index"], axis=1)
    if "exp" in characteristics_data.columns:
        characteristics_data = characteristics_data.drop(["exp"], axis=1)
    print(characteristics_data.columns)
    X = characteristics_data.loc[:, characteristics_data.columns != 'motion']
    y = characteristics_data["motion"]
    y_for_split = characteristics_data["motion"].values
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y_for_split)
    np.save(os.path.join(path_to_save, "X_data.npy"), X)
    np.save(os.path.join(path_to_save, "y_data.npy"), y)
    np.save(os.path.join(path_to_save, "X_train.npy"), X_train)
    np.save(os.path.join(path_to_save, "X_test.npy"), X_test)
    np.save(os.path.join(path_to_save, "y_train.npy"), y_train)
    np.save(os.path.join(path_to_save, "y_test.npy"), y_test)


if __name__ == "__main__":
    #split_data(characteristics_file_name="characteristics_3D.csv", simulation_folder="Base_corr_Andi_3D",
    #           scenario="Andi")
    #split_data(characteristics_file_name="characteristics_1D.csv", simulation_folder="Base_corr_Andi_1D",
    #           scenario="Andi")
    split_data(characteristics_file_name="characteristics_2D.csv", simulation_folder="Base_corr_Andi_2D",
               scenario="Andi")
