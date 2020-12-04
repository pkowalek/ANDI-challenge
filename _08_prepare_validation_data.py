import csv
import os

import numpy as np
import pandas as pd


def save_data(file):
    project_directory = os.getcwd()
    path_to_file = os.path.join(project_directory, "ValidationDatasets", file)
    trajs_from_files = csv.reader(open(path_to_file, 'r'), delimiter=';',
                                  lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)

    path_to_save = os.path.join(project_directory, "ValidationDatasets")

    validation = [[], [], []]
    for trajs in enumerate(trajs_from_files):
        validation[int(trajs[1][0]) - 1].append(trajs[1][1:])
    np.save(os.path.join(path_to_save, "task2.npy"), validation)


def prepare_data(characteristics_file_name, save_file_name, set_name):
    """
    Function for spliting data into test and train set
    """
    project_directory = os.getcwd()
    path_to_data = os.path.join(project_directory, "ValidationDatasets","Results", set_name)
    file_with_characteristics = os.path.join(path_to_data, characteristics_file_name)
    characteristics_data = pd.read_csv(file_with_characteristics)
    characteristics_data = characteristics_data.drop(["diff_type", "file", "index"], axis=1)
    if "exp" in characteristics_data.columns:
        characteristics_data = characteristics_data.drop(["exp"], axis=1)
    print(characteristics_data.columns)
    X = characteristics_data.loc[:, characteristics_data.columns != 'motion']

    np.save(os.path.join(path_to_data, save_file_name), X)


def marge_results(res_folder1, res_folder2, res_folder3):
    project_directory = os.getcwd()
    path_to_val = os.path.join(project_directory, "ValidationDatasets","Results")

    if res_folder1 is not None:
        t1 = np.loadtxt(os.path.join(path_to_val, res_folder1, "GB", "task2.txt"), delimiter=';')
        t1 = pd.DataFrame(t1, columns=["i", "x1", "x2", "x3", "x4", "x5"])
        t1["i2"] = 1
    else:
        t1 = pd.DataFrame([])
    if res_folder2 is not None:
        t2 = np.loadtxt(os.path.join(path_to_val, res_folder2, "GB", "task2.txt"), delimiter=';')
        t2 = pd.DataFrame(t2, columns=["i", "x1", "x2", "x3", "x4", "x5"])
        t2["i2"] = 2
    else:
        t2 = pd.DataFrame([])
    if res_folder3 is not None:
        t3 = np.loadtxt(os.path.join(path_to_val, res_folder3, "GB", "task2.txt"), delimiter=';')
        t3 = pd.DataFrame(t3, columns=["i", "x1", "x2", "x3", "x4", "x5"])
        t3["i2"] = 3
    else:
        t3 = pd.DataFrame([])

    df = pd.concat([t1, t2, t3])
    df = df[["i2", "x1", "x2", "x3", "x4", "x5"]]

    np.savetxt(os.path.join(path_to_val, "task2.txt"), df.values, delimiter=';')
