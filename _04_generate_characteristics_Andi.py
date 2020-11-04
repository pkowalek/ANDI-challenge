import multiprocessing as mp
import os
from itertools import repeat

import numpy as np
import pandas as pd

from _03_characteristics_1D_3D import Characteristic
from _03_characteristics_2D import Characteristic as Characteristic2

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def get_characteristics(dim, data, dt=1, typ="", motion="", file=""):
    if dim != 2:
        ch = Characteristic(data=data, dim=dim, dt=dt, percentage_max_n=1, typ=typ, motion=motion, file=file)
    else:
        ch = Characteristic2(data=data, dt=dt, percentage_max_n=1, typ=typ, motion=motion, file=file)

    data = ch.data
    return data


def get_characteristics_single(X, Y, num, dim):
    """
    :param initial_data: dataframe, info about all generated data
    :param path_to_trajectories: str, path to folder with trajectories
    :param trajectory: str, trajectory name
    :return: dataframe with characteristics
    """
    print(num)
    d = get_characteristics(dim, X, typ="", motion=Y, file=num)
    return d


def generate_characteristics(characteristics_filename, dim, dataset):
    """
    Function for generating the characteristics file for given scenario
    - characteristics are needed for featured based classifiers
    """

    project_directory = os.getcwd()
    path_to_save = os.path.join(project_directory,"Data", "characteristics")
    path_to_data = os.path.join(project_directory, "Data","datasets", dataset)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    X = np.load(os.path.join(path_to_data, "X.npy"))
    Y = np.load(os.path.join(path_to_data, "Y.npy"))
    characteristics_input = zip(X, Y, range(len(X)), repeat(dim))
    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    characteristics_data = pool.starmap(get_characteristics_single, characteristics_input)
    pool.close()
    pool.join()
    results = pd.concat(characteristics_data)
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.dropna()
    results.reset_index(inplace=True)
    results.to_csv(os.path.join(path_to_save, characteristics_filename), index=False)


if __name__ == "__main__":
    generate_characteristics(characteristics_filename="characteristics_1D.csv", dim=1, dataset="dataset_Andi_1")
    generate_characteristics(characteristics_filename="characteristics_3D.csv", dim=3, dataset="dataset_Andi_3")
    generate_characteristics(characteristics_filename="characteristics_2D.csv", dim=2, dataset="dataset_Andi_2")

    np.load = np_load_old
