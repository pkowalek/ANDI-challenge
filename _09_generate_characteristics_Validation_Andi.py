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


def get_characteristics_single(X, num, dim):
    """
    :param initial_data: dataframe, info about all generated data
    :param path_to_trajectories: str, path to folder with trajectories
    :param trajectory: str, trajectory name
    :return: dataframe with characteristics
    """
    print(num)
    d = get_characteristics(dim, X, typ="", motion="", file=num)
    return d


def generate_characteristics(characteristics_filename, set_name, dim):
    """
    Function for generating the characteristics file for given scenario
    - characteristics are needed for featured based classifiers
    """

    project_directory = os.getcwd()
    path_to_save = os.path.join(project_directory, "ValidationDatasets", "Results")
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    path_to_data = os.path.join(project_directory, "ValidationDatasets")
    X = np.load(os.path.join(path_to_data, "task2.npy"))
    if dim == 1:
        X = X[0]
    elif dim == 2:
        X = X[1]
    else:
        X = X[2]
    characteristics_input = zip(X, range(len(X)), repeat(dim))
    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    characteristics_data = pool.starmap(get_characteristics_single, characteristics_input)
    pool.close()
    pool.join()
    results = pd.concat(characteristics_data)
    results = results.replace([np.inf, -np.inf], np.nan)
    results = results.dropna()
    results.reset_index(inplace=True)
    if not os.path.exists(os.path.join(path_to_save, set_name)):
        os.makedirs(os.path.join(path_to_save, set_name))
    results.to_csv(os.path.join(path_to_save, set_name, characteristics_filename), index=False)

    # np.load = np_load_old
