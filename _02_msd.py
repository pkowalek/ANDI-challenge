from math import log

import numpy as np


def generate_theoretical_msd_normal(n_list, D, dt):
    """
    Function for generating msd of normal diffusion
    :param n_list: number of points in msd
    :param D: float, diffusion coefficient
    :param dt: float, time between steps
    :return: array of theoretical msd
    """
    r = 4 * D * dt * n_list
    return r


def generate_theoretical_msd_anomalous(n_list, D, dt, alpha):
    """
    Function for generating msd of anomalous diffusion
    :param n_list: number of points in msd
    :param D: float, diffusion coefficient
    :param dt: float, time between steps
    :param alpha: float, anomalous exponent (alpha<1)
    :return: array of theoretical msd
    """
    r = 4 * D * (dt * n_list) ** alpha
    return r


def generate_theoretical_msd_anomalous_log(log_dt_n_list, log_D, alpha):
    """
    Function for generating msd of anomalous diffusion
    :param log_dt_n_list: logarithm of points in msd times dt
    :param log_D: float, logarithm of diffusion coefficient
    :param alpha: float, anomalous exponent (alpha<1)
    :return: array of theoretical msd
    """
    r = log(4) + log_D + alpha * log_dt_n_list
    return r


def generate_empirical_msd(data, dim, n_list, k=2):
    r = []
    for n in n_list:
        r.append(empirical_msd(data, dim, n, k))
    return np.array(r)


def empirical_msd(data, dim, n, k):
    """
    Function for generating empirical msd, where N is number of positions
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param n: int, point of msd
    :param k: int, power of msd
    :return: point of empirical msd for given point
    """
    if dim == 1:
        x = data
        N = len(x)
        x1 = np.array(x[:N - n])
        x2 = np.array(x[n:N])

        c = np.sqrt(np.array(list(x2 - x1)) ** 2) ** k
        r = np.mean(c)
    elif dim == 2:
        l = int(len(data) / dim)
        x = data[:l]
        y = data[l:]
        N = len(x)
        x1 = np.array(x[:N - n])
        x2 = np.array(x[n:N])
        y1 = np.array(y[:N - n])
        y2 = np.array(y[n:N])
        c = np.sqrt(np.array(list(x2 - x1)) ** 2 + np.array(list(y2 - y1)) ** 2) ** k
        r = np.mean(c)
    elif dim == 3:
        l = int(len(data) / dim)
        x = data[:l]
        y = data[l:2 * l]
        z = data[2 * l:]
        N = len(x)
        x1 = np.array(x[:N - n])
        x2 = np.array(x[n:N])
        y1 = np.array(y[:N - n])
        y2 = np.array(y[n:N])
        z1 = np.array(z[:N - n])
        z2 = np.array(z[n:N])
        c = np.sqrt(np.array(list(x2 - x1)) ** 2 + np.array(list(y2 - y1)) ** 2 + np.array(list(z2 - z1)) ** 2) ** k
        r = np.mean(c)
    return r


def generate_empirical_pvariation(data, p_list=[2], m_list=[1]):
    """
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param n_list: number of points in msd
    :param p_list: powers of p-variation, default p=[2] - quadratic variation
    :param m_list: the choice of lags, default m=[1] - simple differences
    :return: array of empirical pvariation
    """
    l = int(len(data) / 2)
    x = data[:l]
    y = data[l:]
    N = len(x)

    pvar = np.zeros((len(p_list), len(m_list)))
    for p_index in range(len(p_list)):
        for m_index in range(len(m_list)):
            p = p_list[p_index]
            m = m_list[m_index]
            sample_indexes = np.arange(0, N - m, m)
            x_diff = np.take(x, sample_indexes + m) - np.take(x, sample_indexes)
            y_diff = np.take(y, sample_indexes + m) - np.take(y, sample_indexes)
            pvar[p_index][m_index] = sum(np.sqrt(x_diff ** 2 + y_diff ** 2) ** p)
    return pvar


def generate_empirical_velocity_autocorrelation(x,y, n_list, dt, delta=1):
    """
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param n_list: list, number of points in autocorrelation
    :param dt: float, time between steps
    :param delta: the time lag between observations (default=1)
    :return: array of empirical autocorrelation
    """
    r = []
    for n in n_list:
        r.append(empirical_velocity_autocorrelation(x, y, n, dt, delta))
    return np.array(r)


def empirical_velocity_autocorrelation(x, y, n, dt, delta):
    """
    Function for generating empirical autocorrelation, where N is number of positions
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param n: int, point of autocorrelation
    :param dt: float, time between steps
    :param delta: the time lag between observations (default=1)
    :return: point of empirical msd for given point
    """

    velocities_x = np.diff(x, delta) / (delta * dt)
    velocities_y = np.diff(y, delta) / (delta * dt)
    N = len(velocities_x)

    vx1 = np.array(velocities_x[:N - n])
    vx2 = np.array(velocities_x[n:])
    vy1 = np.array(velocities_y[:N - n])
    vy2 = np.array(velocities_y[n:])

    c = vx2 * vx1 + vy2 * vy1
    r = np.mean(c)

    return r
