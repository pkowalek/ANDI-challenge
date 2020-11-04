import math

import numpy as np
import pandas as pd
from numpy import log, mean, sqrt, where, std, exp
from scipy import linalg as LA
from scipy.optimize import curve_fit
from scipy.stats import kurtosis

from _02_msd import generate_theoretical_msd_normal, generate_theoretical_msd_anomalous_log, generate_empirical_msd


class Characteristic:
    """
    Class representing base characteristics of given trajectory
    """

    def __init__(self, data, dim, dt, percentage_max_n, typ="", motion="", file="", exp=None):
        """

        :param dt: float, time between steps
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        """
        self.data = data
        self.dim = dim
        self.dt = dt
        self.percentage_max_n = percentage_max_n
        self.type = typ
        self.motion = motion
        self.file = file
        self.exp = exp

        self.N = self.get_length_of_trajectory()
        self.T = self.get_duration_of_trajectory()
        self.max_number_of_points_in_msd = self.get_max_number_of_points_in_msd()
        self.n_list = self.get_range_for_msd()
        self.empirical_msd = generate_empirical_msd(self.data, self.dim, self.n_list)
        self.displacements = self.get_displacements()
        self.d = self.get_max_displacement()
        self.L = self.get_total_length_of_path()
        self.D = self.get_diffusion_coef()
        self.alpha = self.get_exponent_alpha()
        self.radius_gyration_tensor = self.get_tensor()
        self.eigenvalues, self.eigenvectors = LA.eig(self.radius_gyration_tensor)

        self.asymmetry = self.get_asymmetry()
        self.efficiency = self.get_efficiency()
        self.trappedness = self.get_trappedness()
        self.diff_kurtosis = self.get_kurtosis_corrected()
        self.fractal_dimension = self.get_fractal_dimension()
        self.gaussianity = self.get_gaussianity()
        self.mean_gaussianity = self.get_mean_gaussianity()
        self.spec_gaussianity = self.get_point_of_gaussianity()
        self.mean_squared_displacement_ratio = self.get_mean_squared_displacement_ratio()
        self.straightness = self.get_straightness()
        if self.dim == 1:
            self.values = [self.file, self.type, self.motion, self.D, self.alpha, self.asymmetry, self.efficiency,
                           self.fractal_dimension, self.mean_gaussianity,
                           self.mean_squared_displacement_ratio, self.straightness, self.trappedness]
            self.columns = ["file", "diff_type", "motion", "D", "alpha", "asymmetry", "efficiency", "fractal_dimension",
                            "mean_gaussianity", "mean_squared_displacement_ratio", "straightness",
                            "trappedness"]
        else:
            self.values = [self.file, self.type, self.motion, self.D, self.alpha, self.efficiency,
                           self.fractal_dimension, self.mean_gaussianity, self.diff_kurtosis,
                           self.mean_squared_displacement_ratio, self.straightness, self.trappedness]
            self.columns = ["file", "diff_type", "motion", "D", "alpha", "efficiency", "fractal_dimension",
                            "mean_gaussianity", "diff_kurtosis", "mean_squared_displacement_ratio", "straightness",
                            "trappedness"]
        self.data = pd.DataFrame([self.values], columns=self.columns)

    def get_length_of_trajectory(self):
        """
        :return: int, length of trajectory represented by N parameter
        """
        return int(len(self.data) / self.dim)

    def get_duration_of_trajectory(self):
        """
        :return: int, duration of the trajectory life represented by T parameter
        """
        return int((self.N - 1) * self.dt)

    def get_max_number_of_points_in_msd(self):
        """
        :return: int, maximal number which can be used to generate msd
        """
        if self.percentage_max_n != None:

            pm = math.floor(self.percentage_max_n * self.N)
            if pm < 5:
                return 5
            else:
                return pm
        else:
            return self.N if self.N <= 100 else 101

    def get_range_for_msd(self):
        """
        :return: array, range of steps in msd function
        """
        return np.array(range(1, self.max_number_of_points_in_msd))

    def get_displacements(self):
        """
        :return: array, list of displacements between x and y coordinates
        """
        if self.dim == 1:
            x = self.data
            return np.array(
                [self.get_displacement_1(x[i], x[i - 1]) for i in range(1, self.N - 1)])
        elif self.dim == 3:
            l = int(len(self.data) / self.dim)
            x = self.data[:l]
            y = self.data[l:2 * l]
            z = self.data[2 * l:]
            return np.array(
                [self.get_displacement_3(x[i], y[i], z[i], x[i - 1], y[i - 1], z[i - 1])
                 for i in
                 range(1, self.N - 1)])

    @staticmethod
    def get_displacement_3(x1, y1, z1, x2, y2, z2):

        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    @staticmethod
    def get_displacement_1(x1, x2):
        return sqrt((x1 - x2) ** 2)

    def get_total_length_of_path(self):
        """
        :return: int, total length of path represented by L parameter
        """
        return sum(self.displacements)

    def get_max_displacement(self):
        """
        :return: float, maximum displacement represented by d in all displacement list
        """
        return max(self.displacements)

    def get_diffusion_coef(self):
        """
        :return: float, diffusion coefficient represented by D parameter;
        estimated based on curve fitting of empirical and normal theoretical diffusion.
        """
        popt, _ = curve_fit(lambda x, d: generate_theoretical_msd_normal(x, d, self.dt), self.n_list,
                            self.empirical_msd)
        D = popt[0]
        return D

    def get_exponent_alpha(self):
        """
        :return: float, exponential anomalous parameter by alpha parameter;
        estimated based on curve fitting of empirical and normal anomalous diffusion.
        Modification of this function can also estimate D parameter
        """

        try:
            popt, _ = curve_fit(
                lambda x, log_D, a: generate_theoretical_msd_anomalous_log(log(self.dt * self.n_list), log_D, a),
                log(self.dt * self.n_list), log(self.empirical_msd), bounds=((-np.inf, 0), (np.inf, 2)))
            alpha = popt[1]
        except:
            alpha = None
        return alpha

    def get_tensor(self):
        """
        :return: matrix, the tensor T for given trajectory
        """
        if self.dim == 1:
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif self.dim == 3:
            l = int(len(self.data) / self.dim)
            x = self.data[:l]
            y = self.data[l:2 * l]
            z = self.data[2 * l:]
            a = sum((x - mean(x)) ** 2) / len(x)
            c = sum((y - mean(y)) ** 2) / len(y)
            e = sum((z - mean(z)) ** 2) / len(z)
            b = sum((x - mean(x)) * (y - mean(y))) / len(x)
            d = sum((x - mean(x)) * (z - mean(z))) / len(x)
            f = sum((y - mean(y)) * (z - mean(z))) / len(x)
            return np.array([[a, b, d], [b, c, f], [d, f, e]])

    def get_asymmetry(self):
        """
        The asymmetry of a trajectory can be used to detect directed motion.
        :return: float, asymmetry parameter - only real part of
        """

        lambda1 = self.eigenvalues[0]
        lambda2 = self.eigenvalues[1]
        lambda3 = self.eigenvalues[2]
        a = -1 * log(1 - ((lambda1 - lambda2) ** 2 + (lambda1 - lambda3) ** 2 + (lambda2 - lambda3) ** 2) / (
                2 * (lambda1 + lambda2 + lambda3) ** 2))
        return a.real

    def get_efficiency(self):
        """
        Efficiency relates the net squared displacement of a particle to the sum of squared step lengths
        :return: float, efficiency parameter
        """
        if self.dim == 1:
            x = self.data
            upper = self.get_displacement_1(x[self.N - 2], x[0]) ** 2
        else:
            l = int(len(self.data) / self.dim)
            x = self.data[:l]
            y = self.data[l:2 * l]
            z = self.data[2 * l:]
            upper = self.get_displacement_3(x[self.N - 2], y[self.N - 2], z[self.N - 1], x[0], y[0], z[0]) ** 2
        displacements_to_squere = self.displacements ** 2
        lower = (self.N - 1) * sum(displacements_to_squere)
        E = upper / lower
        return E

    def get_trappedness(self, n=3):
        """
        Trappedness is the probability that a diffusing particle with the diffusion coefficient D
        and traced for a time interval t is trapped in a bounded region with radius r0.
        :param n: int, given point of trappedness
        :return: float, probability of trappedness in point n
        """
        t = self.n_list * self.dt
        popt, _ = curve_fit(lambda x, d: generate_theoretical_msd_normal(self.n_list[:2], d, self.dt),
                            self.n_list[:2], self.empirical_msd[:2])
        d = popt[0]
        p = 1 - exp(0.2048 - 0.25117 * ((d * t) / (self.d / 2) ** 2))
        p = np.array([i if i > 0 else 0 for i in p])[n]
        return p

    def get_kurtosis_corrected(self):
        """
        Kurtosis measures the asymmetry and peakedness of the distribution of points within a trajectory
        :return: float, kurtosis for trajectory
        """
        if self.dim == 1:
            x = self.data
            K = kurtosis(x)
        else:
            l = int(len(self.data) / self.dim)
            x = self.data[:l]
            y = self.data[l:2 * l]
            z = self.data[2 * l:]
            index = where(self.eigenvalues == max(self.eigenvalues))[0][0]
            dominant_eigenvector = self.eigenvectors[index]
            a_prod_b = np.array([sum(np.array([x[i], y[i], z[i]]) * dominant_eigenvector) for i in range(len(x))])
            K = 1 / self.N * sum((a_prod_b - mean(a_prod_b)) ** 4 / std(a_prod_b) ** 4) - 3
        return K

    def get_fractal_dimension(self):
        """
        The fractal dimension is a measure of the space-filling capacity of a pattern.
        :return: float, fractional dimension parameter
        """
        upper = log(self.N)
        lower = log(self.N * self.L ** (-1) * self.d)
        D = upper / lower
        return D

    def get_gaussianity(self):
        """
        A trajectory’s Gaussianity checks the Gaussian statistics on increments
        :return: array, list of gaussianity points
        """
        r4 = generate_empirical_msd(self.data, self.dim, self.n_list, k=4)
        r2 = generate_empirical_msd(self.data, self.dim, self.n_list, k=2)

        g = -1 + 2 * r4 / (3 * r2 ** 2)
        return g

    def get_mean_gaussianity(self):
        """
        :return: float, mean of gaussianity points
        """
        return mean(self.gaussianity)

    def get_point_of_gaussianity(self, n=3):
        """
        :param n: int, point
        :return: float, point in gaussianity lists of points
        """
        return self.gaussianity[n]

    def get_mean_squared_displacement_ratio(self):
        """
        The mean square displacement ratio characterizes the shape of the MSD curve.
        :return: float, mean squared displacement ratio parameter
        """
        n1 = np.array(range(1, self.max_number_of_points_in_msd - 1))
        n2 = np.array(range(2, self.max_number_of_points_in_msd))
        r_n1 = self.empirical_msd[0:self.max_number_of_points_in_msd - 2]
        r_n2 = self.empirical_msd[1:self.max_number_of_points_in_msd]
        r = mean(r_n1 / r_n2 - n1 / n2)
        return r

    def get_straightness(self):
        """
        Straightness is a measure of the average direction change between subsequent steps.
        :return: float, straing
        """
        if self.dim == 1:
            x = self.data
            upper = self.get_displacement_1(x[self.N - 2], x[0])
            displacements = np.array(
                [self.get_displacement_1(x[i], x[i - 1]) for i in range(1, self.N - 1)])
        else:
            l = int(len(self.data) / self.dim)
            x = self.data[:l]
            y = self.data[l:2 * l]
            z = self.data[2 * l:]
            upper = self.get_displacement_3(x[self.N - 2], y[self.N - 2], z[self.N - 2], x[0], y[0], z[0])
            displacements = np.array(
                [self.get_displacement_3(x[i], y[i], z[i], x[i - 1], y[i - 1], z[i - 1])
                 for i in
                 range(1, self.N - 1)])

        lower = sum(displacements)
        S = upper / lower
        return S
