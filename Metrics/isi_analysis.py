import scipy.stats as stats
import numpy as np


def fit_gamma_distribution(isi_values):
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(isi_values)
    print(fit_alpha, fit_loc, fit_beta)
    return fit_alpha


def calculate_cv(isi_values):
    sd = np.std(isi_values)
    mean = np.mean(isi_values)
    return sd/mean







