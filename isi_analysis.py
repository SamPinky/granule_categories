import scipy.stats as stats


def fit_gamma_distribution(isi_values):
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(isi_values)
    print(fit_alpha, fit_loc, fit_beta)

