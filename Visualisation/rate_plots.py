import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ParabolicModel
from sklearn.linear_model import LinearRegression


from Processing.process_raw_trace import get_spike_times_for_cc
from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing


def fit_quadratic(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    qmodel = ParabolicModel()
    pars = qmodel.guess(y, x=x)
    result = qmodel.fit(y, pars, x=x)
    e = 0
    for xi, yi in zip(x, y):
        p = qmodel.eval(result.params, x=xi)
        e += (yi - p) ** 2
    return result.params["a"].value, result.params["b"].value, result.params["c"].value, e/len(x)


def fit_linear(points):
    x = np.array([point[0] for point in points])
    x = x[:, None]
    y = [point[1] for point in points]
    model = LinearRegression().fit(x, y)
    e = model.score(x, y)
    m = model.coef_
    c = model.intercept_
    return m, c, e


def create_psth(spike_times, end, function=False, name=None):
    # x_d = np.linspace(0, max(spike_times)+0.5, 1000)
    x_d = np.linspace(0, end, 1000)
    dens = calculate_spike_rate_kernel_smoothing(spike_times, end)
    plt.fill_between(x_d, dens)
    plt.title(name)
    plt.plot(spike_times, np.full_like(spike_times, -0.1), '|k', markeredgewidth=1)
    if function:
        indexes = range(1000)
        maxima = [[x, y] for i, x, y in zip(indexes, x_d, dens) if dens[i - 1] < y > dens[i + 1]]
        # if len(maxima) > 2:
        #     new_a, new_b, new_c, new_e = fit_quadratic(maxima)
        #     y = (x_d**2)*new_b + x_d*new_b + new_c
        if len(maxima) > 1:
            new_b, new_c, new_e = fit_linear(maxima)
            y = x_d*new_b + new_c
        else:
            return
        plt.plot(x_d, y, "g")
        plt.title(f"{name}, Maxima: {len(maxima)}, m={new_b}, c={new_c}")
    plt.show()


def plot_spike_rates(spike_rates):
    plt.hist(spike_rates, 30)
    plt.show()


def plot_all_psth(abf_objects, function=False):
    for i in abf_objects:
        # neuron_spikes = []
        # for j in range(i.sweepCount):
        #     i.setSweep(j)
        #     neuron_spikes = neuron_spikes + get_spike_times_for_cc(i, j)
        neuron_spikes = get_spike_times_for_cc(i, 9)
        if len(neuron_spikes) > 1:
            create_psth(neuron_spikes, max(i.sweepX), function, i.abfFolderPath.split("/")[-1])

