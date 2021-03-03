import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from process_abf import get_all_sweep_data
from get_spike_data import calculate_spike_rate, get_spike_times_for_epsp, get_spike_times_for_trace


def plot_spike_rates(spike_rates):
    plt.hist(spike_rates, 30)
    plt.show()


def get_all_spike_rates(abf_objects):
    spike_rates = []
    for abf_ob in abf_objects:
        sweep_data = get_all_sweep_data(abf_ob)
        for sweep in sweep_data:
            sweep = get_spike_times_for_trace(sweep)
            rate = calculate_spike_rate(sweep)
            spike_rates.append(rate[0])
    return spike_rates


def calculate_spike_rate_kernel_smoothing(spike_times):
    # Get all spike times in a single list.
    spike_times = spike_times[:, np.newaxis]
    model = KernelDensity()
    model.fit(spike_times)
    log_dens = model.score_samples(spike_times)
    plt.fill(spike_times, np.exp(log_dens), c='cyan')
    plt.show()




