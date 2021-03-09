import numpy as np
from sklearn.neighbors import KernelDensity

from process_raw_trace import get_all_sweep_data
from Processing.process_raw_trace import get_spike_times_for_epsp, get_spike_times_for_trace


def calculate_spike_rate(spike_times, bin_size=0.5):
    rate_values = []
    for b in np.arange(0, 2, bin_size):
        bin_spikes = [spike for spike in spike_times if b < spike <= b + bin_size]
        rate_values.append(len(bin_spikes)/bin_size)
    return rate_values


def get_all_spike_rates(abf_objects):
    spike_rates = []
    for abf_ob in abf_objects:
        sweep_data = get_all_sweep_data(abf_ob)
        for sweep in sweep_data:
            sweep = get_spike_times_for_trace(sweep)
            rate = calculate_spike_rate(sweep)
            spike_rates.append(rate)
    return spike_rates


def get_all_spike_rates_epsp(abf_objects):
    spike_rates = []
    for abf_ob in abf_objects:
        for sweep in range(abf_ob.sweepCount):
            abf_ob.setSweep(sweep)
            times = get_spike_times_for_epsp(abf_ob)
            rate = calculate_spike_rate(times)
            spike_rates.append(rate)
    return spike_rates


def calculate_spike_rate_kernel_smoothing(spike_times):
    x_d = np.linspace(0, max(spike_times)+0.5, 1000)
    spike_times = np.array(spike_times)
    model = KernelDensity(bandwidth=0.1, kernel='gaussian')
    model.fit(spike_times[:, None])
    log_dens = model.score_samples(x_d[:, None])
    whole_bin_spike_rate = len(spike_times)/2
    return np.exp(log_dens) * whole_bin_spike_rate
