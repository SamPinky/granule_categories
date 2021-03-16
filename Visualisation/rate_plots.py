import numpy as np
import matplotlib.pyplot as plt


from Processing.process_raw_trace import get_spike_times_for_cc
from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing


def create_psth(spike_times):
    x_d = np.linspace(0, max(spike_times)+0.5, 1000)
    dens = calculate_spike_rate_kernel_smoothing(spike_times)
    plt.fill_between(x_d, dens)
    plt.plot(spike_times, np.full_like(spike_times, -0.1), '|k', markeredgewidth=1)
    plt.show()


def plot_spike_rates(spike_rates):
    plt.hist(spike_rates, 30)
    plt.show()


def plot_all_psth(abf_objects):
    # TODO: Note this is for cc only.
    for i in abf_objects:
        neuron_spikes = []
        for j in range(i.sweepCount):
            i.setSweep(j)
            neuron_spikes = neuron_spikes + get_spike_times_for_cc(i, j)
        if len(neuron_spikes) > 1:
            create_psth(neuron_spikes)


