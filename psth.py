import matplotlib.pyplot as plt
import numpy as np


def create_psth(spike_counts):
    total_trials = len(spike_counts)
    spike_times = np.array([t for trial in spike_counts for t in trial])
    plt.hist(spike_times, 100)
    plt.show()















