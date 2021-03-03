from scipy.signal import find_peaks
import numpy as np


def get_spike_times_for_cc(abfdata):
    peaks, _ = find_peaks(abfdata.sweepY, height=-30)
    spike_times = [abfdata.sweepX[point] for point in peaks]
    return spike_times


def get_spike_times_for_epsp(abfdata):
    peaks, _ = find_peaks(abfdata.sweepY, height=0)
    spike_times = [abfdata.sweepX[point] for point in peaks]
    return spike_times


def get_spike_times_for_trace(trace):
    peaks, _ = find_peaks(trace[1], height=0)
    spike_times = [trace[0][point] for point in peaks]
    return spike_times


def get_isi_values(spike_times):
    isi_values = [spike_times[i]-spike_times[i-1] for i, t in enumerate(spike_times) if i != 0]
    return isi_values


def calculate_spike_rate(spike_times, bin_size=None):
    rate_values = []
    if bin_size is None:
        try:
            end_spike_time = max(spike_times)
            bin_size = end_spike_time
        except ValueError:
            bin_size = 0.1
            end_spike_time = 2
    for b in np.arange(0, end_spike_time+bin_size, bin_size):
        bin_spikes = [spike for spike in spike_times if b < spike <= b + bin_size]
        rate_values.append(len(bin_spikes)/bin_size)
    return rate_values
