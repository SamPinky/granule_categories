from scipy.signal import find_peaks


def get_spike_times_for_epsp(abfdata):
    peaks, _ = find_peaks(abfdata.sweepY, height=0)
    spike_times = [abfdata.sweepX[point] for point in peaks]
    return spike_times


def get_isi_values(spike_times):
    isi_values = [spike_times[i]-spike_times[i-1] for i, t in enumerate(spike_times) if i != 0]
    return isi_values

