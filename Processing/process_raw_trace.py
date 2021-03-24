import numpy as np
from scipy.signal import find_peaks
import peakutils
import matplotlib.pyplot as plt


def get_all_sweep_data(abf_object):
    sweep_data = []
    for i in range(abf_object.sweepCount):
        abf_object.setSweep(i)
        sweep_data.append([abf_object.sweepX, abf_object.sweepY])
    return sweep_data


def get_isi_values(spike_times):
    isi_values = [spike_times[i]-spike_times[i-1] for i, t in enumerate(spike_times) if i != 0]
    return isi_values


def get_all_isis(abf_objects):
    isis = []
    for abf_ob in abf_objects:
        sweep_data = get_all_sweep_data(abf_ob)
        for sweep in sweep_data:
            sweep = get_spike_times_for_trace(sweep)
            isis.append(get_isi_values(sweep))
    return isis


def get_spike_times_for_cc(abfdata, sweep_num, refrac=0.015, cleanup=True):
    spike_times = []
    for channel in range(abfdata.channelCount):
        abfdata.setSweep(sweep_num, channel=channel)
        thresholds = [np.mean(peakutils.baseline(abfdata.sweepY[int(len(abfdata.sweepY)/10): int(len(abfdata.sweepY)*2/3)])) + 25, -50]
        peaks, _ = find_peaks(abfdata.sweepY, height=max(thresholds))
        spike_times = spike_times + [abfdata.sweepX[point] for point in peaks]
    spike_times = sorted(spike_times)

    if not cleanup:
        return spike_times

    for p in range(10):
        to_remove = []
        next_pass = True
        for i, t in enumerate(spike_times):
            if next_pass:
                next_pass = False
                pass
            else:
                if t - spike_times[i-1] < refrac:
                    to_remove.append(i)
                    next_pass = True
        for r in reversed(to_remove):
            spike_times.pop(r)

    return spike_times


def get_spike_times_for_epsp(abfdata, threshold=0):
    peaks, _ = find_peaks(abfdata.sweepY, height=threshold)
    spike_times = [abfdata.sweepX[point] for point in peaks]
    return spike_times


def get_spike_times_for_trace(trace):
    peaks, _ = find_peaks(trace[1], height=0)
    spike_times = [trace[0][point] for point in peaks]
    return spike_times

