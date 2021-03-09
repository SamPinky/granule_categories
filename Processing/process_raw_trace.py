from scipy.signal import find_peaks


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
            isi = get_isi_values(sweep)
            isis = isis + isi
    return isis


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

