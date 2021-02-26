from get_spike_data import get_isi_values, calculate_spike_rate, get_spike_times_for_epsp, get_spike_times_for_trace


def get_all_sweep_data(abf_object):
    sweep_data = []
    for i in range(abf_object.sweepCount):
        abf_object.setSweep(i)
        sweep_data.append([abf_object.sweepX, abf_object.sweepY])
    return sweep_data


def get_all_isis(abf_objects):
    isis = []
    for abf_ob in abf_objects:
        sweep_data = get_all_sweep_data(abf_ob)
        for sweep in sweep_data:
            sweep = get_spike_times_for_trace(sweep)
            isi = get_isi_values(sweep)
            isis = isis + isi
    return isis

