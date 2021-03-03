import numpy as np
from get_spike_data import calculate_spike_rate, get_spike_times_for_epsp, get_spike_times_for_cc
from check_spike_rate_distribution import calculate_spike_rate_kernel_smoothing


def calculate_all_metrics_for_epsp(abf_object):
    spike_t = []
    for sweep in range(abf_object.sweepCount):
        abf_object.setSweep(sweep)
        spike_t = spike_t + get_spike_times_for_cc(abf_object)
    if len(spike_t) > 1:
        pdf = calculate_spike_rate_kernel_smoothing(spike_t)
        return [calculate_ifc(pdf) * 100, calculate_sfc(pdf)]
    else:
        return None


def calculate_sfc(spike_rates):
    fresp = np.mean(spike_rates)
    return (fresp - 50)/50


def calculate_ifc(spike_rates):
    # Intrinsic frequency change (ffinal-finitial)/finitial
    finitial = np.mean(spike_rates[:500])
    ffinal = np.mean(spike_rates[-500:])
    return (ffinal-finitial)/finitial


def calculate_all_metrics_for_cc(abf_object):
    spike_t = []
    for sweep in range(abf_object.sweepCount):
        abf_object.setSweep(sweep)
        spike_t = spike_t + get_spike_times_for_cc(abf_object)
    if len(spike_t) > 1:
        pdf = calculate_spike_rate_kernel_smoothing(spike_t)
        return calculate_ifc(pdf) * 100
    else:
        return None


def get_f_initial(abf_objects):
    results = []
    for object in abf_objects:
        spike_t = []
        for sweep in range(object.sweepCount):
            object.setSweep(sweep)
            spike_t = spike_t + get_spike_times_for_cc(object)
        if len(spike_t) > 1:
            pdf = calculate_spike_rate_kernel_smoothing(spike_t)
            results.append([np.mean(pdf[:500]), calculate_ifc(pdf) * 100])
    return results
