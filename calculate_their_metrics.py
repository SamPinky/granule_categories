import numpy as np
from get_spike_data import calculate_spike_rate, get_spike_times_for_epsp, get_spike_times_for_cc
from check_spike_rate_distribution import calculate_spike_rate_kernel_smoothing

def calculate_all_metrics_for_epsp(abf_object):
    ifc_values = []
    sfc_values = []
    snr_values = []

    # Loop over sweeps, getting spike times and rates for each, the using them to calculate the values.
    # Intrinsic frequency change (ffinal-finitial)/finitial
    # Synaptic frequncy change (fresp-fstim)/fstim
    # SN ratio frestp100hz/fresp20hz
    ...


def calculate_ifc(spike_rates):
    # Intrinsic frequency change (ffinal-finitial)/finitial
    return (spike_rates[-1]-spike_rates[0]/spike_rates[0])


def calculate_all_metrics_for_cc(abf_object):
    ifc_values = []
    for sweep in range(abf_object.sweepCount):
        abf_object.setSweep(sweep)
        spike_t = get_spike_times_for_cc(abf_object)
        calculate_spike_rate_kernel_smoothing(spike_t)
        spike_r = calculate_spike_rate(spike_t)
        if spike_r:
            ifc_values.append(calculate_ifc(spike_r))
    return np.mean(ifc_values)
