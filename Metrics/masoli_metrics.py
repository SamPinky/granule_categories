import numpy as np
import pandas as pd

from Processing.process_raw_trace import get_spike_times_for_cc, get_spike_times_for_epsp
from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing


def calculate_all_metrics_for_epsp(abf_object):
    spike_t = []
    for sweep in range(abf_object.sweepCount):
        abf_object.setSweep(sweep)
        spike_t = spike_t + get_spike_times_for_cc(abf_object)
    if len(spike_t) > 1:
        pdf = calculate_spike_rate_kernel_smoothing(spike_t, max(obj.sweepX))
        return [calculate_ifc(pdf) * 100, calculate_sfc(pdf)]
    else:
        return None


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
        pdf = calculate_spike_rate_kernel_smoothing(spike_t, max(obj.sweepX))
        return calculate_ifc(pdf) * 100
    else:
        return None


def calculate_ifc_from_bins(spike_rates):
    ifc_values = []
    for rate in spike_rates:
        finitial = rate[0]
        ffinal = rate[-1]
        if finitial == 0:
            pass
        else:
            ifc_values.append([finitial, (ffinal-finitial)/finitial * 100])
    return ifc_values


def get_f_initial(abf_objects):
    results = []
    for object in abf_objects:
        spike_t = []
        for sweep in range(object.sweepCount):
            object.setSweep(sweep)
            spike_t = spike_t + get_spike_times_for_cc(object)
        if len(spike_t) > 1:
            pdf = calculate_spike_rate_kernel_smoothing(spike_t, max(obj.sweepX))
            results.append([np.mean(pdf[:500]), calculate_ifc(pdf) * 100])
    return results


def do_masoli_analysis(epsp_obj, cc_obj):
    neuron_names = set([obj.abfFolderPath.split("/")[-1] for obj in epsp_obj + cc_obj])
    epsp_results = pd.DataFrame(index=neuron_names, columns=["SFC", "average IFC"])
    for obj in epsp_obj:
        neuron_name = obj.abfFolderPath.split("/")[-1]
        fresp = []
        for sweep in range(obj.sweepCount):
            obj.setSweep(sweep)
            spikes = get_spike_times_for_epsp(obj)
            spikes = [spike for spike in spikes if 0.5 <= spike < 0.75]
            fresp.append(len(spikes)/0.25)
        fresp = np.mean(fresp)
        epsp_results.loc[neuron_name]["SFC"] = (fresp - 50)/50

    col_names = [i for i in range(-8, 26, 2)]
    neuron_names = [obj.abfFolderPath.split("/")[-1] for obj in cc_obj]
    new_neuron_names = []
    for i, neuron in enumerate(neuron_names):
        if neuron in neuron_names[:i]:
            new_neuron_names.append(neuron+"B")
        else:
            new_neuron_names.append(neuron+"A")
    ifc_results = pd.DataFrame(index=new_neuron_names, columns=col_names)
    neurons = []
    for i, obj in enumerate(cc_obj):  # TODO: Fix indexing for negative nums.
        neuron_name = new_neuron_names[i]
        for sweep in range(obj.sweepCount):
            obj.setSweep(sweep)
            spikes = get_spike_times_for_cc(obj, sweep)
            f_initial = len([spike for spike in spikes if spike <= 0.5])/0.5
            f_final = len([spike for spike in spikes if 1.5 < spike])/0.5
            if f_initial > 0:
                ifc_results.loc[neuron_name][sweep] = (f_final-f_initial)/f_initial
            else:
                ifc_results.loc[neuron_name][sweep] = f_final
    x = True


