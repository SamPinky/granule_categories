import matplotlib.pyplot as plt

from load_data import load_data, load_all_epsp_data, tabulate_data_format, load_all_cc_data
from process_abf import get_all_sweep_data, get_all_isis
from check_spike_rate_distribution import plot_spike_rates, get_all_spike_rates,calculate_spike_rate_kernel_smoothing
from get_spike_data import get_spike_times_for_epsp, get_spike_times_for_cc, calculate_spike_rate, get_isi_values
from isi_analysis import fit_gamma_distribution
from visualisation import plot_all_abf_data, create_psth
from calculate_their_metrics import calculate_all_metrics_for_cc, get_f_initial

# isi = get_isi_values(spike_times)


# d = load_data("../../Granule-Data/GrC_Subject22_220118", "22118_0003 EPSP.abf")
# ds = get_all_sweep_data(d)
# spike_times = get_spike_times_for_epsp(d)

# r = calculate_spike_rate(spike_times, 0.001)
abfobjects = load_all_epsp_data() + load_all_cc_data()
plot_all_abf_data(abfobjects)



# results = get_f_initial(abfobjects)
# x = [i[0] for i in results]
# y = [i[1] for i in results]
# plt.scatter(x, y)
# plt.show()
#
# c = []
# for obj in abfobjects:
#     for sweep in range(obj.sweepCount):
#         c.append(min(obj.sweepC))
#
# print(max(c))
# print(min(c))
#
ifc = []

for i in abfobjects:
    ifc.append(calculate_all_metrics_for_cc(i))
abfobjects = load_all_cc_data()
for i in abfobjects:
    ifc.append(calculate_all_metrics_for_cc(i))

# for i in abfobjects:
#     neuron_spikes = []
#     for j in range(i.sweepCount):
#         i.setSweep(j)
#         neuron_spikes = neuron_spikes + get_spike_times_for_cc(i)
#     if len(neuron_spikes) > 1:
#         create_psth(neuron_spikes)
# # #


# # rates = get_all_spike_rates(abfobjects)
# isis = get_all_isis(abfobjects)
# # print(f"Number of recordings: {len(rates)}")
# fit_gamma_distribution(isis)

x = True