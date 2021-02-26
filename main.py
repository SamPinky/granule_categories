import matplotlib.pyplot as plt

from load_data import load_data, load_all_epsp_data, tabulate_data_format
from process_abf import get_all_sweep_data, get_all_isis
from check_spike_rate_distribution import plot_spike_rates, get_all_spike_rates
from get_spike_data import get_spike_times_for_epsp, calculate_spike_rate, get_isi_values
from isi_analysis import fit_gamma_distribution

# isi = get_isi_values(spike_times)


# d = load_data("../../Granule-Data/GrC_Subject22_220118", "22118_0003 EPSP.abf")
# ds = get_all_sweep_data(d)
# spike_times = get_spike_times_for_epsp(d)

# r = calculate_spike_rate(spike_times, 0.001)
abfobjects = load_all_epsp_data()
tabulate_data_format(abfobjects)

# # rates = get_all_spike_rates(abfobjects)
# isis = get_all_isis(abfobjects)
# # print(f"Number of recordings: {len(rates)}")
# fit_gamma_distribution(isis)

x = True