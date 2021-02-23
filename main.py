import matplotlib.pyplot as plt

from load_data import load_data, load_all_epsp_data
from process_abf import get_all_sweep_data
from get_spike_data import get_spike_times_for_epsp, calculate_spike_rate, get_isi_values

# isi = get_isi_values(spike_times)


d = load_data("../../Granule-Data/GrC_Subject22_220118", "22118_0003 EPSP.abf")
# ds = get_all_sweep_data(d)
spike_times = get_spike_times_for_epsp(d)

r = calculate_spike_rate(spike_times, 0.001)
# abfobjects = load_all_epsp_data()

plt.plot(r)
plt.show()


x = True
