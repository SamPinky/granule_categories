import matplotlib.pyplot as plt

from load_data import load_data
from get_spike_data import get_spike_times_for_epsp, get_isi_values

epsp_file = "090216_0004 EPSP.abf"

data = load_data(file_name=epsp_file)

for i in range(7):
    data.setSweep(i)
    plt.plot(data.sweepX, data.sweepY)
    plt.show()

spike_times = get_spike_times_for_epsp(data)
isi = get_isi_values(spike_times)

x = True
