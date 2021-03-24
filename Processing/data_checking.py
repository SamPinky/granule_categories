import matplotlib.pyplot as plt
import seaborn as sns

from Processing.calculate_spike_rate import calculate_spike_rate
from Processing.process_raw_trace import get_isi_values, get_spike_times_for_cc, get_spike_times_for_epsp


def check_isi_normality(abf_data, cc=True, cleanup=True):
    isis = []
    if cc:
        for abf in abf_data:
            for sweep in range(abf.sweepCount):
                spike_times = get_spike_times_for_cc(abf, sweep, cleanup=cleanup)
                isis = isis + get_isi_values(spike_times)
    else:
        for abf in abf_data:
            for sweep in range(abf.sweepCount):
                abf.setSweep(sweep)
                spike_times = get_spike_times_for_epsp(abf)
                isis = isis + get_isi_values(spike_times)
    sns.histplot(isis)
    if cc:
        plt.title(f"ISI normality for cc Data, cleaned: {cleanup}")
    else:
        plt.title(f"ISI normality for EPSP Data, cleaned: {cleanup}")
    plt.show()


def check_whole_rate_normality(abf_data, cc, cleanup):
    spike_rates = []
    if cc:
        for abf in abf_data:
            for sweep in range(abf.sweepCount):
                spike_times = get_spike_times_for_cc(abf, sweep, cleanup=cleanup)
                spike_rates = spike_rates + calculate_spike_rate(spike_times, 2)
    else:
        for abf in abf_data:
            for sweep in range(abf.sweepCount):
                abf.setSweep(sweep)
                spike_times = get_spike_times_for_epsp(abf)
                spike_rates = spike_rates + calculate_spike_rate(spike_times, 2)

    sns.histplot(spike_rates)
    if cc:
        plt.title(f"Rate normality for cc Data, cleaned: {cleanup}")
    else:
        plt.title(f"Rate normality for EPSP Data, cleaned: {cleanup}")
    plt.show()

