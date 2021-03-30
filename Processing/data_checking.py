import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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


def check_metric_normality(response_vectors):
    sfc = [vector[0] for vector in response_vectors if vector[0] != 0]
    ifc = [vector[1] for vector in response_vectors]
    f_initial = [vector[2] for vector in response_vectors]
    B_frac = [vector[3] for vector in response_vectors]
    max_v = [vector[4] for vector in response_vectors]
    mean = [vector[5] for vector in response_vectors]
    m = [vector[6] for vector in response_vectors]
    c = [vector[7] for vector in response_vectors]
    tau = [vector[8] for vector in response_vectors]


    fig, axs = plt.subplots(3, 3, sharex=False)
    fig.suptitle(f"Normalcy Tests for Metrics", fontsize=30)
    fig.set_size_inches(24, 20)

    sns.histplot(sfc, ax=axs[0, 0])
    k2, p = stats.normaltest(sfc)
    axs[0,0].set_title(f"SFC, p={p}", fontsize=25)

    sns.histplot(ifc, ax=axs[0, 1])
    k2, p = stats.normaltest(ifc)
    axs[0,1].set_title(f"IFC, p={p}", fontsize=25)

    sns.histplot(f_initial, ax=axs[0, 2])
    k2, p = stats.normaltest(f_initial)
    axs[0,2].set_title(f"F_initial, p={p}", fontsize=25)

    sns.histplot(B_frac, ax=axs[1, 0])
    k2, p = stats.normaltest(B_frac)
    axs[1,0].set_title(f"B_fraction, p={p}", fontsize=25)

    sns.histplot(max_v, ax=axs[1, 1])
    k2, p = stats.normaltest(max_v)
    axs[1,1].set_title(f"Max, p={p}", fontsize=25)

    sns.histplot(mean, ax=axs[1, 2])
    k2, p = stats.normaltest(mean)
    axs[1,2].set_title(f"Mean, p={p}", fontsize=25)

    sns.histplot(m, ax=axs[2, 0])
    k2, p = stats.normaltest(m)
    axs[2,0].set_title(f"m_norm, p={p}", fontsize=25)

    sns.histplot(c, ax=axs[2, 1])
    k2, p = stats.normaltest(c)
    axs[2,1].set_title(f"c, p={p}", fontsize=25)

    sns.histplot(tau, ax=axs[2, 2])
    k2, p = stats.normaltest(tau)
    axs[2,2].set_title(f"tau, p={p}", fontsize=25)

    plt.show()


