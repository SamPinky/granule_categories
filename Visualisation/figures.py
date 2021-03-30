import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


from Processing.process_raw_trace import get_spike_times_for_cc
from Processing.calculate_spike_rate import get_spike_times_for_epsp
from Visualisation.rate_plots import create_psth, fit_linear
from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing


def figure_1(cc_file, epsp_file):
    fig, axs = plt.subplots(2, sharex=True)

    for j in range(cc_file.channelCount):
        cc_file.setSweep(9, channel=j)
        axs[0].plot(cc_file.sweepX, cc_file.sweepY)

    axs[0].hlines(y=min(cc_file.sweepY) - 10, xmin=0.1, xmax=2.1, color="r", linewidth=5)

    for j in range(epsp_file.channelCount):
        epsp_file.setSweep(9, channel=j)
        axs[1].plot(epsp_file.sweepX, epsp_file.sweepY)

    axs[1].hlines(y=min(epsp_file.sweepY) - 30, xmin=0.5, xmax=0.75, color="r", linewidth=5)

    axs[0].set_ylabel(f"V (mV)", fontsize=30)
    axs[1].set_ylabel(f"V (mV)", fontsize=30)
    axs[1].set_xlabel(f"Time (s)", fontsize=40)
    axs[0].tick_params(labelsize=25)
    axs[1].tick_params(labelsize=25)

    fig.set_size_inches(18.5, 20)
    plt.show()


def figure_2(cc_objects, epsp_file):
    fig, axs = plt.subplots(3, sharex=True)

    colors = ['C{}'.format(i) for i in range(3)]
    colors = colors[2:]
    line_length = [30]

    for i in range(0, 2):
        spike_times = get_spike_times_for_cc(cc_objects[i], 9)
        for j in range(cc_objects[i].channelCount):
            cc_objects[i].setSweep(9, channel=j)
            axs[i].plot(cc_objects[i].sweepX, cc_objects[i].sweepY)
        line_offset = [30 + max(cc_objects[i].sweepY)]
        axs[i].eventplot(np.unique(spike_times), colors=colors, lineoffsets=line_offset, linelengths=line_length)
        axs[i].hlines(y=min(cc_objects[i].sweepY) - 10, xmin=0.1, xmax=0.6, color="r", linewidth=5)
        axs[i].hlines(y=min(cc_objects[i].sweepY) - 10, xmin=1.6, xmax=2.1, color="g", linewidth=5)

    spike_times = get_spike_times_for_epsp(epsp_file, 9)
    for j in range(epsp_file.channelCount):
        epsp_file.setSweep(9, channel=j)
        axs[2].plot(epsp_file.sweepX, epsp_file.sweepY)
    line_offset = [30 + max(epsp_file.sweepY)]
    axs[2].eventplot(np.unique(spike_times), colors=colors, lineoffsets=line_offset, linelengths=line_length)

    axs[2].hlines(y=min(epsp_file.sweepY) - 30, xmin=0.5, xmax=0.75, color="r", linewidth=5)

    axs[0].set_ylabel(f"V (mV)", fontsize=30)
    axs[1].set_ylabel(f"V (mV)", fontsize=30)
    axs[2].set_ylabel(f"V (mV)", fontsize=30)
    axs[2].set_xlabel(f"Time (s)", fontsize=40)
    axs[0].tick_params(labelsize=25)
    axs[1].tick_params(labelsize=25)
    axs[2].tick_params(labelsize=25)

    fig.set_size_inches(18.5, 20)
    plt.show()


    fig, axs = plt.subplots(2, sharex=True)

    for i in range(0, 2):
        spike_times = get_spike_times_for_cc(cc_objects[i], 9)
        x_d = np.linspace(0, 2.5, 1000)
        dens = calculate_spike_rate_kernel_smoothing(spike_times, 2.5)
        axs[i].fill_between(x_d, dens)
        axs[i].plot(spike_times, np.full_like(spike_times, -0.1), '|k', markeredgewidth=1)

    axs[0].set_ylabel(f"Spike Rate (HZ)", fontsize=20)
    axs[1].set_ylabel(f"Spike Rate (HZ)", fontsize=20)
    axs[1].set_xlabel(f"Time (s)", fontsize=30)
    axs[0].tick_params(labelsize=25)
    axs[1].tick_params(labelsize=25)
    fig.set_size_inches(9.25, 10)

    plt.show()


def figure_3(cc_object):
    fig, axs = plt.subplots(3, sharex=True)

    for i in range(0, 3):
        spike_times = get_spike_times_for_cc(cc_object, 9)
        x_d = np.linspace(0, 2.5, 1000)
        dens = calculate_spike_rate_kernel_smoothing(spike_times, 2.5)
        axs[i].fill_between(x_d, dens)
        axs[i].plot(spike_times, np.full_like(spike_times, -0.1), '|k', markeredgewidth=1)
        if i == 0:
            indexes = range(1000)
            maxima = [[x, y] for i, x, y in zip(indexes, x_d, dens) if dens[i - 1] < y > dens[i + 1]]
            new_b, new_c, new_e = fit_linear(maxima)
            maxima.append([0, new_c])
            print(new_b, new_c)
            y = x_d * new_b + new_c
            axs[i].plot(x_d, y, "g")
            axs[i].scatter([m[0] for m in maxima], [m[1] for m in maxima])
        elif i == 1:
            indexes = range(1000)
            maximum = [[x, y] for i, x, y in zip(indexes, x_d, dens) if dens[i - 1] < y > dens[i + 1]][0]
            axs[i].hlines(y=np.mean(dens), color="r", linewidth=2, xmin=0, xmax=2)
            axs[i].scatter([maximum[0]], [maximum[1]])

    axs[0].set_ylabel(f"Spike Rate (HZ)", fontsize=20)
    axs[1].set_ylabel(f"Spike Rate (HZ)", fontsize=20)
    axs[2].set_ylabel(f"Spike Rate (HZ)", fontsize=20)
    axs[2].set_xlabel(f"Time (s)", fontsize=30)
    axs[0].tick_params(labelsize=25)
    axs[1].tick_params(labelsize=25)
    fig.set_size_inches(9.25, 15)

    plt.show()


def figure_5(response_vectors, all_clusters):
    cluster2 = [str(all_clusters.iloc[1, i]) for i in range(len(all_clusters.columns))]
    cluster3 = [str(all_clusters.iloc[2, i]) for i in range(len(all_clusters.columns))]
    cluster4 = [str(all_clusters.iloc[3, i]) for i in range(len(all_clusters.columns))]

    sfc = [vector[0] for vector in response_vectors]
    for i, v in enumerate(sfc):
        if v == 0:
            sfc[i] = None
    ifc = [vector[1] for vector in response_vectors]
    f_initial = [vector[2] for vector in response_vectors]
    B_frac = [vector[3] for vector in response_vectors]
    max_v = [vector[4] for vector in response_vectors]
    mean = [vector[5] for vector in response_vectors]
    m = [vector[6] for vector in response_vectors]
    c = [vector[7] for vector in response_vectors]
    tau = [vector[8] for vector in response_vectors]

    colours = {"0": "red", "1": "green", "2": "blue", "3": "yellow"}
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    sc = ax.scatter(mean, max_v, B_frac, c=[colours[c] for c in cluster2])
    ax.set_xlabel("mean (Hz)", size=15)
    ax.set_ylabel("max (Hz)", size=15)
    ax.set_zlabel("B_fraction", size=15)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig("scatter_hue1", bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    sc = ax.scatter(tau, c, m, c=[colours[c] for c in cluster3])
    ax.set_xlabel("Tau (s)", size=15)
    ax.set_ylabel("c (Hz)", size=15)
    ax.set_zlabel("m_norm (s-1)", size=15)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig("scatter_hue2", bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    sc = ax.scatter(B_frac, m, tau, c=[colours[c] for c in cluster4])
    ax.set_xlabel("B_fraction", size=15)
    ax.set_ylabel("m_norm (s-1)", size=15)
    ax.set_zlabel("Tau (s)", size=15)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig("scatter_hue3", bbox_inches='tight')
    plt.show()



