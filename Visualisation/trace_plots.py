import re
import numpy as np
import matplotlib.pyplot as plt

from Processing.process_raw_trace import get_spike_times_for_cc


def create_nep_plot(abf_objects, plot_number):
    n_subplots = max([o.sweepCount for o in abf_objects]) * 2
    if n_subplots > 2:
        fig, axs = plt.subplots(int(n_subplots / 2), 2, sharex=True)
    else:
        fig, axs = plt.subplots(2, 2, sharex=True)

    fig.suptitle(f"Plot group {plot_number}", fontsize=16)
    colors = ['C{}'.format(i) for i in range(3)]
    colors = colors[2:]
    line_length = [30]

    axs[0, 0].title.set_text(re.sub("/home/samp/Granule-Data/", "", abf_objects[0].abfFilePath))
    for i in range(abf_objects[0].sweepCount):
        spike_times = get_spike_times_for_cc(abf_objects[0], i)
        for j in range(abf_objects[0].channelCount):
            abf_objects[0].setSweep(i, channel=j)
            axs[i, 0].plot(abf_objects[0].sweepX, abf_objects[0].sweepY)
        line_offset = [30 + max(abf_objects[0].sweepY)]
        axs[i, 0].eventplot(np.unique(spike_times), colors=colors, lineoffsets=line_offset, linelengths=line_length)
        axs[i, 0].set_ylabel(f"I={abf_objects[0].sweepC[int(len(abf_objects[0].sweepC)/2)]}")
        axs[i, 0].tick_params(labelsize=15)

    if len(abf_objects) > 1:
        axs[0, 1].title.set_text(re.sub("/home/samp/Granule-Data/", "", abf_objects[1].abfFilePath))
        for i in range(abf_objects[1].sweepCount):
            spike_times = get_spike_times_for_cc(abf_objects[1], i)
            for j in range(abf_objects[1].channelCount):
                abf_objects[1].setSweep(i, channel=j)
                axs[i, 1].plot(abf_objects[1].sweepX, abf_objects[1].sweepY)
            line_offset = [30 + max(abf_objects[1].sweepY)]
            axs[i, 1].eventplot(spike_times, colors=colors, lineoffsets=line_offset, linelengths=line_length)
            axs[i, 1].plot(abf_objects[1].sweepX, abf_objects[1].sweepY)
            axs[i, 1].set_ylabel(f"I={abf_objects[1].sweepC[int(len(abf_objects[1].sweepC) / 2)]}")
            axs[i, 1].tick_params(labelsize=15)

    # Add graph annotations
    axs[int(n_subplots / 2 - 1), 0].set_xlabel("t", fontsize=25)
    axs[int(n_subplots / 2 - 1), 1].set_xlabel("t (", fontsize=25)
    fig.set_size_inches(18.5, 20)
    plt.show()


def plot_all_abf_data(abf_objects):
    last_i = len(abf_objects)
    x = 1
    it = iter(abf_objects)
    for obj in it:
        if last_i < x*2:
            create_nep_plot([obj], x)
        else:
            try:
                create_nep_plot([obj, next(it)], x)
            except IndexError:
                create_nep_plot(obj, x)
            x += 1
