import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from Processing.process_raw_trace import get_all_isis


def fit_gamma_distribution(isi_values):
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(isi_values)
    return fit_alpha


def calculate_cv(isi_values):
    sd = np.std(isi_values)
    mean = np.mean(isi_values)
    return sd/mean


def do_isi_analysis(abf_objects):
    isis = get_all_isis(abf_objects)
    metrics = []
    for isi in isis:
        if len(isi) > 5:
            metrics.append([fit_gamma_distribution(isi), calculate_cv(isi)])
    shape = [metric[0] for metric in metrics if metric[0] <1500]
    cv = [metric[1] for metric in metrics]
    metrics = [metric for metric in metrics if metric[0] < 1500]

    plt.hist(shape)
    plt.show()
    plt.hist(cv)
    plt.show()
    # test normal distribution
    kmeans_clustering(metrics)


def kmeans_clustering(metrics):
    km = KMeans(n_clusters=3).fit(metrics)

    plots = {}
    plots["shape"] = [metric[0] for metric in metrics]
    plots["cv"] = [metric[1] for metric in metrics]
    plots["labels"] = km.labels_

    plt.figure(figsize=(16, 10))
    plt.title(f"Kmeans")

    p1 = sns.scatterplot(
        x="shape", y="cv",
        # palette=sns.color_palette("hls", 10),
        hue="labels",
        # palette=sns.color_palette("hls"),
        data=plots,
        legend="full",
        alpha=1
    )
    plt.show()









