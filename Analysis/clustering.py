import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing
from Processing.process_raw_trace import get_spike_times_for_cc

from Visualisation.metrics_plots import plot_metrics_against_clusters


def get_frequency_components(abf_objects):
    freq_components = []
    indexes = [i for i in range(0, 1000, 10)]
    for abf_obj in abf_objects:
        for sweep in range(abf_obj.sweepCount):
            abf_obj.setSweep(sweep)
            spike_times = get_spike_times_for_cc(abf_obj)
            if len(spike_times) > 1:
                kds_data = calculate_spike_rate_kernel_smoothing(spike_times, max(abf_obj.sweepX))
                kds_data = [kds_data[i] for i in indexes]
                freq_components.append(kds_data)
    return freq_components


def calculate_distances_between_categories(tsne_ouput, categories):
    distances = np.zeros(shape=(len(categories), len(categories)))
    same_categories = np.zeros(shape=(len(categories), len(categories)))
    for i, row in enumerate(distances):
        for j, val in enumerate(row):
            if categories[i] == categories[j]:
                same_categories[i, j] = 1
            distances[i, j] = ((tsne_ouput["tsne-2d-one"][i]-tsne_ouput["tsne-2d-one"][j])**2 + (tsne_ouput["tsne-2d-two"][i]-tsne_ouput["tsne-2d-two"][j])**2)**0.5
    distances_between_categories = 0
    distances_between_all = 0
    num_sane_category = 0
    for i, row in enumerate(distances):
        for j, val in enumerate(row):
            if j <= i:
                pass
            else:
                if same_categories[i, j] == 1:
                    num_sane_category += 1
                    distances_between_categories += val
                distances_between_all += val
    print(f"Distances between categories = {distances_between_categories/num_sane_category}")
    total_comparisons = ((len(categories)-1)/(len(categories)*2))*(len(categories)**2)
    print(f"Distances between all = {distances_between_all/total_comparisons}")


def tsne_on_full_vector(vectors, neuron_names, labels=None, labels2=None):
    tsne = TSNE(n_components=2, n_iter=1000, perplexity=3)
    for i, vector in enumerate(vectors):
        for j, metric in enumerate(vector):
            if metric is None:
                vectors[i, j] = 0
    tsne_results = tsne.fit_transform(vectors)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(6, 6))

    if labels is not None:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=labels,
            palette=sns.color_palette("hls", len(set(labels))),
            data=tpd,
            legend=True,
            alpha=1
        )
    else:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            data=tpd,
            legend=True,
            alpha=1
        )
    if neuron_names is not None:
        for line in range(len(vectors)):
            p1.text(tpd['tsne-2d-one'][line] + 0.01, tpd['tsne-2d-two'][line],
                    neuron_names[line], horizontalalignment='left',
                    size='small', color='black', weight='bold')
    plt.show()
    plt.figure(figsize=(6, 6))

    if labels2 is not None:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=labels2,
            palette=sns.color_palette("hls", len(set(labels2))),
            data=tpd,
            alpha=1
        )
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.tight_layout()

        plt.show()
    print("Original labels")
    calculate_distances_between_categories(tsne_ouput=tpd, categories=labels)
    print("New labels")
    calculate_distances_between_categories(tsne_ouput=tpd, categories=labels2)
    return tpd


def do_tsne_on_kdfs(kdfs, neuron_names, labels=None):
    tsne = TSNE(n_components=2, n_iter=1000, perplexity=3)
    tsne_results = tsne.fit_transform(kdfs)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(6, 6))
    plt.title(f"T-SNE results on KDFs")

    if labels is not None:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=labels,
            palette=sns.color_palette("hls", len(set(labels))),
            data=tpd,
            legend="brief",
            alpha=1
        )
    else:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            data=tpd,
            legend="full",
            alpha=1
        )
    # for line in range(len(kdfs)):
    #     p1.text(tpd['tsne-2d-one'][line] + 0.01, tpd['tsne-2d-two'][line],
    #             neuron_names[line], horizontalalignment='left',
    #             size='small', color='black', weight='bold')
    print("KDF labels")
    calculate_distances_between_categories(tpd, labels)
    plt.show()


def knn_on_ifc_initial(vectors):
    ifc = [vector[1] for vector in vectors]
    finitial = [vector[2] for vector in vectors]

    ifc_subset = []
    finitial_subset = []
    labels_full = [None for i in range(len(ifc))]
    for i, ifcc, finit in zip(range(len(ifc)), ifc, finitial):
        if ifcc >= 40:
            labels_full[i] = 3
        else:
            ifc_subset.append(ifcc)
            finitial_subset.append(finit)

    # Finidng optimal cluster num
    sil = []
    for i in range(2, 5):
        nbrs = KMeans(n_clusters=i, n_init=20).fit([[f, ifc_v] for f, ifc_v in zip(finitial_subset, ifc_subset)])
        labels_new = nbrs.labels_
        sil.append(silhouette_score([[f, ifc_v] for f, ifc_v in zip(finitial_subset, ifc_subset)], labels_new, metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil)) + 1
    nbrs = KMeans(n_clusters=optimal_num, n_init=20).fit([[f, ifc_v] for f, ifc_v in zip(finitial_subset, ifc_subset)])
    labels_new = nbrs.labels_

    i = 0
    for j, l in enumerate(labels_full):
        if l is None:
            labels_full[j] = labels_new[i]
            i += 1
        else:
            labels_full[j] = max(labels_new) + 1
    return labels_full


def knn_on_strong_weak(response_vectors):
    max_v = [vector[4] for vector in response_vectors]
    mean = [vector[5] for vector in response_vectors]
    B_frac = [vector[3] for vector in response_vectors]

    # Finidng optimal cluster num
    sil = []
    for i in range(2, 5):
        nbrs = KMeans(n_clusters=i, n_init=20).fit([[mx, mn, bf] for mx, mn, bf in zip(max_v, mean, B_frac)])
        labels = nbrs.labels_
        sil.append(silhouette_score([[mx, mn, bf] for mx, mn, bf in zip(max_v, mean, B_frac)], labels,
                                    metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil)) + 2
    nbrs = KMeans(n_clusters=optimal_num, n_init=20).fit([[mx, mn, bf] for mx, mn, bf in zip(max_v, mean, B_frac)])
    labels = nbrs.labels_

    return labels


def knn_on_slow_fast_onset(response_vectors):
    m = [vector[6] for vector in response_vectors]
    tau = [vector[8] for vector in response_vectors]
    c = [vector[7] for vector in response_vectors]

    sil = []
    for i in range(2, 5):
        nbrs = KMeans(n_clusters=i, n_init=20).fit([[cc, mm, t] for cc, mm, t in zip(c, m, tau)])
        labels = nbrs.labels_
        sil.append(silhouette_score([[cc, mm, t] for cc, mm, t in zip(c, m, tau)], labels,
                                    metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil)) + 2
    nbrs = KMeans(n_clusters=optimal_num, n_init=20).fit([[cc, mm, t] for cc, mm, t in zip(c, m, tau)])
    labels = nbrs.labels_

    return labels


def knn_on_slow_fast_adapt_accel(response_vectors):
    B_frac = [vector[3] for vector in response_vectors]
    m = [vector[6] for vector in response_vectors]
    tau = [vector[8] for vector in response_vectors]

    sil = []
    for i in range(2, 5):
        nbrs = KMeans(n_clusters=i, n_init=20).fit([[bf, mm, t] for bf, mm, t in zip(B_frac, m, tau)])
        labels = nbrs.labels_
        sil.append(silhouette_score([[bf, mm, t] for bf, mm, t in zip(B_frac, m, tau)], labels,
                                    metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil)) + 2
    nbrs = KMeans(n_clusters=optimal_num, n_init=20).fit([[bf, mm, t] for bf, mm, t in zip(B_frac, m, tau)])
    labels = nbrs.labels_

    return labels


def knn_full_response_vector(vectors):
    vectors = np.delete(vectors, 0, 1)

    sil = []
    for i in range(2, 5):
        nbrs = KMeans(n_clusters=i, n_init=20).fit(vectors)
        labels = nbrs.labels_
        sil.append(silhouette_score(vectors, labels, metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil))
    nbrs = KMeans(n_clusters=optimal_num, n_init=20).fit(vectors)
    labels = nbrs.labels_

    return labels


def agglomerative_clustering_on_vectors(response_vectors):
    ifc = [vector[1] for vector in response_vectors]
    finitial = [vector[2] for vector in response_vectors]
    data = [[fin, ifcc] for fin, ifcc in zip(finitial, ifc)]
    aggl = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data)
    return aggl


def all_clusters(vectors, neuron_names):
    labels1 = knn_on_ifc_initial(vectors)
    plot_metrics_against_clusters(vectors, neuron_names, labels1, "KNN Adapt-Accel")

    labels2 = knn_on_strong_weak(vectors)
    plot_metrics_against_clusters(vectors, neuron_names, labels2, "KNN Strong vs Weak")

    labels3 = knn_on_slow_fast_onset(vectors)
    plot_metrics_against_clusters(vectors, neuron_names, labels3, "KNN Slow vs Fast Onset")

    labels4 = knn_on_slow_fast_adapt_accel(vectors)
    plot_metrics_against_clusters(vectors, neuron_names, labels4, "KNN Slow vs Fast Adapt-Accel")
    results = pd.DataFrame([labels1, labels2, labels3, labels4], columns=neuron_names)
    return results

