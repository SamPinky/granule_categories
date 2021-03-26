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


def do_tsne_on_ks(freq_components):
    iters = [i for i in range(250, 1000, 250)]
    perplexi = [i for i in range(50, 150, 50)]
    for it in iters:
        for p in perplexi:
            tsne = TSNE(n_components=2, n_iter=it, perplexity=p)
            tsne_results = tsne.fit_transform(freq_components)

            tpd = {}

            tpd['tsne-2d-one'] = tsne_results[:, 0]
            tpd['tsne-2d-two'] = tsne_results[:, 1]
            tpd['Point'] = ["Blue" for i in range(len(tsne_results[:, 0]))]
            tpd["Point"][0] = "Red"
            plt.figure(figsize=(16, 10))
            plt.title(f"tsne results {it} {p}")

            p1 = sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                # palette=sns.color_palette("hls", 10),
                hue="Point",
                # palette=sns.color_palette("hls"),
                data=tpd,
                legend="full",
                alpha=0.3
            )
            plt.show()


def tsne_on_full_vector(vectors, neuron_names, labels=None):
    tsne = TSNE(n_components=2, n_iter=1000, perplexity=3)
    for i, vector in enumerate(vectors):
        for j, metric in enumerate(vector):
            if metric is None:
                vectors[i, j] = 0
    tsne_results = tsne.fit_transform(vectors)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))

    if labels is not None:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=labels,
            palette=sns.color_palette("hls", len(set(labels))),
            data=tpd,
            legend="full",
            alpha=1
        )
    else:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            data=tpd,
            legend="full",
            alpha=1
        )
    if neuron_names is not None:
        for line in range(len(vectors)):
            p1.text(tpd['tsne-2d-one'][line] + 0.01, tpd['tsne-2d-two'][line],
                    neuron_names[line], horizontalalignment='left',
                    size='small', color='black', weight='bold')
    plt.show()
    return tpd


def do_tsne_on_kdfs(kdfs, neuron_names, labels=None):
    tsne = TSNE(n_components=2, n_iter=1000, perplexity=3)
    tsne_results = tsne.fit_transform(kdfs)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    plt.title(f"T-SNE results on KDFs")

    if labels is not None:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=labels,
            palette=sns.color_palette("hls", len(set(labels))),
            data=tpd,
            legend="full",
            alpha=1
        )
    else:
        p1 = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            data=tpd,
            legend="full",
            alpha=1
        )
    for line in range(len(kdfs)):
        p1.text(tpd['tsne-2d-one'][line] + 0.01, tpd['tsne-2d-two'][line],
                neuron_names[line], horizontalalignment='left',
                size='small', color='black', weight='bold')
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
        nbrs = KMeans(n_clusters=i).fit([[f, ifc_v] for f, ifc_v in zip(finitial_subset, ifc_subset)])
        labels_new = nbrs.labels_
        sil.append(silhouette_score([[f, ifc_v] for f, ifc_v in zip(finitial_subset, ifc_subset)], labels_new, metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil)) + 1
    nbrs = KMeans(n_clusters=optimal_num).fit([[f, ifc_v] for f, ifc_v in zip(finitial_subset, ifc_subset)])
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
        nbrs = KMeans(n_clusters=i).fit([[mx, mn, bf] for mx, mn, bf in zip(max_v, mean, B_frac)])
        labels = nbrs.labels_
        sil.append(silhouette_score([[mx, mn, bf] for mx, mn, bf in zip(max_v, mean, B_frac)], labels,
                                    metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil)) + 2
    nbrs = KMeans(n_clusters=optimal_num).fit([[mx, mn, bf] for mx, mn, bf in zip(max_v, mean, B_frac)])
    labels = nbrs.labels_

    return labels


def knn_on_slow_fast_onset(response_vectors):
    m = [vector[6] for vector in response_vectors]
    tau = [vector[8] for vector in response_vectors]
    c = [vector[7] for vector in response_vectors]

    sil = []
    for i in range(2, 5):
        nbrs = KMeans(n_clusters=i).fit([[cc, mm, t] for cc, mm, t in zip(c, m, tau)])
        labels = nbrs.labels_
        sil.append(silhouette_score([[cc, mm, t] for cc, mm, t in zip(c, m, tau)], labels,
                                    metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil)) + 2
    nbrs = KMeans(n_clusters=optimal_num).fit([[cc, mm, t] for cc, mm, t in zip(c, m, tau)])
    labels = nbrs.labels_

    return labels


def knn_on_slow_fast_adapt_accel(response_vectors):
    B_frac = [vector[3] for vector in response_vectors]
    m = [vector[6] for vector in response_vectors]
    tau = [vector[8] for vector in response_vectors]

    sil = []
    for i in range(2, 5):
        nbrs = KMeans(n_clusters=i).fit([[bf, mm, t] for bf, mm, t in zip(B_frac, m, tau)])
        labels = nbrs.labels_
        sil.append(silhouette_score([[bf, mm, t] for bf, mm, t in zip(B_frac, m, tau)], labels,
                                    metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil)) + 2
    nbrs = KMeans(n_clusters=optimal_num).fit([[bf, mm, t] for bf, mm, t in zip(B_frac, m, tau)])
    labels = nbrs.labels_

    return labels


def knn_full_response_vector(vectors):
    vectors = np.delete(vectors, 0, 1)

    sil = []
    for i in range(2, 5):
        nbrs = KMeans(n_clusters=i).fit(vectors)
        labels = nbrs.labels_
        sil.append(silhouette_score(vectors, labels, metric='euclidean'))

    # Doing this number of clusters
    optimal_num = sil.index(min(sil))
    nbrs = KMeans(n_clusters=optimal_num).fit(vectors)
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

