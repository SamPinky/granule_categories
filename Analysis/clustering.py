import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering

from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing
from Processing.process_raw_trace import get_spike_times_for_cc


def get_frequency_components(abf_objects):
    freq_components = []
    indexes = [i for i in range(0, 1000, 10)]
    for abf_obj in abf_objects:
        for sweep in range(abf_obj.sweepCount):
            abf_obj.setSweep(sweep)
            spike_times = get_spike_times_for_cc(abf_obj)
            if len(spike_times) > 1:
                kds_data = calculate_spike_rate_kernel_smoothing(spike_times)
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


def do_tsne_on_vectors(vectors, neuron_names, labels=None):
    tsne = TSNE(n_components=2, n_iter=1000, perplexity=3)
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

    # for i, neuron in enumerate(neuron_names):
    #     p1.text(tpd['tsne-2d-one'][i] + 0.01, tpd['tsne-2d-two'][i],
    #             neuron, horizontalalignment='left',
    #             size='small', color='black', weight='bold')
    plt.show()
    return tpd


def do_tsne_on_kdfs(kdfs, neuron_names, labels=None):
    tsne = TSNE(n_components=2, n_iter=400, perplexity=5)
    tsne_results = tsne.fit_transform(kdfs)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    plt.title(f"tsne results on KDFs")

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
    plt.show()


def knn_on_metrics(vectors):
    ifc = [vector[1] for vector in vectors]
    finitial = [vector[2] for vector in vectors]
    nbrs = KMeans(n_clusters=4).fit([[f, ifc_v] for f, ifc_v in zip(finitial, ifc)])
    labels = nbrs.labels_
    return labels


def agglomerative_clustering_on_vectors(response_vectors):
    ifc = [vector[1] for vector in response_vectors]
    finitial = [vector[2] for vector in response_vectors]
    data = [[fin, ifcc] for fin, ifcc in zip(finitial, ifc)]
    aggl = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data)
    return aggl
