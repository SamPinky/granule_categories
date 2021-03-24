from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np


def plot_masoli_metrics(response_vectors):
    sfc = [vector[0] for vector in response_vectors]
    ifc = [vector[1] for vector in response_vectors]
    f_initial = [vector[2] for vector in response_vectors]
    nbrs = KMeans(n_clusters=4).fit([[f, ifc_v] for f, ifc_v in zip(f_initial, ifc)])
    labels = nbrs.labels_

    sns.scatterplot(f_initial, ifc, palette=sns.color_palette("hls",4), hue=labels)

    plt.title("IFC vs f_initial")
    plt.show()
    plt.title("SFC vs IFC")
    sns.scatterplot(ifc, sfc, palette=sns.color_palette("hls",4), hue=labels)
    plt.show()


def plot_metrics_against_clusters(response_vectors, neurons, labels, clustering_type):
    sfc = [vector[0] for vector in response_vectors]
    ifc = [vector[1] for vector in response_vectors]
    f_initial = [vector[2] for vector in response_vectors]
    B = [vector[3] for vector in response_vectors]
    B_frac = [vector[4] for vector in response_vectors]
    max = [vector[5] for vector in response_vectors]
    mean = [vector[6] for vector in response_vectors]
    median = [vector[7] for vector in response_vectors]
    m = [vector[8] for vector in response_vectors]
    c = [vector[9] for vector in response_vectors]
    e = [vector[10] for vector in response_vectors]
    tau = [vector[11] for vector in response_vectors]

    new_m = []
    for v in m:
        if type(v) is float:
            new_m.append(v)
        else:
            new_m.append(v[0])
    m = new_m

    fig, axs = plt.subplots(4, 3, sharex=False)
    fig.suptitle(f"Clustering type: {clustering_type}", fontsize=30)
    fig.set_size_inches(15, 20)

    axs[0, 0].set_title("IFC vs sfc", size=20)
    sns.scatterplot(ax=axs[0, 0], x=sfc, y=ifc, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[0, 1].set_title("IFC vs f_initial", size=20)
    sns.scatterplot(ax=axs[0, 1], x=f_initial, y=ifc, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[0, 2].set_title("IFC vs c", size=20)
    sns.scatterplot(ax=axs[0, 2], x=c, y=ifc, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[1, 0].set_title("IFC vs mean", size=20)
    sns.scatterplot(ax=axs[1, 0], x=mean, y=ifc, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[1, 1].set_title("IFC vs bfrac", size=20)
    sns.scatterplot(ax=axs[1, 1], x=B_frac, y=ifc, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[1, 2].set_title("IFC vs m", size=20)
    sns.scatterplot(ax=axs[1, 2], x=m, y=ifc, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[2, 0].set_title("m vs c", size=20)
    sns.scatterplot(ax=axs[2, 0], x=m, y=c, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[2, 1].set_title("m vs bfrac", size=20)
    sns.scatterplot(ax=axs[2, 1], x=m, y=B_frac, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[2, 2].set_title("mean vs bfrac", size=20)
    sns.scatterplot(ax=axs[2, 2], x=mean, y=B_frac, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[3, 0].set_title("IFC vs Tau", size=20)
    sns.scatterplot(ax=axs[3, 0], x=ifc, y=tau, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[3, 1].set_title("m vs Tau", size=20)
    sns.scatterplot(ax=axs[3, 1], x=m, y=tau, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    axs[3, 2].set_title("Bfrac vs Tau", size=20)
    sns.scatterplot(ax=axs[3, 2], x=B_frac, y=tau, palette=sns.color_palette("hls", len(set(labels))), hue=labels)

    plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)




