import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


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


def plot_comparison_with_new(response_vectors):
    sfc = [vector[0] for vector in response_vectors]
    ifc = [vector[1] for vector in response_vectors]
    m = [vector[7] for vector in response_vectors]
    c = [vector[8] for vector in response_vectors]
    e = [vector[9] for vector in response_vectors]

    plt.scatter(ifc, m)
    plt.title("IFC vs M")
    plt.show()
    plt.title("IFC vs C")
    plt.scatter(ifc, c)
    plt.show()
    plt.title("IFC vs e")
    plt.scatter(ifc, e)
    plt.show()





