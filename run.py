from Processing.load_data import load_all_cc_data, load_all_epsp_data, load_data
from Processing.process_raw_trace import get_spike_times_for_cc
from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing
from Analysis.clustering import do_tsne_on_ks, get_frequency_components, do_tsne_on_vectors, do_tsne_on_kdfs, knn_on_metrics, agglomerative_clustering_on_vectors
from Metrics.isi_analysis import do_isi_analysis
from Visualisation.trace_plots import plot_all_abf_data
from Visualisation.rate_plots import plot_all_psth
from Metrics.masoli_metrics import do_masoli_analysis
from Metrics.my_metrics import compute_neuron_vectors
from Visualisation.metrics_plots import plot_masoli_metrics, plot_comparison_with_new, plot_metrics_against_clusters, plot_dendrogram
import numpy as np

data = load_all_cc_data()
vectors, neurons = compute_neuron_vectors(data, load_all_epsp_data())
neurons = list(neurons)
#
#
labels = knn_on_metrics(vectors)
# aggl = agglomerative_clustering_on_vectors(vectors)
# labels_2 = aggl.labels_
#
# results_1 = do_tsne_on_vectors(vectors, neurons, labels)
# results_2 = do_tsne_on_vectors(vectors, neurons, labels_2)

neuron_names = set([obj.abfFolderPath.split("/")[-1] for obj in data])

new_data = []
prev = []
for da in data:
    if da.abfFolderPath in prev:
        pass
    else:
        prev.append(da.abfFolderPath)
        new_data.append(da)

spike_times = [get_spike_times_for_cc(da, 8) for da in new_data]
raw_kdfs = [calculate_spike_rate_kernel_smoothing(spikes) for spikes in spike_times]
do_tsne_on_kdfs(raw_kdfs, neuron_names, labels)

# plot_dendrogram(aggl)
#
# plot_metrics_against_clusters(vectors, neurons, labels, "KNN")
# plot_metrics_against_clusters(vectors, neurons, labels_2, "Agglomerative")
#

