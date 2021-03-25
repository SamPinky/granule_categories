import numpy as np

from Processing.load_data import load_all_cc_data, load_all_epsp_data, load_data
from Processing.process_raw_trace import get_spike_times_for_cc
from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing
from Processing.data_checking import check_isi_normality, check_whole_rate_normality

from Analysis.clustering import do_tsne_on_ks, get_frequency_components, tsne_on_full_vector, do_tsne_on_kdfs, knn_full_response_vector, knn_on_ifc_initial, agglomerative_clustering_on_vectors, knn_on_strong_weak, knn_on_slow_fast_onset, knn_on_slow_fast_adapt_accel

from Metrics.isi_analysis import do_isi_analysis
from Metrics.masoli_metrics import do_masoli_analysis
from Metrics.my_metrics import compute_neuron_vectors, get_all_kdfs

from Visualisation.trace_plots import plot_all_abf_data
from Visualisation.rate_plots import plot_all_psth
from Visualisation.metrics_plots import plot_masoli_metrics, plot_metrics_against_clusters, plot_dendrogram


# Checking for normality in cc and epsp
# cc = load_all_cc_data()

# plot_all_abf_data(cc)
# epsp = load_all_epsp_data()
#
# check_whole_rate_normality(cc, True, True)
# check_whole_rate_normality(epsp, False, True)
#
# # Without cleanup
# check_whole_rate_normality(cc, True, False)
#
# check_isi_normality(cc, True, True)
# check_isi_normality(epsp, False, True)
#
# # Without cleanup cleanup steps
# check_isi_normality(cc, True, False)


# Clustering of metrics and kdfs, as well as original clustering.
data = load_all_cc_data()
vectors, neurons = compute_neuron_vectors(data, load_all_epsp_data())
neurons = list(neurons)


# metrics = ["sfc", "ifc", "f_initial", "Bfrac", "max", "mean", "m", "c", "tau"]

# np.savetxt("vectors.csv", vectors, delimiter=", ", header=", ".join(metrics))

#
labels1 = knn_on_ifc_initial(vectors)
plot_metrics_against_clusters(vectors, neurons, labels1, "KNN Adapt-Accel")

# labels2 = knn_on_strong_weak(vectors)
# plot_metrics_against_clusters(vectors, neurons, labels2, "KNN Strong vs Weak")
#
# labels3 = knn_on_slow_fast_onset(vectors)
# plot_metrics_against_clusters(vectors, neurons, labels3, "KNN Slow vs Fast Onset")
#
# labels4 = knn_on_slow_fast_adapt_accel(vectors)
# plot_metrics_against_clusters(vectors, neurons, labels4, "KNN Slow vs Fast Adapt-Accel")

# labels5 = knn_full_response_vector(vectors)
# plot_metrics_against_clusters(vectors, neurons, labels5, "KNN on Full Vector")

kdfs = get_all_kdfs(data, load_all_epsp_data())
do_tsne_on_kdfs(kdfs, neurons, labels1)

# aggl = agglomerative_clustering_on_vectors(vectors)
# labels_2 = aggl.labels_
#
# results_1 = do_tsne_on_vectors(vectors, neurons, labels)
# results_2 = do_tsne_on_vectors(vectors, neurons, labels_2)


# do_tsne_on_kdfs(raw_kdfs, neuron_names, labels)

# plot_dendrogram(aggl)
#
# plot_metrics_against_clusters(vectors, neurons, labels_2, "Agglomerative")
#

