import numpy as np

from Processing.load_data import load_all_cc_data, load_all_epsp_data, load_data
from Processing.data_checking import check_metric_normality

from Analysis.clustering import tsne_on_full_vector, all_clusters, do_tsne_on_kdfs, knn_on_ifc_initial, knn_on_strong_weak, knn_on_slow_fast_onset, knn_on_slow_fast_adapt_accel
from Analysis.statistics import do_anova_on_groups, do_kruskal_wallis_test

from Metrics.my_metrics import compute_neuron_vectors, get_all_kdfs

from Visualisation.figures import figure_1, figure_2, figure_3, figure_6, figure_7
from Visualisation.rate_plots import plot_all_psth


# Creating Figures
cc = load_data("../../Granule-Data/GrC_Subject15_180116/", "180116_0005 CC step.abf")
cc2 = load_data("../../Granule-Data/GrC_Subject01_010818/", "010818_0002 CC step.abf")
cc3 = load_data("../../Granule-Data/GrC_Subject23_250116/", "250116-A_0001 CC step.abf")
epsp = load_data("../../Granule-Data/GrC_Subject14_161115/", "161115_0007 EPSP.abf")
figure_1(cc2, epsp)
figure_2([cc, cc2], epsp)
figure_3(cc3)

# Plot all KDFs
plot_all_psth(load_all_cc_data(), False)

# Clustering
data = load_all_cc_data()
vectors, neurons = compute_neuron_vectors(data, load_all_epsp_data())
clusters = all_clusters(vectors, neurons)
clusters.to_csv("clusters.csv")  # Save to CSV
labels = [clusters.iloc[0, i] for i in range(len(clusters.columns))]
new_clusters = [str(clusters.iloc[0, i]) + str(clusters.iloc[1, i]) + str(clusters.iloc[2, i]) for i in range(len(clusters.columns))]

# Figures
figure_6(vectors, clusters)
figure_7(vectors, new_clusters)

# T-SNE
tsne_on_full_vector(vectors, None, labels, new_clusters)
kdfs = get_all_kdfs(data, load_all_epsp_data())
do_tsne_on_kdfs(kdfs, None, labels)


# STATISTICAL TESTS

# Check the normality of metrics
check_metric_normality(vectors)

# Adapt-accelerate
labels = knn_on_ifc_initial(vectors)
print("Adapt-accelerate")
vectors1 = np.array([vector[1] for vector in vectors])
do_anova_on_groups(vectors1, labels)
do_kruskal_wallis_test(vectors1, labels)

# Strong-weak
labels = knn_on_strong_weak(vectors)
print("Strong-weak")
vectors2 = np.array([vector[5] for vector in vectors])
do_anova_on_groups(vectors2, labels)
do_kruskal_wallis_test(vectors2, labels)

# Slow-fast-onset
labels = knn_on_slow_fast_onset(vectors)
print("Slow-fast-onset")
vectors3 = np.array([vector[6] for vector in vectors])
do_anova_on_groups(vectors3, labels)
do_kruskal_wallis_test(vectors3, labels)

# Adapt-accel effects onset
labels = knn_on_slow_fast_adapt_accel(vectors)
print("slow-fast effects")
vectors4 = np.array([vector[6] for vector in vectors])
do_anova_on_groups(vectors4, labels)
do_kruskal_wallis_test(vectors4, labels)
