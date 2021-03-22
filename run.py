from Processing.load_data import load_all_cc_data, load_all_epsp_data
from Analysis.clustering import do_tsne_on_ks, get_frequency_components, do_tsne_on_vectors
from Metrics.isi_analysis import do_isi_analysis
from Visualisation.trace_plots import plot_all_abf_data
from Visualisation.rate_plots import plot_all_psth
from Metrics.masoli_metrics import do_masoli_analysis
from Metrics.my_metrics import compute_neuron_vectors
from Visualisation.metrics_plots import plot_masoli_metrics, plot_comparison_with_new
import numpy as np


vectors, neurons = compute_neuron_vectors(load_all_cc_data(), load_all_epsp_data())
neurons = list(neurons)
np.savetxt("vectors.csv", vectors, delimiter=',')
plot_masoli_metrics(vectors)
# do_tsne_on_vectors(vectors, neurons)

