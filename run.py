from Processing.load_data import load_all_cc_data, load_all_epsp_data
from Analysis.clustering import do_tsne_on_ks, get_frequency_components
from Metrics.isi_analysis import do_isi_analysis
from Visualisation.trace_plots import plot_all_abf_data


abfobjects = load_all_cc_data() + load_all_epsp_data()

plot_all_abf_data(abf_objects=abfobjects)


# do_isi_analysis(abfobjects)


