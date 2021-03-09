from Processing.load_data import load_all_cc_data
from Analysis.clustering import do_tsne_on_ks, get_frequency_components


abfobjects = load_all_cc_data() #+load_all_epsp_data()
# plot_all_abf_data(abfobjects)
freq_components = get_frequency_components(abfobjects)
do_tsne_on_ks(freq_components)


