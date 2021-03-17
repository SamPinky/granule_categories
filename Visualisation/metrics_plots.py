import matplotlib.pyplot as plt


def plot_masoli_metrics(response_vectors):
    sfc = [vector[0] for vector in response_vectors]
    ifc = [vector[1] for vector in response_vectors]
    f_initial = [vector[2] for vector in response_vectors]
    plt.scatter(f_initial, ifc)
    plt.title("IFC vs f_initial")
    plt.show()
    plt.title("SFC vs IFC")
    plt.scatter(ifc, sfc)
    plt.show()





