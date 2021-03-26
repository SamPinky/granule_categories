import numpy as np
import decimal
from lmfit.models import ParabolicModel
from sklearn.linear_model import LinearRegression


from Processing.process_raw_trace import get_spike_times_for_cc, get_spike_times_for_epsp
from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing


def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)


def calculate_sfc(abf_objs):
    if len(abf_objs) == 0:
        return 0
    fresp = []
    for obj in abf_objs:
        for sweep in range(obj.sweepCount):
            obj.setSweep(sweep)
            spikes = get_spike_times_for_epsp(obj)
            spikes = [spike for spike in spikes if 0.6 <= spike < 0.85]
            fresp.append(len(spikes)/0.25)
    fresp = np.mean(fresp)
    return ((fresp-50)/50) * 100


def calculate_ifc(abf_objs):
    if len(abf_objs) == 0:
        return None
    ifc = 0
    f_init = 0
    # At 10pa
    for obj in abf_objs:
        spikes = get_spike_times_for_cc(obj, 9)
        f_initial = len([spike for spike in spikes if spike <= 0.6]) / 0.5
        f_final = len([spike for spike in spikes if 1.6 < spike]) / 0.5
        if "Subject18" in obj.abfFolderPath.split("/")[-1]:
            x = True
        if f_initial > 0:
            ifc += ((f_final - f_initial) / f_initial) * 100
        else:
            ifc += 100
        f_init += f_initial
    return ifc/len(abf_objs), f_init/len(abf_objs)


def calculate_bfrac(times, kdf, B):
    B_area = B * (times[100:-100][-1] - times[100:-100][0])
    total_area = np.trapz(kdf)
    return B_area/total_area


def fit_quadratic(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    qmodel = ParabolicModel()
    result = qmodel.fit(y, x=x, a=1, b=2, c=0)
    e = 0
    for xi, yi in zip(x, y):
        p = qmodel.eval(result.params, x=xi)
        e += (yi - p) ** 2
    return result.params["a"].value, result.params["b"].value, result.params["c"].value, e/len(x)


def fit_linear(points):
    x = np.array([point[0] for point in points])
    x = x[:, None]
    y = [point[1] for point in points]
    model = LinearRegression().fit(x, y)
    e = model.score(x, y)
    m = model.coef_
    c = model.intercept_
    return m[0], c, e


def get_tau(kdf, x_d):
    peak = max(kdf)
    for point, t in zip(kdf, x_d):
        if point > peak * 2/3:
            return t
    print("Problem")
    return 1


def sort_objects_by_neuron(cc_objects, epsp_objects):
    subjects = set([obj.abfFolderPath.split("/")[-1] for obj in epsp_objects + cc_objects])
    neuron_names = []
    neurons = []
    for subject in subjects:
        subject_files = []
        for obj in cc_objects:
            if obj.abfFolderPath.split("/")[-1] == subject:
                subject_files.append(obj)
        for obj in epsp_objects:
            if obj.abfFolderPath.split("/")[-1] == subject:
                subject_files.append(obj)

        subject_neurons = []
        a_neurons = []
        b_neurons = []
        for file in subject_files:
            if "-A" not in file.abfFilePath.split("/")[-1] and "-B" not in file.abfFilePath.split("/")[-1]:
                subject_neurons.append(subject_files)
                break
            elif "-A" in file.abfFilePath.split("/")[-1]:
                a_neurons.append(file)
            elif "-B" in file.abfFilePath.split("/")[-1]:
                b_neurons.append(file)
            else:
                print("PROBLEM")
        if len(subject_neurons) > 0:
            neurons.append(subject_neurons)
            neuron_names.append(subject)
        if len(a_neurons) > 0:
            neurons.append(a_neurons)
            neuron_names.append(subject + "A")
        if len(b_neurons) > 0:
            neurons.append(b_neurons)
            neuron_names.append(subject + "B")

    new_neurons = []
    for neuron in neurons:
        all_files = []
        for item in neuron:
            if isinstance(item, list):
                all_files = all_files + item
            else:
                all_files.append(item)
        new_neurons.append(all_files)
    return neuron_names, new_neurons


def compute_neuron_vectors(cc_objects, epsp_objects):
    neuron_names, neurons = sort_objects_by_neuron(cc_objects, epsp_objects)
    vectors = []
    for n, neuron in enumerate(neurons):
        vector = []
        sub_cc = [obj for obj in neuron if "CC step" in obj.abfFilePath]
        sub_epsp = [obj for obj in neuron if "EPSP" in obj.abfFilePath]

        vector.append(calculate_sfc(sub_epsp))
        ifc, f_initial = calculate_ifc(sub_cc)
        vector.append(ifc)
        vector.append(f_initial)
        if len(sub_cc) > 0:
            B = 0
            B_frac = 0
            max_v = 0
            mean = 0
            median = 0
            m = 0
            c = 0
            e = 0
            tau = 0
            for obj in sub_cc:
                spikes = get_spike_times_for_cc(obj, 9)
                if len(spikes) == 0:
                    pass
                else:
                    obj.setSweep(9)
                    kdf = calculate_spike_rate_kernel_smoothing(spikes, max(obj.sweepX))
                    x_d = np.linspace(0, max(obj.sweepX), 1000)
                    indexes = range(1000)
                    B += min(kdf[100:-200])
                    B_frac += calculate_bfrac(x_d, kdf, B)
                    max_v += max(kdf[100:-100])
                    mean += np.mean(kdf[100:-100])
                    median += np.median(kdf[100:-100])

                    maxima = [[x, y] for i, x, y in zip(indexes, x_d, kdf) if kdf[i-1] < y > kdf[i+1]]
                    if len(maxima) > 1:
                        new_m, new_c, new_e = fit_linear(maxima)
                    else:
                        new_m, new_c, new_e = 0, 0, 0
                    m += new_m
                    c += new_c
                    e += new_e
                    tau += get_tau(kdf, x_d)

            # vector.append(B/len(sub_cc))
            vector.append(B_frac/len(sub_cc))
            vector.append(max_v/len(sub_cc))
            vector.append(mean/len(sub_cc))
            # vector.append(median/len(sub_cc))
            if (mean/len(sub_cc)) != 0:
                vector.append((m/len(sub_cc))/(mean/len(sub_cc)))
            else:
                vector.append(0)
            vector.append(c/len(sub_cc))
            # vector.append(e/len(sub_cc))
            vector.append(tau/len(sub_cc))
        else:
            vector.append(0)
            vector.append(0)
            vector.append(0)
            vector.append(0)
            vector.append(0)
            vector.append(0)
            # vector.append(0)
            # vector.append(0)
            # vector.append(0)
        vectors.append(vector)
    return np.array(vectors), neuron_names


def get_all_kdfs(cc_objects, epsp_objects):
    neuron_names, neurons = sort_objects_by_neuron(cc_objects, epsp_objects)
    kdfs = []
    for n, neuron in enumerate(neurons):
        kdf = []
        sub_cc = [obj for obj in neuron if "CC step" in obj.abfFilePath]

        if len(sub_cc) == 1:
            for obj in sub_cc:
                spikes = get_spike_times_for_cc(obj, 9)
                if len(spikes) == 0:
                    kdf = np.zeros(shape=(1000))
                else:
                    obj.setSweep(9)
                    kdf = calculate_spike_rate_kernel_smoothing(spikes, max(obj.sweepX))
        elif len(sub_cc) > 1:
            print("Problemo")
        else:
            print("Double problemo")
        kdfs.append(kdf)
    return kdfs
