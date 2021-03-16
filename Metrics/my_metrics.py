import numpy as np
import decimal
from lmfit.models import ParabolicModel


from Processing.process_raw_trace import get_spike_times_for_cc, get_spike_times_for_epsp
from Processing.calculate_spike_rate import calculate_spike_rate_kernel_smoothing


def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)


def calculate_sfc(abf_objs):
    if len(abf_objs) == 0:
        return None
    fresp = []
    for obj in abf_objs:
        for sweep in range(obj.sweepCount):
            obj.setSweep(sweep)
            spikes = get_spike_times_for_epsp(obj)
            spikes = [spike for spike in spikes if 0.5 <= spike < 0.75]
            fresp.append(len(spikes)/0.25)
    fresp = np.mean(fresp)
    return (fresp-50)/50


def calculate_ifc(abf_objs):
    if len(abf_objs) == 0:
        return None
    ifc = 0
    # At 10pa
    for obj in abf_objs:
        spikes = get_spike_times_for_cc(obj, 9)
        f_initial = len([spike for spike in spikes if spike <= 0.5]) / 0.5
        f_final = len([spike for spike in spikes if 1.5 < spike]) / 0.5
        if f_initial > 0:
            ifc += (f_final - f_initial) / f_initial
        else:
            ifc += f_final
    return ifc/len(abf_objs)


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


def compute_neuron_vectors(cc_objects, epsp_objects):
    neuron_names = set([obj.abfFolderPath.split("/")[-1] for obj in epsp_objects + cc_objects])
    vectors = []
    for neuron in neuron_names:
        vector = []
        sub_cc = [obj for obj in cc_objects if neuron in obj.abfFolderPath]
        sub_epsp = [obj for obj in cc_objects if neuron in obj.abfFolderPath]

        vector.append(calculate_sfc(sub_epsp))
        vector.append(calculate_ifc(sub_cc))
        if len(sub_cc) > 0:
            B = 0
            B_frac = 0
            max_v = 0
            mean = 0
            median = 0
            a = 0
            b = 0
            c = 0
            e = 0
            for obj in sub_cc:
                spikes = get_spike_times_for_cc(obj, 9)
                if len(spikes) == 0:
                    pass
                else:
                    obj.setSweep(9)
                    kdf = calculate_spike_rate_kernel_smoothing(spikes)
                    x_d = np.linspace(0, max(obj.sweepX), 1000)
                    indexes = range(1000)

                    B += min(kdf[100:-100])
                    B_frac += calculate_bfrac(x_d, kdf, min(kdf[100:-100]))
                    max_v += max(kdf[100:-100])
                    mean += np.mean(kdf[100:-100])
                    median += np.median(kdf[100:-100])

                    maxima = [[x, y] for i, x, y in zip(indexes, x_d, kdf) if kdf[i-1] < y > kdf[i+1]]
                    if len(maxima) > 2:
                        new_a, new_b, new_c, new_e = fit_quadratic(maxima)
                    else:
                        new_a, new_b, new_c, new_e = 0, 0, 0, 0
                    a += new_a
                    b += new_b
                    c += new_c
                    e += new_e
            vector.append(B/len(sub_cc))
            vector.append(B_frac/len(sub_cc))
            vector.append(max_v/len(sub_cc))
            vector.append(mean/len(sub_cc))
            vector.append(a/len(sub_cc))
            vector.append(b/len(sub_cc))
            vector.append(c/len(sub_cc))
            vector.append(e/len(sub_cc))
        else:
            vector.append(None)
            vector.append(None)
            vector.append(None)
            vector.append(None)
            vector.append(None)
            vector.append(None)
            vector.append(None)
            vector.append(None)
        vectors.append(vector)
        # TODO: Normalise vector
    return np.array(vectors)
