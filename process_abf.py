

def get_all_sweep_data(abf_object):
    sweep_data = []
    for i in range(abf_object.sweepCount):
        abf_object.setSweep(i)
        sweep_data.append([abf_object.sweepX, abf_object.sweepY])
    return sweep_data


