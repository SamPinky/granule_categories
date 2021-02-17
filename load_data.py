import matplotlib.pyplot as plt

import pyabf


file_location = "../../Granule-Data/"

example_location = "GrC_Subject06_090216/"

iv_file = "090216_0002  IV -70.abf"
cc_file = "090216_0003  CC step.abf"
epsp_file = "090216_0004 EPSP.abf"


data = pyabf.ABF(file_location + example_location + epsp_file)
for i in range(7):
    data.setSweep(i)
    plt.plot(data.sweepX, data.sweepY)
    plt.show()
