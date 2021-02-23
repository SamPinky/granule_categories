import os
import pyabf


def load_data(subject_path, file_name):
    # file_location = "../../Granule-Data/"
    print(subject_path + "/" + file_name)
    data = pyabf.ABF(subject_path + "/" + file_name)
    return data


def load_all_epsp_data():
    root_dir = "../../Granule-Data/"
    abf_objects = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if "EPSP" in file:
                abf_objects.append(load_data(subdir, file))
    return abf_objects

# /home/samp/Granule-Data/GrC_Subject22_220118/22118_0003 EPSP.abf
