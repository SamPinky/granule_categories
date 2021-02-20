import pyabf


def load_data(subject_path="GrC_Subject06_090216/", file_name="090216_0003 CC step.abf"):
    file_location = "../../Granule-Data/"
    data = pyabf.ABF(file_location + subject_path + file_name)
    return data

