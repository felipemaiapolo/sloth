import numpy as np

def filter(s):
    try:s = s.split("/")[1]
    except: s = s
    try:s = s.split("__")[1]
    except: s = s
    return s.lower().replace("-hf","").replace("_","").replace("-","")

def get_true_indices(bool_array):
    return np.where(bool_array)[0]
    
