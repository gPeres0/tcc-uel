# import matplotlib as pl
# import pandas as pd
import numpy as np

import nolds
import ordpy
import antropy as ant
import EntropyHub as enhub
from scipy import stats

def lempel_ziv_complexity(data):
    threshold = np.mean(data)
    binary_data = (data >= threshold) * 1
    return ant.lziv_complexity(data, normalize=False)

def lyapunov_exponent(data):
    return nolds.lyap_r(data)

def shannon_entropy(data):
    return stats.entropy(data)

def permutation_entropy(data):
    pass

def multiscale_entropy():
    pass
