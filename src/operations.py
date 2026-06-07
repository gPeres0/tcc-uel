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

def lyapunov_exponent():
    pass

def shannon_entropy():
    pass

def permutation_entropy():
    pass

def multiscale_entropy():
    pass

def calculate_score():
    pass
