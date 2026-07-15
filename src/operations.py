import math
import numpy as np
import pandas as pd
from scipy import stats

import nolds
import ordpy
import antropy as ant
import EntropyHub as enhub


def lempel_ziv_complexity(data):
    threshold = np.mean(data)
    binary_data = (data >= threshold) * 1
    return ant.lziv_complexity(binary_data, normalize=True)


def lyapunov_exponent(data):
    return nolds.lyap_r(data, emb_dim=10, lag=1) # Estudar e verificar parâmetros


def shannon_entropy(data):
    hist, _ = np.histogram(data, bins='auto')
    
    shan = stats.entropy(hist)
    shan_norm = shan/np.log(len(hist))
    
    return shan, shan_norm, len(hist)


def permutation_entropy(data):
    D = 3
    # Para a PE, a literatura recomenda uma embedding dimension 'D' tal que N >= 5*D!.
    for _ in range(4):
        if len(data) >= 5*math.factorial(D):
            D+=1
        else:
            D-=1
            break

    return ordpy.permutation_entropy(data, dx=D, taux=1, normalized=True)


def multiscale_entropy(data, entype='SampEn'):
    # Para a SampEn, a literatura recomenda um 'm' tal que 10^m < N < 30^m.
    m = 1
    if len(data) > 450:
        m = 2 
    
    Mobj = enhub.MSobject(EnType=entype, m=m) # r = 0.2*np.std(Sig)

    # A literatura diz que a série granulada não deve ficar com
    # menos de 50 a 100 pontos. Portanto, definido que N/tau > 75.  
    tau = 2
    for i in range(10):
        if len(data)/(i+1) > 75:
            tau+=1
        else:
            tau-=1
            break
    
    return enhub.MSEn(Sig=data, Mbjx=Mobj, Scales=tau)
