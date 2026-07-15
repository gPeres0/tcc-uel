import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import antropy as ant
import EntropyHub as enhub
import nolds
import ordpy
from scipy import stats


# ========== CONFIGURAÇÃO DOS DADOS ========== #

df_batmin_1d = pd.read_csv('../data/1day/batmin_1d.csv')
df_batmin_3d = pd.read_csv('../data/3days/batmin_3d.csv')
df_batmin_7d = pd.read_csv('../data/7days/batmin_7d.csv')

batmin_1d = df_batmin_1d['valor'].values
batmin_3d = df_batmin_3d['valor'].values
batmin_7d = df_batmin_7d['valor'].values

full_data = [batmin_1d, batmin_3d, batmin_7d]


# ============= FUNÇÕES DE TESTE ============= #

def lempelZivTest_Padrao(data):
    threshold = np.median(data)
    # print(f"LZ threshold: {threshold}")
    binary_data = (data >= threshold) * 1
    # print(binary_data)

    # print(f"LZ: {ant.lziv_complexity(binary_data, normalize=False)}")
    return ant.lziv_complexity(binary_data, normalize=True)

# ---

def lyapunovTest(data):
    # Kwargs mais realistas para 1500 data points (segundo o gemini)
    return nolds.lyap_r(data, emb_dim=3, lag=1) # Expoente de Lyapunov com algoritmo de Rosenstein et al

# ---

def shannonTest(data):
    """
    Shannon for continuous data: Discretization by bins (Binning) 
    This involves grouping continuous values into a finite number of discrete categories or "bins".
    How it works: Divide the entire amplitude range of your time series into equal-sized or 
    quantile-based bins. You then assign each data point to its corresponding bin, calculate the 
    probability (pi) of each bin, and apply the standard Shannon formula.
    """

    # 1. Criar um histograma para aproximar a Distribuição de Probabilidade
    # O parâmetro bins='auto' calcula automaticamente o número ideal de divisões
    hist, _ = np.histogram(data, bins='auto')
    print(f"bins: {len(hist)}")
    
    shan = stats.entropy(pk=hist)

    return shan, shan/np.log(len(hist))

# ---

def permEnTest(data):
    # PE AntroPy
    # print("== PE AntroPy ==")
    # print(f"PE: {ant.perm_entropy(data):.4}")                     # 2.5840
    # print(f"PE norm: {ant.perm_entropy(data, normalize=True):.4}") # 0.9996

    # PE EntropyHub
    # Perm, Pnorm, cPE = enhub.PermEn(data) # Norm=True gera erro, pois usa np.math (obsoleto) 
    # print("== PE ENtropyHub ==")
    # print(f"Perm: {Perm}")
    # print(f"Pnorm: {Pnorm}")
    # print(f"cPE: {cPE}") 

    # PE Ordpy
    # print("== PE Ordpy ==")
    # print(f"PE: {ordpy.permutation_entropy(data, normalized=False):.4}") # 1.7910
    return ordpy.permutation_entropy(data)

# ---

def mseTest(data):
    """
    To compare datasets with different baseline amplitudes or to eliminate arbitrary upper bounds, 
    researchers utilize normalized indices: 
    - Normalized Complexity Index (NCI): Often applied in symbolic dynamics and heart rate 
    variability (HRV), the NCI divides the raw CI by the maximum possible entropy (like Shannon Entropy) 
    to yield a scale-independent value between 0 (perfectly regular) and 1 (maximum irregularity).
    
    https://hal.science/hal-01392073/document
    """

    Mobj = enhub.MSobject(EnType='SampEn', m=2, r=0.2) # Estudar kwargs
    return enhub.MSEn(Sig=data, Mbjx=Mobj, Scales=5) 
    

# =========== EXECUÇÃO DOS TESTES =========== #

def run_all_tests(full_data):
    i = 1
    for data in full_data:
        if i == 1:
            data_type = 'batmin_1d'
        elif i == 2:
            data_type = 'batmin_3d'
        else:
            data_type = 'batmin_7d'

        print(f"\n===== Testing with {data_type} =====\n")
        # print("--- Lempel-Ziv Complexity Regular Norm ---")
        # LZp = lempelZivTest_Padrao(data)
        # print(f"LZp norm: {LZp}")

        # print("--- Lyapunov Exponent ---")
        # lyap_r = lyapunovTest(data)
        # print(f"LYAP: {lyap_r:.4}")

        # print("--- Shannon Entropy ---")
        # shan, shan_norm = shannonTest(data)
        # print(f"SHAN: {shan}")
        # print(f"SHAN norm: {shan_norm}")

        print("--- Permutation Entropy ---")
        PE = permEnTest(data)
        print(f"PE norm: {PE:.4}")

        # print("--- Multiscale Entropy ---")
        # MSx, Ci = mseTest(data)
        # print(f"MSE: {MSx}")
        # # Ci (complexity index) é definido como a soma cumulativa (área) das complexidades (curva). 
        # print(f"Ci: {Ci:.4}") 

        i+=1

run_all_tests(full_data)
