import numpy as np
import antropy as ant
import nolds
from scipy import stats

data = np.array([0.2, 0.7, 0.4, 0.9, 0.1, 0.5, 1, 8, 16, 12, 0.9, 0.7, 20, 7, 3, 25, 70, 5])

def lempelZivTest(data):
    # Para data = [0.2, 0.7, 0.4, 0.9, 0.1, 0.5, 1, 8, 16, 12, 0.9, 0.7]
    threshold = np.mean(data)
    print(threshold) # 3.4499
    binary_data = (data >= threshold) * 1
    print(binary_data) # [0 0 0 0 0 0 0 1 1 1 0 0]
    print(ant.lziv_complexity(binary_data, normalize=False)) # 4
    print(ant.lziv_complexity(binary_data, normalize=True))  # 1.1949

def permEnTest(data):
    # PE AntroPy
    print('PE AntroPy: ' + ant.perm_entropy(data)) # 2.4464
    print('PE AntroPy (norm): ' + ant.perm_entropy(data, normalize=True)) # 0.9464

    # PE EntropyHub

def lyapunovTest(data):
    # Kwargs definidos APENAS para teste da biblioteca
    lyap_r = nolds.lyap_r(data, emb_dim=3, min_tsep=1, trajectory_len=4) # Expoente de Lyapunov com algoritmo de Rosenstein et al
    lyap_e = nolds.lyap_e(data, emb_dim=2, matrix_dim=2, min_nb=2, min_tsep=1) # Expoente de Lyapunov com algoritmo de Eckman et al
    print(lyap_r) # 0.7584
    print(lyap_e) # [ 1.89774988 -0.19483742]

def shannonTest(data):
    # Verificar se precisa discretizar / se tem muita diferença no resultado
    print(stats.entropy(data)) # 1.9257

def mseTest(data):
    pass

# lempelZivTest(data)
# permEnTest(data)
# lyapunovTest(data)
# shannonTest(data)
mseTest(data)