"""
Roda todas as funções de complexidade/entropia definidas em operations.py
sobre cada um dos sensores, para os 3 períodos de coleta (1, 3 e 7 dias),
e salva os resultados em um único CSV.

Estrutura de pastas esperada (irmã da pasta onde este script está):

    data/
        1day/   *_1d.csv
        3days/  *_3d.csv
        7days/  *_7d.csv

Cada CSV tem as colunas: cidade, id_cidade, sensor, data, valor
"""

import os
import glob
import numpy as np
import pandas as pd

from operations import (
    lempel_ziv_complexity,
    lyapunov_exponent,
    shannon_entropy,
    permutation_entropy,
    multiscale_entropy,
)

# --- Configuração -----------------------------------------------------

DATA_DIR = '../data'
OUTPUT_CSV = '../data/resultados_entropia.csv'

# (nome da pasta, sufixo do arquivo, número de dias)
PERIODOS = [
    ('1day', '_1d', 1),
    ('3days', '_3d', 3),
    ('7days', '_7d', 7),
]


# --- Utilitários --------------------------------------------------------

def nome_sensor(caminho_csv, sufixo):
    """Extrai o nome do sensor a partir do nome do arquivo, removendo o sufixo do período."""
    base = os.path.splitext(os.path.basename(caminho_csv))[0]
    if base.endswith(sufixo):
        base = base[: -len(sufixo)]
    return base


def rodar_com_seguranca(func, *args, **kwargs):
    """Executa func e devolve (resultado, None) ou (None, mensagem_de_erro) sem derrubar o script."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# --- Execução principal ---------------------------------------------------

def main():
    resultados = []

    for pasta, sufixo, n_dias in PERIODOS:
        pasta_completa = os.path.join(DATA_DIR, pasta)
        arquivos_csv = sorted(glob.glob(os.path.join(pasta_completa, '*.csv')))

        if not arquivos_csv:
            print(f"[aviso] Nenhum CSV encontrado em: {pasta_completa}")
            continue

        for caminho_csv in arquivos_csv:
            sensor = nome_sensor(caminho_csv, sufixo)
            print(f"\n=== {sensor} | {n_dias} dia(s) ===")

            df = pd.read_csv(caminho_csv)
            data = df['valor'].dropna().values
            n_pontos = len(data)
            print(f"  n_pontos: {n_pontos}")

            linha = {
                'sensor': sensor,
                'dias': n_dias,
                'n_pontos': n_pontos,
                'lempelziv': np.nan,
                # 'lyapunov': np.nan,
                'shannon': np.nan,
                'shannon_bins': np.nan,
                'perm': np.nan,
                'mse_ci': np.nan,
                'mse_sampen': np.nan,
            }

            # Lempel-Ziv
            res, err = rodar_com_seguranca(lempel_ziv_complexity, data)
            if err:
                print(f"  [lempelziv] ERRO: {err}")
            else:
                linha['lempelziv'] = round(res,5)

            # # Lyapunov
            # res, err = rodar_com_seguranca(lyapunov_exponent, data)
            # if err:
            #     print(f"  [lyapunov] ERRO: {err}")
            # else:
            #     linha['lyapunov'] = res

            # Shannon (usa a versão normalizada)
            res, err = rodar_com_seguranca(shannon_entropy, data)
            if err:
                print(f"  [shannon] ERRO: {err}")
            else:
                _, shan_norm, bins = res
                linha['shannon'] = round(shan_norm,5)
                linha['shannon_bins'] = bins

            # Permutation entropy
            res, err = rodar_com_seguranca(permutation_entropy, data)
            if err:
                print(f"  [perm] ERRO: {err}")
            else:
                linha['perm'] = round(res,5)

            # Multiscale entropy (usa o Complexity Index / CI como resumo)
            res, err = rodar_com_seguranca(multiscale_entropy, data)
            if err:
                print(f"  [mse] ERRO: {err}")
            else:
                msx, ci = res
                linha['mse_ci'] = round(ci,5)
                linha['mse_sampen'] = msx

            print(f"  resultado: {linha}")
            resultados.append(linha)

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResultados salvos em: {os.path.abspath(OUTPUT_CSV)}")

    return df_resultados


if __name__ == '__main__':
    main()