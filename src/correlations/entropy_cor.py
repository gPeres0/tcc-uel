"""
Correlação entre métricas de complexidade/entropia e previsibilidade (WAPE)
============================================================================

Calcula correlação de Spearman e Kendall-tau entre cada métrica de
complexidade/entropia e o WAPE do AutoARIMA, separadamente para cada
janela temporal (1, 3 e 7 dias) — usando os 10 sensores como os pontos
da correlação (n=10 por janela/métrica). Aplica correção de múltiplas
comparações via FDR (Benjamini-Hochberg).

Requisitos: pandas, scipy, numpy
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

# =============================================================================
# CONFIG — ajuste conforme a estrutura real dos seus dados
# =============================================================================

# Arquivos de entrada
ENTROPY_PATH = "../../data/resultados_entropia.csv"
ARIMA_PATH = "../arima/results/resultados_autoarima.csv"

# Nome da coluna que identifica a janela temporal (em dias) após o merge
COL_WINDOW = "dias"

# Nome da coluna que identifica o sensor
COL_SENSOR = "sensor"

# Nome da coluna de WAPE (já convertida para float, sem '%')
COL_WAPE = "wape"

# Nomes das colunas de métricas de complexidade/entropia a testar
ENTROPY_COLS = [
    "lempelziv",
    "shannon",
    "perm",
    "mse_ci",
]

# Nível de significância para a correção FDR
ALPHA = 0.05

# Caminho de saída
OUTPUT_PATH = "results/correlacoes_entropia_wape.csv"


def load_and_merge_data() -> pd.DataFrame:
    """
    Carrega os arquivos de entropia e de AutoARIMA, normaliza os campos
    necessários e faz o merge por (sensor, dias).
    """
    entropia = pd.read_csv(ENTROPY_PATH)

    arima = pd.read_csv(ARIMA_PATH)
    # "1 dia" / "3 dias" / "7 dias" -> 1 / 3 / 7
    arima["dias"] = arima["period"].str.extract(r"(\d+)").astype(int)
    # "0.896%" -> 0.896 (mantido em %, não dividido por 100, para
    # preservar a mesma unidade usada nos relatórios)
    arima["wape"] = arima["wape"].astype(str).str.rstrip("%").astype(float)

    merged = entropia.merge(
        arima[["sensor", "dias", "mae", "mse", "wape"]],
        on=["sensor", "dias"],
        how="inner",
    )

    return merged

# =============================================================================
# Funções auxiliares
# =============================================================================


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Aplica a correção de Benjamini-Hochberg (FDR) em um array de p-valores.
    Retorna um array booleano indicando quais hipóteses são rejeitadas
    (i.e., permanecem significativas após a correção).
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked_p = pvalues[order]

    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked_p <= thresholds

    # Maior índice onde p_(i) <= (i/n)*alpha define o corte
    if not below.any():
        reject_sorted = np.zeros(n, dtype=bool)
    else:
        max_idx = np.max(np.where(below))
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[: max_idx + 1] = True

    reject = np.empty(n, dtype=bool)
    reject[order] = reject_sorted
    return reject


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Spearman e Kendall entre cada métrica de entropia e o WAPE,
    separadamente para cada JANELA TEMPORAL — usando os 10 sensores
    como os pontos da correlação.
    """
    results = []

    for window in sorted(df[COL_WINDOW].unique()):
        subset = df[df[COL_WINDOW] == window]

        for metric in ENTROPY_COLS:
            valid = subset[[metric, COL_WAPE]].dropna()
            n = len(valid)

            if n < 3:
                # Amostra insuficiente para correlação confiável
                results.append(
                    {
                        "window_days": window,
                        "metric": metric,
                        "n": n,
                        "spearman_rho": np.nan,
                        "spearman_p": np.nan,
                        "kendall_tau": np.nan,
                        "kendall_p": np.nan,
                    }
                )
                continue

            rho, p_spearman = spearmanr(valid[metric], valid[COL_WAPE])
            tau, p_kendall = kendalltau(valid[metric], valid[COL_WAPE])

            results.append(
                {
                    "window_days": window,
                    "metric": metric,
                    "n": n,
                    "spearman_rho": round(rho,5),
                    "spearman_p": round(p_spearman,5),
                    "kendall_tau": round(tau,5),
                    "kendall_p": round(p_kendall,5),
                }
            )

    results_df = pd.DataFrame(results)

    # Correção de múltiplas comparações (FDR), aplicada separadamente
    # para Spearman e para Kendall, sobre todos os testes do dataframe
    for col_p, col_flag in [
        ("spearman_p", "spearman_significant_fdr"),
        ("kendall_p", "kendall_significant_fdr"),
    ]:
        mask_valid = results_df[col_p].notna()
        results_df[col_flag] = False
        if mask_valid.any():
            reject = benjamini_hochberg(
                results_df.loc[mask_valid, col_p].values, alpha=ALPHA
            )
            results_df.loc[mask_valid, col_flag] = reject

    return results_df


def interpret_strength(rho: float) -> str:
    """Classificação qualitativa aproximada da força da correlação."""
    r = abs(rho)
    if pd.isna(rho):
        return "n/a"
    if r < 0.1:
        return "desprezível"
    elif r < 0.3:
        return "fraca"
    elif r < 0.5:
        return "moderada"
    elif r < 0.7:
        return "forte"
    else:
        return "muito forte"


# =============================================================================
# Execução principal
# =============================================================================

if __name__ == "__main__":
    df = load_and_merge_data()

    missing = [
        c
        for c in [COL_WINDOW, COL_SENSOR, COL_WAPE] + ENTROPY_COLS
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Colunas ausentes no arquivo de entrada: {missing}. "
            "Ajuste a seção CONFIG no topo do script."
        )

    results_df = compute_correlations(df)
    results_df["spearman_strength"] = results_df["spearman_rho"].apply(
        interpret_strength
    )
    results_df["kendall_strength"] = results_df["kendall_tau"].apply(
        interpret_strength
    )

    results_df = results_df.sort_values(["window_days", "metric"]).reset_index(
        drop=True
    )

    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH}")