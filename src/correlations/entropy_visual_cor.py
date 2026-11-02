import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

from pathlib import Path

from src.correlations.entropy_wape_cor import interpret_strength

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ENTROPY_PATH = DATA_DIR / "resultados_entropia.csv"

VISUAL_PATH = DATA_DIR / "data_separation_sheet.csv"

# Nome da coluna que identifica a janela temporal (em dias) após o merge
COL_WINDOW = "dias"

# Nome da coluna que identifica o sensor
COL_SENSOR = "sensor"

COL_VISUAL = ["visual_gabriel", "visual_bruno"]

# Nomes das colunas de métricas de complexidade/entropia a testar
ENTROPY_COLS = [
    "lempelziv",
    "lyapunov",
    "shannon",
    "perm",
    "mse_ci"
]

OUTPUT_PATH = DATA_DIR / f"correlations_results/correlacoes_entropia_visual_bruno.csv"


def load_and_merge_data() -> pd.DataFrame:
    entropia = pd.read_csv(ENTROPY_PATH)

    visual = pd.read_csv(VISUAL_PATH)
    # "1 dia" / "3 dias" / "7 dias" -> 1 / 3 / 7
    visual["dias"] = visual["Período Total"].str.extract(r"(\d+)").astype(int)

    merged = entropia.merge(
        visual[["sensor", "dias", "visual_gabriel", "visual_bruno"]],
        on=["sensor", "dias"],
        how="inner",
    )

    return merged


def compute_correlations(df: pd.DataFrame, col_visual) -> pd.DataFrame:
    results = []

    for window in sorted(df[COL_WINDOW].unique()):
        subset = df[df[COL_WINDOW] == window]

        for metric in ENTROPY_COLS:
            valid = subset[[metric, col_visual]].dropna()
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

            rho, p_spearman = spearmanr(valid[metric], valid[col_visual])
            tau, p_kendall = kendalltau(valid[metric], valid[col_visual])

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
    
    return results_df


if __name__ == "__main__":
    col_visual = COL_VISUAL[1]
    
    df = load_and_merge_data()

    missing = [
        c
        for c in [COL_WINDOW, COL_SENSOR, col_visual] + ENTROPY_COLS
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Colunas ausentes no arquivo de entrada: {missing}. "
            "Ajuste a seção CONFIG no topo do script."
        )

    results_df = compute_correlations(df, col_visual)
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