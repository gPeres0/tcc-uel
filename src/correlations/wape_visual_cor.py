import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

from pathlib import Path

from src.correlations.entropy_wape_cor import interpret_strength


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ARIMA_PATH = DATA_DIR / "arima_results/resultados_autoarima.csv"

LSTM_PATH = DATA_DIR / "lstm_results/resultados_lstm.csv"

VISUAL_PATH = DATA_DIR / "data_separation_sheet.csv"

# Nome da coluna que identifica a janela temporal (em dias) após o merge
COL_WINDOW = "dias"

# Nome da coluna que identifica o sensor
COL_SENSOR = "sensor"

COL_VISUAL = [
    "visual_gabriel", 
    "visual_bruno"
]

WAPE_COLS = [
    "wape_arima",
    "wape_lstm"
]

OUTPUT_PATH = [
    DATA_DIR / f"correlations_results/correlacoes_wape_visual_gabriel.csv",
    DATA_DIR / f"correlations_results/correlacoes_wape_visual_bruno.csv"
]


def load_and_merge_data() -> pd.DataFrame:
    """
    Carrega os arquivos de entropia e do modelo, normaliza os campos
    necessários e faz o merge por (sensor, dias).
    """
    arima = pd.read_csv(ARIMA_PATH)
    lstm = pd.read_csv(LSTM_PATH)
    visual = pd.read_csv(VISUAL_PATH)

    visual["dias"] = visual["Período Total"].str.extract(r"(\d+)").astype(int)
    
    arima["dias"] = arima["period"].str.extract(r"(\d+)").astype(int)
    lstm["dias"] = lstm["period"].str.extract(r"(\d+)").astype(int)

    arima["wape_arima"] = arima["wape"].astype(str).str.rstrip("%").astype(float)
    lstm["wape_lstm"] = lstm["wape"].astype(str).str.rstrip("%").astype(float)

    merged = arima.merge(
        visual[["sensor", "dias", "visual_gabriel", "visual_bruno"]],
        on=["sensor", "dias"],
        how="inner",
    )

    merged = merged.merge(
        lstm[["sensor", "dias", "wape_lstm"]],
        on=["sensor", "dias"],
        how="inner",
    )

    return merged


def compute_correlations(df: pd.DataFrame, col_visual) -> pd.DataFrame:
    results = []

    for window in sorted(df[COL_WINDOW].unique()):
        subset = df[df[COL_WINDOW] == window]

        for metric in WAPE_COLS:
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
        for c in [COL_WINDOW, COL_SENSOR, col_visual] + WAPE_COLS
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

    results_df.to_csv(OUTPUT_PATH[1], index=False)
    print(f"\nResultados salvos em: {OUTPUT_PATH[1]}")