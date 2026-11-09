import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def plot_data(output_path:str, title:str, data):
    plt.figure(figsize=(12, 5))
    plt.plot(data["data"], data["valor"], color="blue")
    plt.title(f"Plot dados {title}")
    plt.xlabel("Data")
    plt.ylabel("Valor")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

    return print(f"Salvo arquivo {title}")


def plot_predictions(path:str, sensor:str, period:str, y_treino, y_teste, y_pred, model):
    if period == '_1d':     plot_dir = 'plots_1d'
    elif period == '_3d':   plot_dir = 'plots_3d'    
    elif period == '_7d':   plot_dir = 'plots_7d'
    elif period == '_15d':  plot_dir = 'plots_15d'
    else:                   return print(f"[PLOT_ERROR] period = {period}")    
        
    file_name = f"{path}/{plot_dir}/{sensor}{period}_plot_{model}.png"
    
    # série de treino/histórico
    timestamps_treino = y_treino.index
    valores_treino = y_treino['valor']

    # período de teste: real vs predito
    timestamps_teste = y_teste.index
    valores_reais_teste = y_teste['valor']

    plt.figure(figsize=(12, 5))

    # histórico (treino)
    plt.plot(timestamps_treino, valores_treino, label='Real (treino)', color='blue')
    # período de teste: real e predição sobrepostos
    plt.plot(timestamps_teste, valores_reais_teste, label='Real (teste)', color='blue')
    plt.plot(timestamps_teste, y_pred, label='Predição', color='orange')

    # linha pontilhada vertical separando treino de teste
    plt.axvline(x=timestamps_teste[0], color='gray', linestyle=':', linewidth=1.5)

    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.title('Real vs Predição')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    # plt.show()

    return print(f"Salvo {file_name}")


def plot_correlations_heatmap(data, pvals, titulo, nome_arquivo):
    """
    Gera dois heatmaps (Spearman e Kendall) relacionando métricas de entropia
    com janelas temporais (window_days), a partir de correlacoes_entropia_wape.csv.
    """

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data.values, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect=1)

    ax.set_xticks(range(len(window_order)))
    ax.set_xticklabels([f"{d}d" for d in window_order])
    ax.set_yticks(range(len(metric_order)))
    ax.set_yticklabels(metric_order)
    ax.set_xlabel("Janela (dias)")
    # ax.set_title(titulo, fontsize=12, fontweight="bold")

    # Anotações: valor da correlação + asterisco se p < 0.05 (sem FDR)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iat[i, j]
            p = pvals.iat[i, j]
            sig = "*" if pd.notna(p) and p < 0.05 else ""
            # cor do texto adapta ao fundo para manter contraste
            text_color = "white" if val < 0 else "black"
            ax.text(
                j, i, f"{val:.2f}{sig}",
                ha="center", va="center",
                color=text_color, fontsize=10, fontweight="bold"
            )

    # cbar = fig.colorbar(im, ax=ax, orientation="vertical", location="right", fraction=0.046, pad=0.04)
    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.04)
    cbar.set_label("Coeficiente de correlação")

    # fig.text(
    #     0.5, -0.03,
    #     "* p-valor bruto < 0.05 (sem correção FDR)",
    #     ha="center", fontsize=9, style="italic"
    # )

    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=200, bbox_inches="tight")
    # plt.show()
    # plt.close(fig)


def pivot(value_col):
    return (
        df.pivot(index="metric", columns="window_days", values=value_col)
        .reindex(index=metric_order, columns=window_order)
    )

if __name__ == "__main__":
    I = 3

    arquivos = [
        "correlacoes_entropia_visual_gabriel.csv",
        "correlacoes_entropia_visual_bruno.csv",
        "correlacoes_wape_visual_gabriel.csv",
        "correlacoes_wape_visual_bruno.csv",
    ]

    nome_output = [
        "entropy_visual1",
        "entropy_visual2",
        "wape_visual1",
        "wape_visual2",
    ] 

    # --- 1. Carregar dados -------------------------------------------------
    df = pd.read_csv(DATA_DIR / "correlations_results" / arquivos[I])

    # Ordem fixa para deixar os eixos consistentes e legíveis
    # metric_order = ["lempelziv", "lyapunov", "shannon", "perm", "mse_ci"]
    metric_order = ["wape_arima", "wape_lstm"]
    window_order = sorted(df["window_days"].unique())


    rho = pivot("spearman_rho")
    tau = pivot("kendall_tau")
    p_rho = pivot("spearman_p")
    p_tau = pivot("kendall_p")

    # --- 2. Configuração visual compartilhada -------------------------------
    VMIN, VMAX = -1, 1
    # CMAP = mcolors.LinearSegmentedColormap.from_list(
    #     "rdbu_r", plt.cm.RdBu_r(np.linspace(0, 1, 256))
    # )
    # magma e viridis 
    CMAP = plt.cm.viridis

    spearman_heatmap_path = DATA_DIR / "correlations_results/heatmaps" / f"{nome_output[I]}_spearman.png"
    kendall_heatmap_path = DATA_DIR / "correlations_results/heatmaps" / f"{nome_output[I]}_kendall.png"

    plot_correlations_heatmap(rho, p_rho, "Correlação de Spearman (ρ) — entropia x WAPE", spearman_heatmap_path)

    plot_correlations_heatmap(tau, p_tau, "Correlação de Kendall (τ) — entropia x WAPE", kendall_heatmap_path)



    # files = ['batmin', 'molfoliar1m_sec', 'pressao', 'raj10m', 'tempminar2m', 'tempsolocm2cm', 'tempsolonu20cm', 'tempsolonu100cm', 'ventodir10m', 'ventovel10m']
    # for folder in [('1day', '_1d'), ('3days', '_3d'), ('7days', '_7d'), ('15days', '_15d')]:
    #     for file in files:
    #         csv_path = f"{DATA_DIR}/{folder[0]}/{file}{folder[1]}.csv"
    #         output_path = f"{DATA_DIR}/{folder[0]}/plots/{file}{folder[1]}_plot.png"
    #         title = f"{file}{folder[1]}"
            
    #         df_dados = pd.read_csv(csv_path)
    #         df_dados['data'] = pd.to_datetime(df_dados['data'])

    #         plot_data(output_path, title, df_dados)
