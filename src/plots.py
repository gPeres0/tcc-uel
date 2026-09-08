import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

def plot_predictions(path:str, sensor:str, period:str, y_treino, y_teste, y_pred):
    if period == '_1d':     plot_dir = 'plots_1d'
    elif period == '_3d':   plot_dir = 'plots_3d'    
    elif period == '_7d':   plot_dir = 'plots_7d'
    else:                   return print(f"[PLOT_ERROR] period = {period}")    
        
    file_name = f"{path}/{plot_dir}/{sensor}{period}_plot.png"
    
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


def plot_correlations_by_sensor():
    # 1. Carregar os dados
    df_entropia = pd.read_csv('../data/resultados_entropia.csv')
    df_autoarima = pd.read_csv('../data/arima_results/resultados_autoarima.csv')

    # 2. Limpeza e preparação dos dados
    # Remove o símbolo de '%' do WAPE e converte para float
    df_autoarima['wape'] = df_autoarima['wape'].str.replace('%', '').astype(float)
    # Extrai o valor numérico (1, 3, 7) da coluna 'period' para alinhar com a coluna 'dias'
    df_autoarima['dias'] = df_autoarima['period'].str.extract(r'(\d+)').astype(int)

    # Definir as colunas de entropia que serão plotadas e seus respectivos estilos
    entropy_cols = ['lempelziv', 'shannon', 'perm', 'mse_ci']
    markers = ['s', 'o', '^', 'D']  # quadrado (square), círculo (circle), triângulo (triangle up), losango (diamond)
    colors = ['red', 'green', 'blue', 'purple']
    labels_entropia = ['Lempel-Ziv', 'Shannon', 'Permutação', 'MSE (CI)']

   # Obter a lista de sensores únicos
    sensors = df_entropia['sensor'].unique()

    # 3. Gerar e salvar um gráfico individual para cada sensor
    for sensor in sensors:
        # Criar uma nova figura para o sensor atual
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Filtrar dados do sensor atual e ordenar pelos dias (período)
        df_e_sens = df_entropia[df_entropia['sensor'] == sensor].sort_values('dias')
        df_a_sens = df_autoarima[df_autoarima['sensor'] == sensor].sort_values('dias')
        
        # Juntar (merge) os dataframes baseando-se no sensor e nos dias
        df_merged = pd.merge(df_e_sens, df_a_sens, on=['sensor', 'dias'], how='inner')
        
        # Eixo X para as posições
        x = np.arange(len(df_merged['dias']))
        width = 0.4
        
        # Eixo 1: WAPE (Barras)
        ax1.bar(x, df_merged['wape']/100, width, color='skyblue', alpha=0.8, label='WAPE')
        ax1.set_ylabel('WAPE', color='tab:blue', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{d} dia(s)' for d in df_merged['dias']])
        ax1.set_xlabel('Período de Previsão', fontsize=12)
        ax1.set_title(f'Sensor: {sensor}', fontsize=14, fontweight='bold')
        
        # Adicionar grid ao eixo do WAPE
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Eixo 2: Entropias (Linhas sobrepostas - eixo gêmeo)
        ax2 = ax1.twinx()
        for col, marker, color, label in zip(entropy_cols, markers, colors, labels_entropia):
            y_vals = pd.to_numeric(df_merged[col], errors='coerce')
            ax2.plot(x, y_vals, marker=marker, color=color, linewidth=2.5, markersize=8, label=label)
        
        ax2.set_ylabel('Medidas de Entropia', color='black', fontsize=12, fontweight='bold')
        
        # Unificar as legendas de ambos os eixos
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Ajustar layout para a legenda não cortar
        plt.tight_layout()
        
        # Salvar a figura com o nome específico do sensor
        filename = f'../data/correlations_results/corr_plot_{sensor}.png'
        plt.savefig(filename, bbox_inches='tight')
        
        # Fechar a figura atual da memória para não sobrepor na próxima iteração
        plt.close(fig)

    print("Todos os gráficos foram gerados e salvos com sucesso!")


def plot_correlations_by_period():
    df_correlacoes = pd.read_csv("../data/correlations_results/correlacoes_entropia_wape.csv")
    janelas = df_correlacoes['window_days'].unique()

    cores = {
        'lempelziv': 'red',
        'mse_ci': 'green',
        'perm': 'blue',
        'shannon': 'purple' # Amarelo adicionado para o shannon
    }

    rotulos = { 
        'lempelziv': 'lempelziv',
        'mse_ci': 'mse-ci',
        'perm': 'perm',
        'shannon': 'shannon'
    }

    for w in janelas:
        fig, ax = plt.subplots(figsize=(11, 6))
        
        # Filtra os dados apenas para a janela atual
        df_w = df_correlacoes[df_correlacoes['window_days'] == w]
        metricas = df_w['metric'].tolist()
        
        # Posições base no eixo X para as métricas (espaçamento entre blocos)
        posicoes_base = [1, 3, 5, 7] 
        largura = 0.25
        
        todas_posicoes = []
        todos_rotulos = []

        # Plotagem iterativa para cada métrica encontrada nesta janela
        for i, metric in enumerate(metricas):
            linha = df_w[df_w['metric'] == metric].iloc[0]
            
            # Coleta as correlações (usando valor absoluto para o tamanho da barra bater com o lado positivo)
            rho = abs(linha['spearman_rho'])
            tau = abs(linha['kendall_tau'])
            
            pos_rho = posicoes_base[i]
            pos_tau = posicoes_base[i] + largura + 0.15
            
            cor = cores.get(metric, '#cccccc')
            label_legenda = rotulos.get(metric, metric)
            
            ax.bar([pos_rho, pos_tau], [rho, tau], width=largura, color=cor, 
                edgecolor='black', linewidth=1, label=label_legenda, zorder=3)
            
            todas_posicoes.extend([pos_rho, pos_tau])
            todos_rotulos.extend([r'$\rho$', r'$\tau$'])

        # --- LINHAS HORIZONTAIS E EIXO Y ---
        niveis = [0.1, 0.3, 0.5, 0.7]
        ax.hlines(y=niveis, xmin=0, xmax=max(todas_posicoes) + 1, 
                colors='black', linestyles='dashed', linewidth=1, zorder=0)

        ax.set_yticks(niveis)
        ax.set_yticklabels(['0,1', '0,3', '0,5', '0,7'])
        ax.set_ylim(0, 0.9)

        # --- EIXO X ---
        ax.set_xticks(todas_posicoes)
        ax.set_xticklabels(todos_rotulos)
        ax.set_xlim(0, max(todas_posicoes) + 1.5)
        
        # --- TEXTOS DE CLASSIFICAÇÃO LATERAIS ---
        x_textos = max(todas_posicoes) + 1.2
        ax.text(x_textos, 0.05, 'desprezível', va='center', ha='left', fontsize=12)
        ax.text(x_textos, 0.20, 'fraca', va='center', ha='left', fontsize=12)
        ax.text(x_textos, 0.40, 'moderada', va='center', ha='left', fontsize=12)
        ax.text(x_textos, 0.60, 'forte', va='center', ha='left', fontsize=12)
        ax.text(x_textos, 0.80, 'muito forte', va='center', ha='left', fontsize=12)

        # --- ESTILIZAÇÃO GERAL ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.15), frameon=True, 
                edgecolor='black', fancybox=True, borderpad=1, labelspacing=1)
                
        plt.title(f'Janela de {w} Dias', pad=20)
        plt.tight_layout()
        
        # Salva a imagem atual
        nome_arquivo = f'../data/correlations_results/plots/plots_by_period/corr_plot_{w}_days.png'
        plt.savefig(nome_arquivo, dpi=300)
        plt.close() # Fecha a figura atual para liberar memória antes da próxima
        
        print(f"Salvo: {nome_arquivo}")


def plot_correlations_heatmap(data, pvals, titulo, nome_arquivo):
    """
    Gera dois heatmaps (Spearman e Kendall) relacionando métricas de entropia
    com janelas temporais (window_days), a partir de correlacoes_entropia_wape.csv.
    """

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data.values, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")

    ax.set_xticks(range(len(window_order)))
    ax.set_xticklabels([f"{d}d" for d in window_order])
    ax.set_yticks(range(len(metric_order)))
    ax.set_yticklabels(metric_order)
    ax.set_xlabel("Janela (dias)")
    ax.set_title(titulo, fontsize=12, fontweight="bold")

    # Anotações: valor da correlação + asterisco se p < 0.05 (sem FDR)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iat[i, j]
            p = pvals.iat[i, j]
            sig = "*" if pd.notna(p) and p < 0.05 else ""
            # cor do texto adapta ao fundo para manter contraste
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}{sig}",
                ha="center", va="center",
                color=text_color, fontsize=10, fontweight="bold"
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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


# plot_correlations_by_period()


# --- 1. Carregar dados -------------------------------------------------
df = pd.read_csv("../data/correlations_results/correlacoes_entropia_wape.csv")

# Ordem fixa para deixar os eixos consistentes e legíveis
metric_order = ["shannon", "perm", "lempelziv", "mse_ci"]
window_order = sorted(df["window_days"].unique())

def pivot(value_col):
    return (
        df.pivot(index="metric", columns="window_days", values=value_col)
        .reindex(index=metric_order, columns=window_order)
    )

rho = pivot("spearman_rho")
tau = pivot("kendall_tau")
p_rho = pivot("spearman_p")
p_tau = pivot("kendall_p")

# --- 2. Configuração visual compartilhada -------------------------------
VMIN, VMAX = -1, 1
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rdbu_r", plt.cm.RdBu_r(np.linspace(0, 1, 256))
)


plot_correlations_heatmap(rho, p_rho, "Correlação de Spearman (ρ) — entropia x WAPE",
                "../data/correlations_results/plots/heatmaps/heatmap_spearman.png")

plot_correlations_heatmap(tau, p_tau, "Correlação de Kendall (τ) — entropia x WAPE",
                "../data/correlations_results/plots/heatmaps/heatmap_kendall.png")
