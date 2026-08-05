import matplotlib.pyplot as plt
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

def plot_correlations():
    # df_entropias = pd.read_csv("../data/resultados_entropia.csv")
    # df_arima = pd.read_csv("arima/results/resultados_autoarima.csv")

    # timestamps = df_arima["period"].str.extract(r"(\d+)").astype(int)
    # valores_wape = df_arima["wape"].str.replace("%", "", regex=False).astype(float)
    # valores_shannon = df_entropias["shannon"]
    # valores_lempelziv = df_entropias["lempelziv"]
    # valores_perm = df_entropias["perm"]
    # valores_mse = df_entropias["mse_ci"]


    # plt.figure(figsize=(5, 12))
    # plt.plot(timestamps, valores_wape, )

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



# plot_correlations()