import matplotlib.pyplot as plt

def plot_predictions(sensor:str, period:str, y_treino, y_teste, y_pred):
    if period == '_1d':     plot_dir = 'plots_1d'
    elif period == '_3d':   plot_dir = 'plots_3d'    
    elif period == '_7d':   plot_dir = 'plots_7d'
    else:                   return print(f"[PLOT_ERROR] period = {period}")    
        
    file_name = f"results/{plot_dir}/{sensor}{period}_plot.png"
    
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