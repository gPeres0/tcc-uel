import os
import csv
import glob
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

from plots import plot_predictions

# ==================== #

REFERENCIA_DIR = 'data_separation_sheet.csv'

DATA_DIR = '../../data'
# (nome da pasta, sufixo do arquivo, número de dias)
PERIODOS = [
    ('1day', '_1d', '1 dia'),
    ('3days', '_3d', '3 dias'),
    ('7days', '_7d', '7 dias'),
]

# ==================== #

def sensor_name(caminho_csv, sufixo):
    """Extrai o nome do sensor a partir do nome do arquivo, removendo o sufixo do período."""
    base = os.path.splitext(os.path.basename(caminho_csv))[0]
    if base.endswith(sufixo):
        base = base[: -len(sufixo)]
    return base

def metrics(y_real, y_pred):
    y_real, y_pred = np.array(y_real), np.array(y_pred)
    
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    # MAPE não funciona com valores 0 e próximos, por isso o uso do WAPE (Weighted MAPE / MAD-Mean ratio)
    wape = np.sum(np.abs(y_real - y_pred)) / np.sum(np.abs(y_real))

    return mae, wape*100, mse

# ==================== #

def main():
    resultados = []

    df_dir = os.path.join(DATA_DIR, REFERENCIA_DIR)
    df = pd.read_csv(df_dir, quoting=csv.QUOTE_MINIMAL)
    
    for pasta, sufixo, n_dias in PERIODOS:
        pasta_completa = os.path.join(DATA_DIR, pasta)
        arquivos_csv = sorted(glob.glob(os.path.join(pasta_completa, '*.csv')))

        if not arquivos_csv:
            print(f"[aviso] Nenhum CSV encontrado em: {pasta_completa}")
            continue

        if sufixo == '_7d':
            print("TA ACABANDO ESSA PORRA, SEGURA AI")

        for arquivo in arquivos_csv:
            sensor = sensor_name(arquivo, sufixo)
            
            filtro = (df['Sensor'] == sensor) & (df['Período Total'] == n_dias)
            df_filtrado = df[filtro]
            
            data_hora_inicio_treino = pd.to_datetime(df_filtrado['data_hora_inicio_treino'].iloc[0])
            data_hora_fim_treino = pd.to_datetime(df_filtrado['data_hora_fim_treino'].iloc[0])
            data_hora_fim_teste = pd.to_datetime(df_filtrado['data_hora_fim_teste'].iloc[0])

            df_dados = pd.read_csv(arquivo)
            df_dados['data'] = pd.to_datetime(df_dados['data'])
            df_dados = df_dados.set_index('data').sort_index()
            
            df_treino = df_dados.loc[data_hora_inicio_treino : data_hora_fim_treino]
            df_teste = df_dados.loc[data_hora_fim_treino : data_hora_fim_teste]
            
            y_treino = df_treino['valor']
            y_teste = df_teste['valor']

            model = auto_arima(y_treino, seasonal=True, m=1, stepwise=True, suppress_warnings=True, trace=False)

            predicoes = []
            for i in range(len(y_teste)):
                # Faz a previsão
                pred_res = model.predict(n_periods=1)
                # Pega o primeiro valor de forma segura, seja Pandas ou NumPy
                prediction = pred_res.iloc[0] if hasattr(pred_res, 'iloc') else pred_res[0]
                predicoes.append(prediction)
                model.update(y_teste.iloc[i])
            
            mae, mape, mse = metrics(y_teste, predicoes)
            
            resultados.append({
                'sensor': sensor,
                'period': n_dias,
                'order': model.order,
                'mae': round(mae,3),
                'mse': round(mse,3),
                'wape': f"{round(mape,3)}%"
            })

            print(F"FOI O {sensor}{sufixo} HEIN...")
            # plot_predictions(sensor, sufixo, df_treino, df_teste, predicoes)

    print("ACABOU ESSA DESGRAÇA PORRAAAAAAAAAAAAAAAAAAAAA")
    return resultados

if __name__ == '__main__':
    resultados = main()
    df = pd.DataFrame(resultados)
    df.to_csv('results/resultados_autoarima.csv', index=False)
