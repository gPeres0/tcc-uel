import os
import csv
import glob
import pandas as pd
from pmdarima.arima import auto_arima

from plots import plot_predictions
from main_arima import sensor_name, metrics

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

def test_main():
    df_dir = os.path.join(DATA_DIR, REFERENCIA_DIR)
    df = pd.read_csv(df_dir, quoting=csv.QUOTE_MINIMAL)

    pasta, sufixo, n_dias = PERIODOS[0]

    pasta_completa = os.path.join(DATA_DIR, pasta)
    arquivos_csv = sorted(glob.glob(os.path.join(pasta_completa, '*.csv')))

    if not arquivos_csv:
        print(f"[aviso] Nenhum CSV encontrado em: {pasta_completa}")

    arquivo = arquivos_csv[5]
    sensor = sensor_name(arquivo, sufixo)
    
    # Filtrando o DataFrame de forma correta
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

    model = auto_arima(y_treino, seasonal=True, m=1, stepwise=True, suppress_warnings=True, trace=True)
    
    predicoes = []
    for i in range(len(y_teste)):
        # Faz a previsão
        pred_res = model.predict(n_periods=1)
        # Pega o primeiro valor de forma segura, seja Pandas ou NumPy
        prediction = pred_res.iloc[0] if hasattr(pred_res, 'iloc') else pred_res[0]
        predicoes.append(prediction)
        model.update(y_teste.iloc[i])

    mae, mape, mse = metrics(y_teste, predicoes)
    
    print(f"MAE: {mae}\nMAPE: {mape}\nMSE: {mse}")
    # return sensor, n_dias, model.order, mae, mape, mse

    plot_predictions(df_treino, df_teste, predicoes)
    return

# ==================== #

if __name__ == '__main__':
    test_main()