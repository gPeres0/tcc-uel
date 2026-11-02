import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from src.arima import main_arima
from src.plots import plot_predictions

# ==================== #

REFERENCIA_DIR = 'data_separation_sheet.csv'

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REFERENCIA_DIR = DATA_DIR / "data_separation_sheet.csv"

PERIODOS = [
    ('1day', '_1d', '1 dia'),
    ('3days', '_3d', '3 dias'),
    ('7days', '_7d', '7 dias'),
    ('15days', '_15d', '15 dias')
]

# Parâmetros variáveis por tamanho da série
CONFIGS = {
    97:   dict(L=6,  units=8,  dropout=0.0, batch=8,  l2=1e-3, patience=10),
    289:  dict(L=12, units=16, dropout=0.1, batch=16, l2=1e-4, patience=15),
    673:  dict(L=24, units=24, dropout=0.2, batch=32, l2=0.0,  patience=15),
    1441: dict(L=48, units=32, dropout=0.2, batch=32, l2=0.0,  patience=15),
}
 
# Parâmetros fixos para todas as séries
LR = 1e-3
CLIPNORM = 1.0
MAX_EPOCHS = 200
VAL_FRAC = 0.2   # fração final das janelas de treino reservada para validação
 
# ==================== #

def escolher_config(n_total):
    """Seleciona a configuração do tamanho de referência mais próximo."""
    ref = min(CONFIGS, key=lambda k: abs(k - n_total))
    return dict(CONFIGS[ref])

# ==================== #

def preprocessar(y_treino, y_teste):
    """
    Escala a série. O scaler é ajustado SOMENTE no treino (sem data leakage).
    StandardScaler: mais seguro que MinMax se o teste sair da faixa do treino.
    """
    treino = np.asarray(y_treino, dtype=float).reshape(-1, 1)
    teste = np.asarray(y_teste, dtype=float).reshape(-1, 1)
 
    scaler = StandardScaler().fit(treino)
    treino_s = scaler.transform(treino).ravel()
    teste_s = scaler.transform(teste).ravel()
    return treino_s, teste_s, scaler
 
 
def janelar(serie, L):
    """Pares (janela de L valores -> valor seguinte). Formato (n, L, 1)."""
    X = np.array([serie[i - L:i] for i in range(L, len(serie))])[..., None]
    y = serie[L:]
    return X, y
 
 
def split_validacao(X, y, frac=VAL_FRAC, min_val=10):
    """Validação = cauda cronológica das janelas de treino (sem shuffle)."""
    n_val = max(int(frac * len(X)), min_val)
    n_val = min(n_val, len(X) // 2)   # garante amostras de treino em séries curtas
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

# ==================== #

# ----------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------
def construir_modelo(L, units, dropout, l2):
    reg = regularizers.l2(l2) if l2 > 0 else None
    modelo = keras.Sequential([
        layers.Input(shape=(L, 1)),
        layers.LSTM(units, kernel_regularizer=reg),
        layers.Dropout(dropout),
        layers.Dense(1),
    ])
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM),
        loss="mse",
    )
    return modelo
 
 
# ----------------------------------------------------------------------
# Treino + previsão rolante (h=1)
# ----------------------------------------------------------------------
def prever_lstm(y_treino, y_teste, seed=0, config=None, verbose=0):
    """
    Treina a LSTM no treino e devolve as previsões h=1 para todo o teste,
    na escala original (np.ndarray de tamanho len(y_teste)).
    """
    cfg = config or escolher_config(len(y_treino) + len(y_teste))
    L = cfg["L"]
 
    treino_s, teste_s, scaler = preprocessar(y_treino, y_teste)
 
    # janelas construídas apenas dentro do treino
    X, y = janelar(treino_s, L)
    X_tr, y_tr, X_val, y_val = split_validacao(X, y)
 
    # reprodutibilidade (use várias seeds, ver prever_lstm_multiseed)
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
 
    modelo = construir_modelo(L, cfg["units"], cfg["dropout"], cfg["l2"])
    modelo.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=cfg["batch"],
        shuffle=True,  # embaralha só as janelas de treino; a validação é cronológica
        verbose=verbose,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=max(cfg["patience"] // 3, 3)),
        ],
    )
 
    # previsão rolante em lote: pesos fixos, entradas = valores reais anteriores
    full = np.concatenate([treino_s, teste_s])
    n_tr = len(treino_s)
    X_te = np.array([full[t - L:t] for t in range(n_tr, len(full))])[..., None]
 
    pred_s = modelo.predict(X_te, verbose=0).ravel()
    return scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()

# ==================== #

def main():
    resultados = []
    
    df_dir = REFERENCIA_DIR
    df = pd.read_csv(df_dir, quoting=csv.QUOTE_MINIMAL)
    
    for pasta, sufixo, n_dias in PERIODOS:
        pasta_completa = DATA_DIR / pasta
        arquivos_csv = sorted(pasta_completa.glob("*.csv"))

        if not arquivos_csv:
            print(f"[aviso] Nenhum CSV encontrado em: {pasta_completa}")
            continue

        if sufixo == '_7d':
            print("TA NO 7_D, SEGURA AI")
        
        if sufixo == '_15d':
            print("TA ACABANDO MUITO ESSA PORRA")

        for arquivo in arquivos_csv:
            sensor = main_arima.sensor_name(arquivo, sufixo)
            
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

            print(f"Iniciando predicao de {sensor}{sufixo}...")
            predictions = prever_lstm(y_treino, y_teste, seed=0)
            
            mae, mse, wape = main_arima.metrics(y_teste, predictions)
            
            resultados.append({
                'sensor': sensor,
                'period': n_dias,
                'mae': round(mae,3),
                'mse': round(mse,3),
                'wape': f"{round(wape,3)}%"
            })

            print(F"FOI O {sensor}{sufixo} HEIN...")
            output_path = DATA_DIR / "lstm_results"
            plot_predictions(output_path, sensor, sufixo, df_treino, df_teste, predictions, "lstm")
    
    print("ACABOU ESSA DESGRAÇA PORRAAAAAAAAAAAAAAAAAAAAA")
    return resultados

# ==================== #

if __name__ == "__main__":
    resultados = main()
    df_resultados = pd.DataFrame(resultados)
    output_file = DATA_DIR / 'lstm_results' / 'resultados_lstm.csv'
    df_resultados.to_csv(
        output_file,
        mode='a',
        header=not output_file.exists() or output_file.stat().st_size == 0,
        index=False,
    )