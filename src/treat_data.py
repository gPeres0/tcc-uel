"""
Script para separar dados de sensores meteorológicos em vários arquivos menores,
um para cada combinação (sensor x período total), com base nas janelas de
treino/teste definidas em um CSV de referência.

Estratégia atual:
- Cada sensor tem seu próprio arquivo CSV de dados brutos, salvo na pasta
  RAW_DATA_DIR com o nome "{sensor}.csv" (ex: raw_data/batmin.csv).
- Cada um desses arquivos tem as colunas: id_sample, id_sensor, value, date_ref
  (date_ref já vem com data e hora juntas, ex: "2021-07-28 12:45:00").
- As janelas de treino/teste (Data Início Treino ... Data Fim Teste) vêm de
  um CSV separado (ex: "data_separation_sheet.csv"), com datas no formato
  "MM/DD" referentes ao ano de ANO_REFERENCIA.
- Só processamos os sensores presentes em SENSORES_VALIDOS.

AJUSTE AS CONFIGURAÇÕES ABAIXO ANTES DE RODAR.
"""

import os
import re
from datetime import datetime

import pandas as pd

# ============================================================
# 1. CONFIGURAÇÕES — AJUSTE CONFORME SEU AMBIENTE
# ============================================================

# Pasta com os arquivos brutos de cada sensor: raw_data/{sensor}.csv
RAW_DATA_DIR = "../data/raw_data"

# Colunas do arquivo bruto de cada sensor
COL_VALUE = "value"
COL_DATE_REF = "date_ref"
FORMATO_DATE_REF = "%Y-%m-%d %H:%M:%S"

# Caminho do CSV com as janelas de treino/teste
JANELAS_CSV = "../data/data_separation_sheet.csv"
SEP_JANELAS = ","       # separador do CSV de janelas

# Pasta onde os arquivos menores (saída) serão salvos
OUTPUT_DIR = {
    "1 dia": "../data/1day/",
    "3 dias": "../data/3days/",
    "7 dias": "../data/7days/",
}
SEP_SAIDA = ","          # separador dos arquivos de saída

# Ano de referência das datas do CSV de janelas, que estão no formato "MM/DD"
ANO_REFERENCIA = 2021

# Identificação fixa da estação/cidade (todos os arquivos são dessa estação)
NOME_CIDADE = "Londrina"
ID_CIDADE = "23265102"

# Sensores que devem ser processados (demais linhas do CSV de janelas são ignoradas)
SENSORES_VALIDOS = {
    "batmin",
    "molfoliar1m_sec",
    "pressao",
    "raj10m",
    "tempminar2m",
    "tempsolocm20cm",
    "tempsolonu20cm",
    "tempsolonu100cm",
    "ventodir10m",
    "ventovel10m",
}

# Mapeamento do texto da coluna "Período Total" para o sufixo do nome do arquivo
PERIODO_SUFIXO = {
    "1 dia": "1d",
    "3 dias": "3d",
    "7 dias": "7d",
}

# Frequência esperada das medições (usada para preencher horários faltantes no período)
FREQ_DADOS = "15min"

# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def parse_data_mmdd(data_str: str, hora_str: str, ano: int) -> datetime:
    """Converte 'MM/DD' + 'HH:MM' + ano em datetime."""
    mes, dia = data_str.strip().split("/")
    return datetime.strptime(f"{ano}-{int(mes):02d}-{int(dia):02d} {hora_str.strip()}",
                              "%Y-%m-%d %H:%M")


def limpar_nome_sensor(nome: str) -> str:
    """Remove marcadores como '*' e espaços extras do nome do sensor."""
    return re.sub(r"[^\w]", "", nome.strip())


# ============================================================
# 3. CARREGAR JANELAS DE TREINO/TESTE
# ============================================================

print("Lendo CSV de janelas (treino/teste)...")
df_janelas = pd.read_csv(JANELAS_CSV, sep=SEP_JANELAS, dtype=str)
df_janelas.columns = [c.strip().strip('"') for c in df_janelas.columns]
df_janelas = df_janelas.dropna(subset=["Sensor"])  # remove linhas totalmente vazias

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache dos arquivos de dados brutos já carregados (evita reler o mesmo sensor várias vezes)
_cache_dados = {}


def carregar_dados_sensor(sensor: str) -> pd.DataFrame:
    if sensor in _cache_dados:
        return _cache_dados[sensor]

    caminho = os.path.join(RAW_DATA_DIR, f"{sensor}.csv")
    if not os.path.isfile(caminho):
        _cache_dados[sensor] = None
        return None

    df = pd.read_csv(caminho, dtype=str)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["__datetime__"] = pd.to_datetime(df[COL_DATE_REF].str.strip(),
                                         format=FORMATO_DATE_REF, errors="coerce")
    df[COL_VALUE] = pd.to_numeric(df[COL_VALUE], errors="coerce")

    n_invalidas = df["__datetime__"].isna().sum()
    if n_invalidas:
        print(f"[AVISO] {sensor}: {n_invalidas} linhas com date_ref inválido foram ignoradas.")

    _cache_dados[sensor] = df
    return df


# ============================================================
# 4. GERAR UM ARQUIVO PARA CADA (SENSOR x PERÍODO TOTAL)
# ============================================================

resumo = []

for _, linha in df_janelas.iterrows():
    sensor_raw = str(linha["Sensor"]).strip()
    sensor = limpar_nome_sensor(sensor_raw)

    if sensor not in SENSORES_VALIDOS:
        continue  # sensor fora da lista de interesse

    periodo_total = str(linha["Período Total"]).strip()
    sufixo = PERIODO_SUFIXO.get(periodo_total, re.sub(r"\s+", "_", periodo_total))
    output = OUTPUT_DIR.get(periodo_total, re.sub(r"\s+", "_", periodo_total))

    try:
        inicio_treino = parse_data_mmdd(linha["Data Início Treino"], linha["Hora1"], ANO_REFERENCIA)
        fim_teste = parse_data_mmdd(linha["Data Fim Teste"], linha["Hora3"], ANO_REFERENCIA)
    except Exception as e:
        print(f"[AVISO] Não foi possível interpretar as datas da linha (sensor={sensor_raw}, "
              f"período={periodo_total}): {e}. Pulando.")
        continue

    # Caso o período cruze a virada do ano (fim menor que início), soma 1 ano ao fim
    if fim_teste < inicio_treino:
        fim_teste = fim_teste.replace(year=fim_teste.year + 1)

    df_sensor = carregar_dados_sensor(sensor)
    if df_sensor is None:
        print(f"[AVISO] Arquivo de dados não encontrado para o sensor '{sensor}' "
              f"(esperado em {os.path.join(RAW_DATA_DIR, sensor + '.csv')}). Pulando.")
        continue

    mask = (df_sensor["__datetime__"] >= inicio_treino) & (df_sensor["__datetime__"] <= fim_teste)
    subset = df_sensor.loc[mask, ["__datetime__", COL_VALUE]].copy()
    subset = subset.rename(columns={"__datetime__": "data", COL_VALUE: "valor"})

    # Preenche horários faltantes dentro do período (gaps na série) com base na frequência esperada
    idx_completo = pd.date_range(start=inicio_treino, end=fim_teste, freq=FREQ_DADOS)
    subset = subset.set_index("data").reindex(idx_completo)
    subset.index.name = "data"
    subset = subset.reset_index()

    # Preenche valores faltantes/ inválidos (NaN) com 0
    n_faltando = subset["valor"].isna().sum()
    if n_faltando:
        print(f"[INFO] {sensor} ({periodo_total}): {n_faltando} valores faltando preenchidos com 0.")
    subset["valor"] = subset["valor"].fillna(0)

    subset.insert(0, "cidade", NOME_CIDADE)
    subset.insert(1, "id_cidade", ID_CIDADE)
    subset.insert(2, "sensor", sensor)

    nome_arquivo = f"{sensor}_{sufixo}.csv"
    caminho_saida = os.path.join(output, nome_arquivo)
    subset.to_csv(caminho_saida, index=False, sep=SEP_SAIDA)

    print(f"Gerado: {caminho_saida}  ({len(subset)} linhas, "
          f"{inicio_treino:%d/%m/%Y %H:%M} a {fim_teste:%d/%m/%Y %H:%M})")
    resumo.append((sensor, periodo_total, len(subset), caminho_saida))

print("\nResumo:")
for sensor, periodo, n, caminho in resumo:
    print(f"  {sensor:20s} {periodo:8s} -> {n:6d} linhas -> {caminho}")

sensores_processados = {s for s, _, _, _ in resumo}
faltando = SENSORES_VALIDOS - sensores_processados
if faltando:
    print(f"\n[AVISO] Sensores da lista de interesse sem nenhum arquivo gerado: {sorted(faltando)}")

print("\nConcluído!")