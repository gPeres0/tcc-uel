# Caracterização por Complexidade e Entropia de Séries Temporais em IoT
#### Autor: Gabriel Peres de Souza

## Métodos de Complexidade
### Complexidade de Lempel-Ziv
A usabilidade formal do método reside em quantificar a taxa de surgimento de novos subpadrões em uma sequência. Para isso, segue os seguintes passos:

1. Inicialmente, a série temporal contı́nua é discretizada (geralmente binarizada com base em um limiar, como a mediana);
2. Em seguida, o algoritmo varre a sequência da esquerda para a direita;
3. Enquanto varre, um contador de complexidade é incrementado sempre que uma subsequência inédita é encontrada;
4. O valor final reflete a diversidade estrutural do sinal.

Séries temporais altamente previsı́veis apresentam um baixo ı́ndice de Lempel-Ziv, enquanto sequências com elevada aleatoriedade produzem valores próximos ao limite superior de complexidade.

**Biblioteca(s) utilizada(s):** 
- AntroPy ([ant.lziv_complexity](https://raphaelvallat.com/antropy/#entropy)).
- lempel_ziv_complexity module ([lempel_ziv_complexity](https://pypi.org/project/lempel-ziv-complexity/)). 

### Expoente de Lyapunov
Desenvolvido para avaliar a estabilidade de soluções em equações diferenciais, é usado na análise de séries temporais para quantificar a sensibilidade do sistema a condições iniciais.  
Ele mede a taxa média exponencial de divergência ou convergência de duas trajetórias no espaço de fase que se iniciam em pontos muito próximos. Se $\lambda_{max} > 0$, o sistema exibe comportamento caótico, indicando que trajetórias vizinhas se separam exponencialmente com o tempo. 

**Biblioteca(s) utilizada(s):** 
- nolds ([lyap_r, lyap_e](https://pypi.org/project/nolds/)).
- Lyapynv ([GitHub - ThomasSavary08](https://github.com/ThomasSavary08/Lyapynov)).

## Métodos de Entropia
### Entropia de Shannon
Métrica clássica principal da Teoria da Informação. A motivação original de Shannon foi quantificar o limite teórico de compressão de dados e a capacidade de transmissão de informações em canais de comunicação com ruı́do.  
Matematicamente, para uma variável aleatória discreta $X$ com uma função de probabilidade $P(x)$, a Entropia de Shannon é definida como: 
$$ H(X) = - \sum_{i=1}^{n}{P(x_i)\log{P(x_i)}} $$
Na análise de séries temporais, esta equação é utilizada para calcular a desordem
global da distribuição dos valores do sinal. Um valor elevado de $H(X)$ indica uma distribuição uniforme das observações, traduzindo-se em alta incerteza na predição de um estado futuro baseando-se em probabilidades de ocorrência.

**Biblioteca(s) utilizada(s):** 
- SciPy ([scipy.stats.entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)).

### Entropia de Permutação
Criada para avaliar a complexidade de séries temporais caóticas de maneira computacionalmente eficiente e com resistência a ruı́dos de medição.  
A usabilidade da PE baseia-se na extração de padrões ordinais, seguindo os seguintes passos de execução:
1. A série de entrada é particionada em vetores de dimensão $D$;
2. Para cada vetor, os elementos são ordenados de forma crescente, e o padrão de permutação correspondente é mapeado;
3. A PE é então calculada aplicando-se a equação de Shannon sobre a distribuição de probabilidade das permutações observadas.

**Biblioteca(s) utilizada(s):** 
- ordpy ([ordpy.permutation_entropy](https://arthurpessa.github.io/ordpy/_build/html/index.html#ordpy.permutation_entropy)).
- AntroPy ([ant.perm_entropy](https://raphaelvallat.com/antropy/#entropy)).
- EntropyHub 2.0 ([PermEn](https://pypi.org/project/EntropyHub/)).

### Entropia Multiescala
Desenvolvida para superar a limitação
de métricas de entropia tradicionais, que operam em uma única escala de tempo e
frequentemente falham em distinguir séries aleatórias (ruı́do branco) de processos complexos que possuem dependências de longo prazo.  
Consiste em dois passos principais:
1. *Coarse-Graining:* Processo de granulação da série onde janelas de tamanho τ (fator de escala) são agrupadas e suas médias são calculadas, gerando múltiplas séries representativas de diferentes escalas de tempo.
2. *Sample Entropy:* Calcula a SampEn para cada sub-série gerada.

**Biblioteca(s) utilizada(s):**
- EntropyHub 2.0 ([MSEn - MSobject('SampEn')](https://pypi.org/project/EntropyHub/)).
- MultiScaleEntropy ([GitHub - inuritdino](https://github.com/inuritdino/MultiScaleEntropy)).
- py-msentropy ([GitHub - antoine-jamin](https://github.com/antoine-jamin/py-msentropy)).
- **Sample Entropy:** AntroPy ([ant.sample_entropy](https://raphaelvallat.com/antropy/#entropy)).
- **Material/biblioteca (C) de apoio:** [Tutorial PhysioNet](https://physionet.org/files/mse/1.0/tutorial/tutorial.pdf).

# Ferramentas
## Bibliotecas Utilizadas
Em cada método descrito acima, foram coletadas possíveis bibliotecas/APIs que implementam dado método. Aqui, são descritas as ferramentas que, após análise, serão de fato utilizadas no projeto.

### AntroPy
Documentação em https://raphaelvallat.com/antropy/.
##### Utilidade:
Cálculo de complexidade de Lempel-Ziv e entropia de permutação.
##### Instalação:
```bash 
pip install antropy
```
##### Dependências:
- Python 3.10+;
- NumPy  (≥ 1.22.4); 
- SciPy (≥ 1.8.0);
- scikit-learn (≥ 1.2.0);
- Numba (≥ 0.57).

### Nolds
Documentação em https://cschoel.github.io/nolds/.
##### Utilidade:
Cálculo de expoente de Lyapunov.
##### Instalação:
```bash 
pip install nolds
```
##### Dependências:
- Python 2.7+ e 3.4+;
- Numpy;
- matplotlib (plots com `nolds.examples`).

### SciPy (scipy.stats)
Documentação em https://docs.scipy.org/doc/scipy/tutorial/stats.html.
##### Utilidade:
Cálculo da entropia de Shannon.
##### Instalação:
```bash 
pip install scipy
```
##### Dependências:
Não descritas na documentação.

### EntropyHub
Documentação em https://github.com/MattWillFlood/EntropyHub/blob/main/EntropyHub%20Guide.pdf.
##### Utilidade:
Cálculo de entropia de permutação (não será utilizado) e MSE.
##### Instalação:
```bash 
pip install EntropyHub
```
##### Dependências:
- Python 3.6+;
- Numpy;
- Scipy;
- Matplotlib;
- PyEMD.

### OrdPy
Documentação em https://arthurpessa.github.io/ordpy/_build/html/index.html#.
##### Utilidade:
Cálculo de entropia de permutação.
##### Instalação:
```bash 
pip install ordpy
```
##### Dependências:
Não descritas na documentação.

### Instalação geral
```bash 
pip install antropy nolds scipy EntropyHub ordpy
```

