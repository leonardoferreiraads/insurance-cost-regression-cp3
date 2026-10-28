# Previsão de custos de seguro saúde: otimização e comparação de modelos de regressão

Checkpoint 3 da disciplina **Machine Learning & Modelling** (FIAP).
Evolução do CP2: além do KNN, comparamos Regressão Linear e Ridge, com engenharia de atributos e seleção de modelo por validação cruzada.

## Objetivo

Prever o valor cobrado (`charges`) a partir de idade, sexo, IMC, número de filhos, tabagismo e região, e comparar de forma objetiva qual abordagem generaliza melhor.

## Dataset

[Insurance](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv): 1.338 registros, 1.337 após remover 1 duplicata. Não há valores ausentes.

## Metodologia

- **Divisão 80/20** estratificada por `smoker`, com `random_state=42`.
- **Sem vazamento de dados:** o escalonamento fica dentro de um `Pipeline`, ajustado apenas no treino de cada fold.
- **Engenharia de atributos:** `obese`, `smoker_obese`, `smoker_bmi`, `age_smoker` e `age2`, motivados pela análise exploratória (efeito do IMC é diferente entre fumantes e não fumantes).
- **Modelos:** KNN padrão (k=5), Regressão Linear básica, Regressão Linear com engenharia, KNN otimizado e Ridge com engenharia.
- **Otimização:** `GridSearchCV` com 5 folds, variando escalonador, número de vizinhos, pesos, métrica de distância e `alpha`.
- **Critério de escolha:** menor RMSE médio de validação cruzada, definido **antes** de consultar o conjunto de teste.

## Resultados no conjunto de teste

| Modelo | MAE | RMSE | R² |
|:--|--:|--:|--:|
| **Ridge + engenharia (selecionado)** | **2.004,35** | **3.368,30** | **0,9213** |
| Regressão Linear + engenharia | 2.006,23 | 3.374,17 | 0,9210 |
| KNN otimizado | 2.094,81 | 3.435,19 | 0,9181 |
| KNN padrão (k=5) | 3.153,14 | 4.945,85 | 0,8303 |
| Regressão Linear básica | 3.592,79 | 5.086,88 | 0,8205 |

**Principais conclusões**

- A engenharia de atributos teve o maior impacto: a Regressão Linear passou de R² 0,82 para 0,92.
- Ridge e Regressão Linear com engenharia são praticamente equivalentes (diferença de RMSE de CV abaixo de 0,01%).
- O KNN otimizado reduziu o RMSE em 30,5% frente ao KNN padrão, mas esse ganho combina ajuste de hiperparâmetros e novos atributos.

## Como executar

```bash
git clone https://github.com/<seu-usuario>/insurance-cost-regression-cp3.git
cd insurance-cost-regression-cp3
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook VS01_CP3_Machine_Learning.ipynb
```

O notebook baixa o dataset automaticamente do GitHub caso `insurance.csv` não esteja na pasta.

## Limitações

Amostra pequena (cerca de 1,3 mil registros), resultados sujeitos à divisão dos dados e erro maior para custos extremos. Projeto acadêmico, sem finalidade de precificação real.

## Integrantes

- Leonardo Ferreira (RM566554)
- Caio Nicolav (RM574784)
- Guilherme Stefani (RM574442)
- Guilherme Giannini (RM576746)
