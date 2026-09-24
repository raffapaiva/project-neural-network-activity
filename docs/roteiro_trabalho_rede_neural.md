# Trabalho - Redes Neurais Artificiais

## Analise do impacto da arquitetura de uma rede neural em um problema de classificacao nao linear

## Objetivo

Modificar a rede neural do `exemplo4.py` e ver como mudar o numero de neuronios, o numero de camadas e as funcoes de ativacao afeta o desempenho da classificacao.

## Base de referencia

O ponto de partida e o Exemplo 4 da aula, que usa a base `make_moons` para gerar um problema de classificacao nao linear com duas entradas e duas classes. Na parte em Keras, o exemplo usa a arquitetura:

2 entradas -> 5 neuronios ReLU -> 5 neuronios tanh -> 1 neuronio sigmoid

Treinamento com SGD, taxa de aprendizado 0.1 e erro quadratico medio.

## Modificacoes feitas

- Base aumentada de 100 para 500 amostras.
- Ruido aumentado de 0.1 para 0.20.
- O modelo do exemplo da aula foi mantido como referencia, sem alteracoes.
- Foi criada uma segunda arquitetura, com uma camada oculta a mais:

2 entradas -> 16 ReLU -> 8 ReLU -> 4 tanh -> 1 sigmoid

A saida continua com sigmoid, ja que o problema e de classificacao binaria.

## Hipotese

A ideia era que uma rede maior conseguisse desenhar uma fronteira de decisao melhor num dataset com mais amostras e mais ruido. Mas aumentar neuronios e camadas nao garante um resultado melhor por si so - por isso a comparacao precisa ser feita na pratica, treinando os dois modelos nas mesmas condicoes.

## Metodologia

A base foi dividida em 80% treino e 20% teste (split estratificado,
`random_state=42` — treino=400, teste=100; ver `src/config.py` e
`src/dados.py`). Os dois modelos usam os mesmos parametros principais:

- otimizador SGD, taxa de aprendizado 0.1
- erro quadratico medio (mean squared error)
- 200 epocas, batch size 10

Para comparar os dois, foram usados: acuracia e loss no conjunto de teste, curva de loss e de acuracia durante o treino, e a fronteira de decisao aprendida por cada rede.

## Resultados

> Fonte oficial dos numeros: `resultados/RESULTADOS.md` (auto-gerado a cada
> run) + `resultados/resultados.csv`. A tabela abaixo e o registro do run
> original do notebook (TF da epoca); runs atuais podem variar ~1pp
> mesmo com seed fixa — ver nota de reprodutibilidade no `README.md`.

| Modelo | Arquitetura | Loss de teste | Acuracia de teste |
|---|---|---:|---:|
| Referencia | 5 ReLU -> 5 tanh -> 1 sigmoid | 0.012728 | 99.00% |
| Proposto | 16 ReLU -> 8 ReLU -> 4 tanh -> 1 sigmoid | 0.014799 | 98.00% |

Graficos gerados em `figuras/` (e exibidos no notebook em `notebooks/`):

- `figuras/01_base_make_moons.png` - base de dados usada
- `figuras/02_modelo_referencia_loss.png` / `figuras/02_modelo_referencia_acuracia.png`
- `figuras/03_modelo_proposto_loss.png` / `figuras/03_modelo_proposto_acuracia.png`
- `figuras/04_fronteira_modelo_referencia.png` / `figuras/05_fronteira_modelo_proposto.png`

## Discussao

O modelo proposto teve acuracia um pouco menor que a referencia (98% contra 99%, ou seja, errou 1 amostra a mais em 100 no teste). A loss final tambem ficou um pouco maior (0.0148 contra 0.0127).

Olhando as curvas de loss, o modelo proposto converge mais rapido: ja estabiliza perto de 0.02 por volta da epoca 40-50, enquanto o modelo de referencia so chega num patamar parecido perto da epoca 100-125. Isso condiz com o esperado, ja que a rede maior tem mais parametros para ajustar os pesos com o mesmo SGD. Em nenhum dos dois casos a loss de teste ficou visivelmente pior que a de treino, entao nao ha sinal de overfitting.

As fronteiras de decisao dos dois modelos ficaram bem parecidas e acompanham bem o formato das duas luas. A unica diferenca perceptivel esta perto do cruzamento entre as duas classes (em torno de x0 entre -0.5 e 0), onde o modelo proposto faz uma transicao um pouco mais abrupta, quase vertical. E provavelmente essa regiao que concentra a diferenca de acuracia entre os dois modelos.

Ou seja: para esse problema e esse nivel de ruido, a rede do exemplo da aula ja e suficiente. Aumentar neuronios e camadas fez o treino convergir mais rapido, mas nao trouxe ganho de acuracia - pelo contrario, piorou levemente, provavelmente porque a capacidade extra deixou o modelo um pouco mais sensivel ao ruido perto da fronteira entre as classes.

## Conclusao

Este trabalho comparou a rede do Exemplo 4 com uma versao modificada (mais neuronios, uma camada oculta a mais e ReLU nas duas primeiras camadas), usando uma base `make_moons` mais ruidosa que a original. Os resultados mostraram que o modelo maior converge mais rapido durante o treino, mas terminou com acuracia levemente menor no teste (98% contra 99% da referencia). Isso confirma a hipotese inicial: aumentar a complexidade da rede nao garante, por si so, um resultado melhor - nesse caso, o modelo mais simples do exemplo da aula ja era adequado para o problema.

## Arquivos gerados pelo programa

Ao executar `python -m src.trabalho_rede_neural` (a partir da raiz), sao criados:

- `figuras/01_base_make_moons.png`
- `figuras/02_modelo_referencia_loss.png`
- `figuras/02_modelo_referencia_acuracia.png`
- `figuras/03_modelo_proposto_loss.png`
- `figuras/03_modelo_proposto_acuracia.png`
- `figuras/04_fronteira_modelo_referencia.png`
- `figuras/05_fronteira_modelo_proposto.png`
- `modelos/referencia.keras` / `modelos/proposto.keras`
- `resultados/resultados.csv`, `resultados/historico_*.csv` e `resultados/RESULTADOS.md` (auto-gerado)

## Como executar

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.trabalho_rede_neural
```
