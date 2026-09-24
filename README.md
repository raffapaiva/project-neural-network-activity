# Trabalho - Redes Neurais (make_moons)

Compara a rede Keras do `exemplo4.py` da aula com uma versao maior
(mais neuronios, uma camada oculta a mais) num problema de
classificacao nao linear (`make_moons`, 500 amostras, ruido 0.20).

## Estrutura

```
src/                        codigo (config, dados, modelos, visualizacao, trabalho_rede_neural)
figuras/                    7 PNGs gerados pelo script
modelos/                    pesos salvos (referencia.keras, proposto.keras)
resultados/                 RESULTADOS.md (auto-gerado) + resultados.csv + historicos por epoca
notebooks/                  trabalho_rede_neural.ipynb (demo) + exemplos_aula/exemplo4.py (baseline)
docs/                       roteiro_trabalho_rede_neural.md (relatorio conceitual)
legado/                     trabalho_rede_neural_original.py (versao anterior, raiz)
test_fundacao.py            smoke test (shapes, params, 1 epoca)
requirements.txt            dependencias (>= flexivel)
```

## Ambiente

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Execucao

```bash
.\.venv\Scripts\python.exe -m src.trabalho_rede_neural
.\.venv\Scripts\python.exe test_fundacao.py
```

Saidas: `figuras/` (7 PNGs), `modelos/` (*.keras),
`resultados/` (`RESULTADOS.md`, `resultados.csv`,
`historico_referencia.csv`, `historico_proposto.csv`).

## Nota sobre reprodutibilidade

Seed fixa (`SEED=42`, split 80/20 estratificado), mas versoes
diferentes de TensorFlow/CPU (oneDNN) podem variar ~1pp de acuracia
entre runs. O `resultados/RESULTADOS.md` de cada run registra a versao
exata usada e e a fonte oficial dos numeros — nao o texto do `docs/`.
