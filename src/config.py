"""Parametros centrais do experimento.

Unica fonte de verdade — `dados.py`, `modelos.py`, `visualizacao.py` e
`trabalho_rede_neural.py` importam daqui, e o `RESULTADOS.md` auto-gerado
espelha estes valores para auditoria.
"""

from pathlib import Path

# Reprodutibilidade
SEED = 42

# Base make_moons (exemplo4.py usa 100 amostras e ruido 0.1)
N_AMOSTRAS = 500
RUIDO = 0.20
TEST_SIZE = 0.20

# Treino (mantido igual para os dois modelos, comparacao justa)
TAXA_APRENDIZADO = 0.1
EPOCAS = 200
BATCH_SIZE = 10
OTIMIZADOR = "SGD"
LOSS = "mean_squared_error"

# Pastas (pathlib — o script funciona a partir de qualquer cwd)
RAIZ = Path(__file__).resolve().parents[1]
DIR_FIGURAS = RAIZ / "figuras"
DIR_MODELOS = RAIZ / "modelos"
DIR_RESULTADOS = RAIZ / "resultados"

ARQUITETURA_REFERENCIA = "5 ReLU -> 5 tanh -> 1 sigmoid"
ARQUITETURA_PROPOSTA = "16 ReLU -> 8 ReLU -> 4 tanh -> 1 sigmoid"
