"""As duas arquiteturas comparadas no trabalho."""

from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD

from . import config


def _compilar(modelo):
    otimizador = SGD(learning_rate=config.TAXA_APRENDIZADO)
    modelo.compile(loss=config.LOSS, optimizer=otimizador, metrics=["accuracy"])
    return modelo


def criar_modelo_referencia():
    """Mesma arquitetura da parte Keras do exemplo4.py."""
    modelo = Sequential(
        [
            Input(shape=(2,)),
            Dense(5, activation="relu"),
            Dense(5, activation="tanh"),
            Dense(1, activation="sigmoid"),
        ]
    )
    return _compilar(modelo)


def criar_modelo_proposto():
    """Versao modificada: mais neuronios, uma camada oculta a mais."""
    modelo = Sequential(
        [
            Input(shape=(2,)),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, activation="tanh"),
            Dense(1, activation="sigmoid"),
        ]
    )
    return _compilar(modelo)
