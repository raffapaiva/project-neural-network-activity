"""Geracao da base make_moons com split estratificado."""

from sklearn import datasets
from sklearn.model_selection import train_test_split

from . import config


def carregar_dados():
    X, Y = datasets.make_moons(
        n_samples=config.N_AMOSTRAS,
        noise=config.RUIDO,
        random_state=config.SEED,
    )
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(
        X, Y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=Y
    )
    return X, Y, X_treino, X_teste, Y_treino, Y_teste
