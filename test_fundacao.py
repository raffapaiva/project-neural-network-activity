"""Smoke test rapido: imports, shapes e 1 epoca por modelo."""

from src import config
from src.dados import carregar_dados
from src.modelos import criar_modelo_proposto, criar_modelo_referencia


def main():
    X, Y, X_treino, X_teste, Y_treino, Y_teste = carregar_dados()
    assert X.shape == (config.N_AMOSTRAS, 2), X.shape
    assert len(X_treino) == 400 and len(X_teste) == 100, (len(X_treino), len(X_teste))

    for criar, params_esperados in [
        (criar_modelo_referencia, 51),
        (criar_modelo_proposto, 225),
    ]:
        modelo = criar()
        assert modelo.count_params() == params_esperados, modelo.count_params()
        hist = modelo.fit(
            X_treino, Y_treino, epochs=1, batch_size=config.BATCH_SIZE, verbose=0
        )
        assert "loss" in hist.history

    print("test_fundacao: OK (shapes, params 51/225, 1 epoca por modelo)")


if __name__ == "__main__":
    main()
