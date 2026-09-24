"""Figuras do trabalho — todas recebem o diretorio de saida explicitamente."""

import matplotlib.pyplot as plt
import numpy as np


def salvar_grafico_base(X, Y, caminho):
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("Base make_moons - 500 amostras e ruido 0.20")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def salvar_curvas(historico, prefixo, titulo):
    """Gera `<prefixo>_loss.png` e `<prefixo>_acuracia.png`."""
    plt.figure(figsize=(7, 5))
    plt.plot(historico.history["loss"], label="Treino")
    plt.plot(historico.history["val_loss"], label="Teste")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title(f"Loss - {titulo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefixo}_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(historico.history["accuracy"], label="Treino")
    plt.plot(historico.history["val_accuracy"], label="Teste")
    plt.xlabel("Epoca")
    plt.ylabel("Acuracia")
    plt.title(f"Acuracia - {titulo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefixo}_acuracia.png", dpi=150)
    plt.close()


def salvar_fronteira_decisao(modelo, X, Y, caminho, titulo):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grade = np.c_[xx.ravel(), yy.ravel()]
    probabilidades = modelo.predict(grade, verbose=0).reshape(xx.shape)
    classes = (probabilidades >= 0.5).astype(int)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, classes, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title(f"Fronteira de decisao - {titulo}")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()
