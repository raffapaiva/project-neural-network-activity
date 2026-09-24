# Trabalho de Redes Neurais Artificiais
# Baseado no exemplo4.py usado em aula (21/08).
#
# A proposta e pegar a rede do exemplo e comparar com uma versao maior,
# pra ver se aumentar neuronios/camadas realmente melhora a classificacao
# no problema das duas luas (make_moons). Mantivemos o otimizador SGD e o
# erro quadratico medio do exemplo original pra comparacao ficar justa.

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.utils import set_random_seed

SEED = 42
set_random_seed(SEED)

N_AMOSTRAS = 500
RUIDO = 0.20
TAXA_APRENDIZADO = 0.1
EPOCAS = 200
BATCH_SIZE = 10

# o exemplo original usa make_moons(100, noise=0.1); aqui aumentamos pra 500
# amostras e ruido 0.20 pra deixar o problema um pouco mais dificil
X, Y = datasets.make_moons(n_samples=N_AMOSTRAS, noise=RUIDO, random_state=SEED)

X_treino, X_teste, Y_treino, Y_teste = train_test_split(
    X, Y, test_size=0.20, random_state=SEED, stratify=Y
)


def salvar_grafico_base():
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("Base make_moons - 500 amostras e ruido 0.20")
    plt.tight_layout()
    plt.savefig("01_base_make_moons.png", dpi=150)
    plt.close()


def criar_modelo_referencia():
    # mesma arquitetura do exemplo4.py, usada como base de comparacao
    modelo = Sequential([
        Input(shape=(2,)),
        Dense(5, activation="relu"),
        Dense(5, activation="tanh"),
        Dense(1, activation="sigmoid"),
    ])
    otimizador = SGD(learning_rate=TAXA_APRENDIZADO)
    modelo.compile(loss="mean_squared_error", optimizer=otimizador, metrics=["accuracy"])
    return modelo


def criar_modelo_proposto():
    # versao modificada: mais neuronios, uma camada oculta a mais e ReLU
    # nas duas primeiras camadas em vez de so na primeira
    modelo = Sequential([
        Input(shape=(2,)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(4, activation="tanh"),
        Dense(1, activation="sigmoid"),
    ])
    otimizador = SGD(learning_rate=TAXA_APRENDIZADO)
    modelo.compile(loss="mean_squared_error", optimizer=otimizador, metrics=["accuracy"])
    return modelo


def treinar_modelo(modelo, nome):
    print(f"\n{nome}")
    modelo.summary()

    historico = modelo.fit(
        X_treino,
        Y_treino,
        validation_data=(X_teste, Y_teste),
        epochs=EPOCAS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    loss, acc = modelo.evaluate(X_teste, Y_teste, verbose=0)
    print(f"Loss no teste: {loss:.6f}")
    print(f"Acuracia no teste: {acc:.4f} ({acc * 100:.2f}%)")

    return historico, loss, acc


def salvar_curvas(historico, nome_arquivo, titulo):
    plt.figure(figsize=(7, 5))
    plt.plot(historico.history["loss"], label="Treino")
    plt.plot(historico.history["val_loss"], label="Teste")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title(f"Loss - {titulo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{nome_arquivo}_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(historico.history["accuracy"], label="Treino")
    plt.plot(historico.history["val_accuracy"], label="Teste")
    plt.xlabel("Epoca")
    plt.ylabel("Acuracia")
    plt.title(f"Acuracia - {titulo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{nome_arquivo}_acuracia.png", dpi=150)
    plt.close()


def salvar_fronteira_decisao(modelo, nome_arquivo, titulo):
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
    plt.savefig(nome_arquivo, dpi=150)
    plt.close()


def main():
    salvar_grafico_base()

    modelo_referencia = criar_modelo_referencia()
    hist_ref, loss_ref, acc_ref = treinar_modelo(
        modelo_referencia, "Modelo de referencia (5 ReLU -> 5 tanh -> 1 sigmoid)"
    )

    modelo_proposto = criar_modelo_proposto()
    hist_prop, loss_prop, acc_prop = treinar_modelo(
        modelo_proposto, "Modelo proposto (16 ReLU -> 8 ReLU -> 4 tanh -> 1 sigmoid)"
    )

    salvar_curvas(hist_ref, "02_modelo_referencia", "Modelo de referencia")
    salvar_curvas(hist_prop, "03_modelo_proposto", "Modelo proposto")

    salvar_fronteira_decisao(
        modelo_referencia, "04_fronteira_modelo_referencia.png", "Modelo de referencia"
    )
    salvar_fronteira_decisao(
        modelo_proposto, "05_fronteira_modelo_proposto.png", "Modelo proposto"
    )

    print("\nResumo")
    print(f"Base: make_moons({N_AMOSTRAS}, noise={RUIDO}), treino={len(X_treino)}, teste={len(X_teste)}")
    print(f"Referencia -> loss={loss_ref:.6f} | acc={acc_ref * 100:.2f}%")
    print(f"Proposto   -> loss={loss_prop:.6f} | acc={acc_prop * 100:.2f}%")

    diferenca = (acc_prop - acc_ref) * 100
    if diferenca > 0:
        print(f"O modelo proposto ficou {diferenca:.2f} ponto(s) percentual(is) acima em acuracia.")
    elif diferenca < 0:
        print(f"O modelo proposto ficou {abs(diferenca):.2f} ponto(s) percentual(is) abaixo em acuracia.")
    else:
        print("Os dois modelos tiveram a mesma acuracia no teste.")


if __name__ == "__main__":
    main()
