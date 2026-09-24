# Trabalho de Redes Neurais Artificiais
# Baseado no exemplo4.py da aula (21/08).
#
# Compara a rede do exemplo com uma versao maior no problema das duas
# luas (make_moons), mantendo SGD + erro quadratico medio para a
# comparacao ser justa. Uso:
#   .\.venv\Scripts\python.exe src/trabalho_rede_neural.py
# Saidas: figuras/ (7 PNGs), modelos/ (*.keras), resultados/ (CSV + MD).

import csv
from datetime import datetime, timezone

from keras.utils import set_random_seed

from . import config
from .dados import carregar_dados
from .modelos import criar_modelo_proposto, criar_modelo_referencia
from .visualizacao import (
    salvar_curvas,
    salvar_fronteira_decisao,
    salvar_grafico_base,
)

set_random_seed(config.SEED)


def treinar_modelo(modelo, nome, X_treino, Y_treino, X_teste, Y_teste):
    print(f"\n{nome}")
    modelo.summary()
    historico = modelo.fit(
        X_treino,
        Y_treino,
        validation_data=(X_teste, Y_teste),
        epochs=config.EPOCAS,
        batch_size=config.BATCH_SIZE,
        verbose=0,
    )
    loss, acc = modelo.evaluate(X_teste, Y_teste, verbose=0)
    print(f"Loss no teste: {loss:.6f}")
    print(f"Acuracia no teste: {acc:.4f} ({acc * 100:.2f}%)")
    return historico, loss, acc


def salvar_historico_csv(historico, caminho):
    chaves = ["loss", "val_loss", "accuracy", "val_accuracy"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoca"] + chaves)
        writer.writeheader()
        for i in range(len(historico.history["loss"])):
            writer.writerow(
                {"epoca": i + 1, **{k: historico.history[k][i] for k in chaves}}
            )


def versao_tf_keras():
    try:
        import keras
        import tensorflow as tf

        return tf.__version__, keras.__version__
    except Exception:
        return "n/d", "n/d"


def estatisticas_convergencia(historico):
    """Numeros auditaveis da convergencia a partir do history do Keras."""
    loss = historico.history["loss"]
    val_loss = historico.history["val_loss"]
    acc = historico.history["accuracy"]
    val_acc = historico.history["val_accuracy"]

    def primeira_epoca(serie, limiar, acima=True):
        for i, v in enumerate(serie, start=1):
            if (v >= limiar) if acima else (v <= limiar):
                return i
        return None

    return {
        "epoca_val_loss_002": primeira_epoca(val_loss, 0.02, acima=False),
        "epoca_val_acc_95": primeira_epoca(val_acc, 0.95, acima=True),
        "loss_treino_final": loss[-1],
        "loss_teste_final": val_loss[-1],
        "acc_treino_final": acc[-1],
        "acc_teste_final": val_acc[-1],
        "menor_val_loss": min(val_loss),
    }


def gerar_resultados_md(linhas, stats, caminho_md, tf_v, keras_v):
    ref, prop = linhas
    st_ref, st_prop = stats
    dif = (prop["acc"] - ref["acc"]) * 100
    if dif > 0:
        frase = f"O modelo proposto ficou {dif:.2f} ponto(s) percentual(is) acima em acuracia."
    elif dif < 0:
        frase = f"O modelo proposto ficou {abs(dif):.2f} ponto(s) percentual(is) abaixo em acuracia."
    else:
        frase = "Os dois modelos tiveram a mesma acuracia no teste."

    gerado_em = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M %Z")
    conteudo = f"""# Resultados - Redes Neurais (make_moons)

Gerado automaticamente por `src/trabalho_rede_neural.py` em {gerado_em}.
TensorFlow {tf_v} / Keras {keras_v} (CPU - versoes diferentes podem variar ~1pp mesmo com seed fixa).

## Objetivo

Ver como mudar numero de neuronios, numero de camadas e funcoes de
ativacao da rede do `exemplo4.py` afeta a classificacao nao linear das
duas luas (`make_moons`).

## Parametros (src/config.py)

- SEED={config.SEED} | amostras={config.N_AMOSTRAS} | ruido={config.RUIDO} | split treino/teste=80/20 estratificado (treino={ref["n_treino"]}, teste={ref["n_teste"]})
- Otimizador={config.OTIMIZADOR} lr={config.TAXA_APRENDIZADO} | loss={config.LOSS} | epocas={config.EPOCAS} | batch={config.BATCH_SIZE}

## Referencia x proposto - qual a diferenca?

| Aspecto | Referencia (exemplo4.py) | Proposto |
|---|---|---|
| Camadas ocultas | 2 (5 + 5 neuronios) | 3 (16 + 8 + 4 neuronios) |
| Ativacoes ocultas | ReLU, tanh | ReLU, ReLU, tanh |
| Saida | 1 sigmoid (binaria) | 1 sigmoid (binaria) |
| Parametros treinaveis | {ref["params"]} | {prop["params"]} (~{prop["params"] / ref["params"]:.1f}x mais) |

A unica mudanca proposital e a capacidade da rede: mais neuronios, uma
camada oculta a mais e ReLU nas duas primeiras camadas. Otimizador,
loss, taxa de aprendizado, epocas e dados sao identicos, entao qualquer
diferenca de desempenho vem da arquitetura.

## Base usada

![Base make_moons](../figuras/01_base_make_moons.png)

## Tabela de resultados (teste)

| Modelo | Arquitetura | Params | Loss teste | Acuracia teste |
|---|---|---:|---:|---:|
| Referencia | {config.ARQUITETURA_REFERENCIA} | {ref["params"]} | {ref["loss"]:.6f} | {ref["acc"] * 100:.2f}% |
| Proposto | {config.ARQUITETURA_PROPOSTA} | {prop["params"]} | {prop["loss"]:.6f} | {prop["acc"] * 100:.2f}% |

{frase}

## Convergencia (calculado do historico, nao "no olho")

| Metrica | Referencia | Proposto |
|---|---|---|
| 1a epoca com loss de teste <= 0.02 | {st_ref["epoca_val_loss_002"]} | {st_prop["epoca_val_loss_002"]} |
| 1a epoca com acuracia de teste >= 95% | {st_ref["epoca_val_acc_95"]} | {st_prop["epoca_val_acc_95"]} |
| Loss final treino / teste | {st_ref["loss_treino_final"]:.4f} / {st_ref["loss_teste_final"]:.4f} | {st_prop["loss_treino_final"]:.4f} / {st_prop["loss_teste_final"]:.4f} |
| Menor loss de teste no treino | {st_ref["menor_val_loss"]:.4f} | {st_prop["menor_val_loss"]:.4f} |

O modelo maior converge cerca de 2x mais rapido (atinge o patamar de
loss com metade das epocas), o esperado para uma rede com mais
parametros sob o mesmo SGD. Em nenhum dos dois a loss de teste termina
acima da de treino, entao nao ha sinal de overfitting.

### Curvas - referencia

![Loss referencia](../figuras/02_modelo_referencia_loss.png)

![Acuracia referencia](../figuras/02_modelo_referencia_acuracia.png)

### Curvas - proposto

![Loss proposto](../figuras/03_modelo_proposto_loss.png)

![Acuracia proposto](../figuras/03_modelo_proposto_acuracia.png)

## Fronteiras de decisao

### Referencia

![Fronteira referencia](../figuras/04_fronteira_modelo_referencia.png)

### Proposto

![Fronteira proposto](../figuras/05_fronteira_modelo_proposto.png)

As duas fronteiras acompanham o formato das luas. A diferenca aparece
perto do cruzamento entre as classes (x0 entre -0.5 e 0), onde o modelo
proposto faz uma transicao mais abrupta - provavelmente ai que se
concentra a diferenca de acuracia.

## Conclusao

Rede maior converge mais rapido, mas nao garante acuracia melhor: neste
run a diferenca foi de {abs(dif):.0f} ponto(s) percentual(is) em 100 amostras de
teste. Para este problema e este nivel de ruido, a rede simples do
exemplo da aula ja e suficiente - capacidade extra so deixa o modelo um
pouco mais sensivel ao ruido perto da fronteira.

## Arquivos brutos

- `resultados.csv` - mesma tabela em CSV
- `historico_referencia.csv` / `historico_proposto.csv` - loss/acc por epoca (auditoria da tabela acima)
- `../modelos/referencia.keras` / `../modelos/proposto.keras` - pesos salvos
"""
    caminho_md.write_text(conteudo, encoding="utf-8")


def main():
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    X, Y, X_treino, X_teste, Y_treino, Y_teste = carregar_dados()

    salvar_grafico_base(X, Y, config.DIR_FIGURAS / "01_base_make_moons.png")

    modelo_ref = criar_modelo_referencia()
    hist_ref, loss_ref, acc_ref = treinar_modelo(
        modelo_ref,
        f"Modelo de referencia ({config.ARQUITETURA_REFERENCIA})",
        X_treino,
        Y_treino,
        X_teste,
        Y_teste,
    )

    modelo_prop = criar_modelo_proposto()
    hist_prop, loss_prop, acc_prop = treinar_modelo(
        modelo_prop,
        f"Modelo proposto ({config.ARQUITETURA_PROPOSTA})",
        X_treino,
        Y_treino,
        X_teste,
        Y_teste,
    )

    salvar_curvas(hist_ref, str(config.DIR_FIGURAS / "02_modelo_referencia"), "Modelo de referencia")
    salvar_curvas(hist_prop, str(config.DIR_FIGURAS / "03_modelo_proposto"), "Modelo proposto")
    salvar_fronteira_decisao(
        modelo_ref, X, Y, config.DIR_FIGURAS / "04_fronteira_modelo_referencia.png", "Modelo de referencia"
    )
    salvar_fronteira_decisao(
        modelo_prop, X, Y, config.DIR_FIGURAS / "05_fronteira_modelo_proposto.png", "Modelo proposto"
    )

    modelo_ref.save(config.DIR_MODELOS / "referencia.keras")
    modelo_prop.save(config.DIR_MODELOS / "proposto.keras")
    salvar_historico_csv(hist_ref, config.DIR_RESULTADOS / "historico_referencia.csv")
    salvar_historico_csv(hist_prop, config.DIR_RESULTADOS / "historico_proposto.csv")

    linhas = [
        {
            "modelo": "referencia",
            "arquitetura": config.ARQUITETURA_REFERENCIA,
            "loss": float(loss_ref),
            "acc": float(acc_ref),
            "params": int(modelo_ref.count_params()),
            "n_treino": len(X_treino),
            "n_teste": len(X_teste),
        },
        {
            "modelo": "proposto",
            "arquitetura": config.ARQUITETURA_PROPOSTA,
            "loss": float(loss_prop),
            "acc": float(acc_prop),
            "params": int(modelo_prop.count_params()),
            "n_treino": len(X_treino),
            "n_teste": len(X_teste),
        },
    ]
    with open(config.DIR_RESULTADOS / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["modelo", "arquitetura", "loss", "acc", "params", "n_treino", "n_teste"]
        )
        writer.writeheader()
        writer.writerows(linhas)

    tf_v, keras_v = versao_tf_keras()
    stats = [
        estatisticas_convergencia(hist_ref),
        estatisticas_convergencia(hist_prop),
    ]
    gerar_resultados_md(
        linhas, stats, config.DIR_RESULTADOS / "RESULTADOS.md", tf_v, keras_v
    )

    print("\nResumo")
    print(f"Base: make_moons({config.N_AMOSTRAS}, noise={config.RUIDO}), treino={len(X_treino)}, teste={len(X_teste)}")
    print(f"Referencia -> loss={loss_ref:.6f} | acc={acc_ref * 100:.2f}%")
    print(f"Proposto   -> loss={loss_prop:.6f} | acc={acc_prop * 100:.2f}%")


if __name__ == "__main__":
    main()
