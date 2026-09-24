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


def gerar_resultados_md(linhas, caminho_md, tf_v, keras_v):
    ref, prop = linhas
    dif = (prop["acc"] - ref["acc"]) * 100
    if dif > 0:
        frase = f"O modelo proposto ficou {dif:.2f} ponto(s) acima em acuracia."
    elif dif < 0:
        frase = f"O modelo proposto ficou {abs(dif):.2f} ponto(s) abaixo em acuracia."
    else:
        frase = "Os dois modelos tiveram a mesma acuracia no teste."

    gerado_em = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M %Z")
    conteudo = f"""# Resultados - Redes Neurais (make_moons)

Gerado automaticamente por `src/trabalho_rede_neural.py` em {gerado_em}.
TensorFlow {tf_v} / Keras {keras_v} (CPU - versoes diferentes podem variar ~1pp mesmo com seed fixa).

## Parametros (src/config.py)

- SEED={config.SEED} | amostras={config.N_AMOSTRAS} | ruido={config.RUIDO} | split treino/teste=80/20 estratificado (treino={ref["n_treino"]}, teste={ref["n_teste"]})
- Otimizador={config.OTIMIZADOR} lr={config.TAXA_APRENDIZADO} | loss={config.LOSS} | epocas={config.EPOCAS} | batch={config.BATCH_SIZE}

## Tabela

| Modelo | Arquitetura | Params | Loss teste | Acuracia teste |
|---|---|---:|---:|---:|
| Referencia | {config.ARQUITETURA_REFERENCIA} | {ref["params"]} | {ref["loss"]:.6f} | {ref["acc"] * 100:.2f}% |
| Proposto | {config.ARQUITETURA_PROPOSTA} | {prop["params"]} | {prop["loss"]:.6f} | {prop["acc"] * 100:.2f}% |

{frase}

## Figuras

- `../figuras/01_base_make_moons.png` - base usada
- `../figuras/02_modelo_referencia_loss.png` / `../figuras/02_modelo_referencia_acuracia.png`
- `../figuras/03_modelo_proposto_loss.png` / `../figuras/03_modelo_proposto_acuracia.png`
- `../figuras/04_fronteira_modelo_referencia.png` / `../figuras/05_fronteira_modelo_proposto.png`

## Arquivos brutos

- `resultados.csv` - mesma tabela em CSV
- `historico_referencia.csv` / `historico_proposto.csv` - loss/acc por epoca (para auditar a convergencia)
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
    gerar_resultados_md(linhas, config.DIR_RESULTADOS / "RESULTADOS.md", tf_v, keras_v)

    print("\nResumo")
    print(f"Base: make_moons({config.N_AMOSTRAS}, noise={config.RUIDO}), treino={len(X_treino)}, teste={len(X_teste)}")
    print(f"Referencia -> loss={loss_ref:.6f} | acc={acc_ref * 100:.2f}%")
    print(f"Proposto   -> loss={loss_prop:.6f} | acc={acc_prop * 100:.2f}%")


if __name__ == "__main__":
    main()
