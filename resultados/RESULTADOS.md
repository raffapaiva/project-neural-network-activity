# Resultados - Redes Neurais (make_moons)

Gerado automaticamente por `src/trabalho_rede_neural.py` em 23/09/2026 21:48 Hora oficial do Brasil.
TensorFlow 2.22.0-rc0 / Keras 3.16.0.dev2026092318 (CPU - versoes diferentes podem variar ~1pp mesmo com seed fixa).

## Parametros (src/config.py)

- SEED=42 | amostras=500 | ruido=0.2 | split treino/teste=80/20 estratificado (treino=400, teste=100)
- Otimizador=SGD lr=0.1 | loss=mean_squared_error | epocas=200 | batch=10

## Tabela

| Modelo | Arquitetura | Params | Loss teste | Acuracia teste |
|---|---|---:|---:|---:|
| Referencia | 5 ReLU -> 5 tanh -> 1 sigmoid | 51 | 0.012633 | 98.00% |
| Proposto | 16 ReLU -> 8 ReLU -> 4 tanh -> 1 sigmoid | 225 | 0.010526 | 99.00% |

O modelo proposto ficou 1.00 ponto(s) acima em acuracia.

## Figuras

- `../figuras/01_base_make_moons.png` - base usada
- `../figuras/02_modelo_referencia_loss.png` / `../figuras/02_modelo_referencia_acuracia.png`
- `../figuras/03_modelo_proposto_loss.png` / `../figuras/03_modelo_proposto_acuracia.png`
- `../figuras/04_fronteira_modelo_referencia.png` / `../figuras/05_fronteira_modelo_proposto.png`

## Arquivos brutos

- `resultados.csv` - mesma tabela em CSV
- `historico_referencia.csv` / `historico_proposto.csv` - loss/acc por epoca (para auditar a convergencia)
- `../modelos/referencia.keras` / `../modelos/proposto.keras` - pesos salvos
