# Sweet Lift Taxi — Predicción horaria de demanda

## Descripción
Proyecto de series temporales para **Sweet Lift Taxi**, compañía de taxis en aeropuertos. El objetivo es predecir la cantidad de pedidos de taxi para la próxima hora, de manera que el operador pueda asignar más conductores durante las horas pico.

**Objetivo:** RECM (RMSE) en el conjunto de prueba debe ser **<= 48**.

## Dataset
- **Archivo:** `taxi.csv`
- **Periodo:** 1 de marzo – 31 de agosto de 2018
- **Granularidad:** pedidos sub-horarios, **remuestreados a intervalos de 1 hora** (suma)
- **Registros tras resample:** 4,416 observaciones horarias
- **Columna objetivo:** `num_orders`
- **Estadísticos:** media 84.4, std 45.0, mín 0, máx 462

## Resultados
| # | Modelo | RECM Train | RECM Test | Tiempo Entrenamiento (s) | Tiempo Predicción (s) | Cumple ≤ 48 |
|---|--------|-----------:|----------:|-------------------------:|----------------------:|:-----------:|
| 1 | LinearRegression | 25.70 | 45.81 | 0.09 | 0.001 | ✅ |
| 2 | RandomForest (n=100) | 8.41 | 42.80 | 0.44 | 0.035 | ✅ |
| 3 | **LightGBM (early stopping)** | **3.87** | **39.67** | 1.04 | 0.004 | ✅ |

**Mejor modelo:** LightGBM con early stopping (mejor iteración 816 / 10,000) — **RECM 39.67**, muy por debajo del umbral de 48.

## Pipeline
1. **Carga y remuestreo** — leer `taxi.csv`, ordenar por fecha, remuestrear a intervalos de 1 hora sumando los pedidos.
2. **EDA** — gráfica de la serie completa, estadísticos descriptivos, descomposición estacional (`statsmodels.seasonal_decompose`).
3. **Ingeniería de características** (`make_features`):
   - Variables de calendario: `year`, `month`, `day`, `dayofweek`, `hour`
   - **24 lags horarios** (`lag_1` … `lag_24`) — captura el ciclo diario completo
   - **Media móvil de ventana 10** (con `shift()` para evitar fuga de datos)
4. **División train/test** — 90 / 10 cronológica (`shuffle=False`); `dropna()` después de generar los lags.
5. **Escalado** — `StandardScaler` ajustado solo en train.
6. **Entrenamiento** — tres modelos con tiempo registrado:
   - LinearRegression (baseline)
   - RandomForestRegressor (`n_estimators=100`, `n_jobs=-1`)
   - LGBMRegressor con `early_stopping(200)` y `log_evaluation(200)`
7. **Evaluación** — RECM en train y test, tabla de resultados con tiempos y cumplimiento de umbral.

## Hallazgos del EDA
- **Tendencia:** alza clara de marzo a agosto (los picos crecen de ~175 a >400).
- **Estacionalidad diaria:** ciclo fuerte de 24 horas (horas pico y valle).
- **Estacionalidad semanal:** forma de "U" dentro de cada semana.
- **Heterocedasticidad:** la amplitud de las oscilaciones crece con la media.
- **Residuos:** sin patrones evidentes — la descomposición captura bien la estructura.

## Stack técnico
| Categoría | Herramientas |
|-----------|--------------|
| ML | scikit-learn, LightGBM |
| Series temporales | statsmodels (`seasonal_decompose`) |
| Datos | pandas, NumPy |
| Visualización | matplotlib |
| Preprocesamiento | StandardScaler, train_test_split (`shuffle=False`) |

## Estructura
```
Sprint 16/
├── README.md                  # English version
├── README-ESP.md              # Versión en español
├── Series temporales.ipynb    # Notebook principal
└── taxi.csv                   # Dataset
```

## Cómo ejecutarlo
1. Activar un entorno de Python con las dependencias necesarias:
   ```
   pip install pandas numpy matplotlib scikit-learn lightgbm statsmodels jupyter
   ```
2. Abrir y ejecutar el notebook:
   ```
   jupyter notebook "Series temporales.ipynb"
   ```
   (Kernel → Restart & Run All. Tiempo total: ~5 segundos.)

## Conclusiones
- Los tres modelos cumplen el objetivo de RECM ≤ 48.
- **LightGBM** es el ganador con RECM 39.67 en test gracias al early stopping; excelente equilibrio entre precisión y velocidad.
- **RandomForest** alcanza 42.80 pero muestra una brecha train/test mayor (sobreajuste).
- **LinearRegression** es el más rápido y simple, pero el menos preciso (45.81).
- **Modelo recomendado para producción:** LightGBM — menor error en test, predicción en menos de un segundo y robusto frente al sobreajuste gracias al early stopping.

## Autor
Fernando Muerza — TripleTen Data Science, Sprint 16.
