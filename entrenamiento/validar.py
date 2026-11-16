from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEMILLA = 42
CAPAS_OCULTAS = [4]
EPOCAS = 800
TASA_APRENDIZAJE = 0.01
REGULARIZACION_L2 = 0.5

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO_DATOS = RAIZ / "datos" / "procesados" / "pib_limpio.csv"
ARCHIVO_RESULTADOS = RAIZ / "modelo" / "validacion_cruzada.json"

df = pd.read_csv(ARCHIVO_DATOS)
X = df.drop(columns="Y").to_numpy(dtype="float32")
y = df["Y"].to_numpy(dtype="float32")


def construir_modelo(X_entrenamiento):
    normalizador = tf.keras.layers.Normalization()
    normalizador.adapt(X_entrenamiento)
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.Input(shape=(8,)))
    modelo.add(normalizador)
    for neuronas in CAPAS_OCULTAS:
        modelo.add(tf.keras.layers.Dense(neuronas, activation="relu",
                   kernel_regularizer=tf.keras.regularizers.l2(REGULARIZACION_L2)))
    modelo.add(tf.keras.layers.Dense(1))
    modelo.compile(optimizer=tf.keras.optimizers.Adam(TASA_APRENDIZAJE), loss="mse")
    return modelo


def calcular_metricas(reales, predichos):
    mse = mean_squared_error(reales, predichos)
    return {
        "MAE": round(float(mean_absolute_error(reales, predichos)), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R2": round(float(r2_score(reales, predichos)), 4),
    }


pred_red = []
pred_lineal = []

for i in range(len(X)):
    print("Ronda", i + 1, "de", len(X))
    entrena = np.arange(len(X)) != i
    prueba = ~entrena

    tf.keras.utils.set_random_seed(SEMILLA)
    modelo = construir_modelo(X[entrena])
    modelo.fit(X[entrena], y[entrena], epochs=EPOCAS, batch_size=len(X) - 1, verbose=0)
    pred_red.append(float(modelo.predict(X[prueba], verbose=0)[0, 0]))

    lineal = LinearRegression().fit(X[entrena], y[entrena])
    pred_lineal.append(float(lineal.predict(X[prueba])[0]))

tabla = pd.DataFrame({"PIB_real": y, "red_neuronal": pred_red, "regresion_lineal": pred_lineal})
print(tabla.round(2))

resultados = {
    "red_neuronal": calcular_metricas(y, pred_red),
    "regresion_lineal": calcular_metricas(y, pred_lineal),
}
print("Red neuronal:    ", resultados["red_neuronal"])
print("Regresion lineal:", resultados["regresion_lineal"])

ARCHIVO_RESULTADOS.write_text(json.dumps(resultados, indent=2))
print("Resultados guardados en:", ARCHIVO_RESULTADOS)
