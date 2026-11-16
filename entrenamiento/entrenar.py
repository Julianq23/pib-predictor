from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEMILLA = 42
CAPAS_OCULTAS = [8, 4]
EPOCAS = 2000
TASA_APRENDIZAJE = 0.01
REGULARIZACION_L2 = 0.01

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO_DATOS = RAIZ / "datos" / "procesados" / "pib_limpio.csv"
ARCHIVO_MODELO = RAIZ / "modelo" / "modelo_pib.keras"
ARCHIVO_METRICAS = RAIZ / "modelo" / "metricas.json"

tf.keras.utils.set_random_seed(SEMILLA)

df = pd.read_csv(ARCHIVO_DATOS)
X = df.drop(columns="Y").to_numpy(dtype="float32")
y = df["Y"].to_numpy(dtype="float32")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=3, random_state=SEMILLA)

normalizador = tf.keras.layers.Normalization()
normalizador.adapt(X_train)

modelo = tf.keras.Sequential()
modelo.add(tf.keras.Input(shape=(8,)))
modelo.add(normalizador)
for neuronas in CAPAS_OCULTAS:
    modelo.add(tf.keras.layers.Dense(neuronas, activation="relu",
               kernel_regularizer=tf.keras.regularizers.l2(REGULARIZACION_L2)))
modelo.add(tf.keras.layers.Dense(1))

modelo.compile(optimizer=tf.keras.optimizers.Adam(TASA_APRENDIZAJE), loss="mse")
modelo.summary()

parada = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True)

historial = modelo.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=EPOCAS, batch_size=len(X_train), callbacks=[parada], verbose=0)


def calcular_metricas(reales, predichos):
    mse = mean_squared_error(reales, predichos)
    return {
        "MAE": round(float(mean_absolute_error(reales, predichos)), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R2": round(float(r2_score(reales, predichos)), 4),
    }


pred_train = modelo.predict(X_train, verbose=0).flatten()
pred_val = modelo.predict(X_val, verbose=0).flatten()

metricas = {
    "epocas_entrenadas": len(historial.history["loss"]),
    "entrenamiento": calcular_metricas(y_train, pred_train),
    "validacion": calcular_metricas(y_val, pred_val),
}

print("Epocas entrenadas:", metricas["epocas_entrenadas"])
print("Entrenamiento:", metricas["entrenamiento"])
print("Validacion:   ", metricas["validacion"])
print("PIB real (validacion):    ", np.round(y_val, 2))
print("PIB predicho (validacion):", np.round(pred_val, 2))

modelo.save(ARCHIVO_MODELO)
ARCHIVO_METRICAS.write_text(json.dumps(metricas, indent=2))
print("Modelo guardado en:", ARCHIVO_MODELO)
