from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

SEMILLA = 42
CAPAS_OCULTAS = [4]
EPOCAS = 800
TASA_APRENDIZAJE = 0.01
REGULARIZACION_L2 = 0.5

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO_DATOS = RAIZ / "datos" / "procesados" / "pib_limpio.csv"
ARCHIVO_MODELO = RAIZ / "modelo" / "modelo_pib.keras"

tf.keras.utils.set_random_seed(SEMILLA)

df = pd.read_csv(ARCHIVO_DATOS)
X = df.drop(columns="Y").to_numpy(dtype="float32")
y = df["Y"].to_numpy(dtype="float32")

normalizador = tf.keras.layers.Normalization()
normalizador.adapt(X)

modelo = tf.keras.Sequential()
modelo.add(tf.keras.Input(shape=(8,)))
modelo.add(normalizador)
for neuronas in CAPAS_OCULTAS:
    modelo.add(tf.keras.layers.Dense(neuronas, activation="relu",
               kernel_regularizer=tf.keras.regularizers.l2(REGULARIZACION_L2)))
modelo.add(tf.keras.layers.Dense(1))

modelo.compile(optimizer=tf.keras.optimizers.Adam(TASA_APRENDIZAJE), loss="mse")
modelo.fit(X, y, epochs=EPOCAS, batch_size=len(X), verbose=0)

predichos = modelo.predict(X, verbose=0).flatten()
print("Ajuste sobre los 11 datos -> MAE:", round(float(mean_absolute_error(y, predichos)), 4),
      "R2:", round(float(r2_score(y, predichos)), 4))

modelo.save(ARCHIVO_MODELO)
print("Modelo final guardado en:", ARCHIVO_MODELO)

modelo_cargado = tf.keras.models.load_model(ARCHIVO_MODELO)
ejemplo = X[:1]
print("Entrada de prueba:", np.round(ejemplo, 2))
print("PIB real:", round(float(y[0]), 2))
print("PIB predicho por el modelo cargado:", round(float(modelo_cargado.predict(ejemplo, verbose=0)[0, 0]), 2))
