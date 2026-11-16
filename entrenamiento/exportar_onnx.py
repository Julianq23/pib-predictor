from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import onnxruntime as ort
import tensorflow as tf

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO_KERAS = RAIZ / "modelo" / "modelo_pib.keras"
ARCHIVO_ONNX = RAIZ / "modelo" / "modelo_pib.onnx"

modelo = tf.keras.models.load_model(ARCHIVO_KERAS)

with tempfile.TemporaryDirectory() as carpeta_temporal:
    modelo.export(carpeta_temporal, format="tf_saved_model")
    subprocess.run([sys.executable, "-m", "tf2onnx.convert",
                    "--saved-model", carpeta_temporal,
                    "--output", str(ARCHIVO_ONNX),
                    "--opset", "17"], check=True)

sesion = ort.InferenceSession(str(ARCHIVO_ONNX))
nombre_entrada = sesion.get_inputs()[0].name

ejemplo = np.array([[19.70, 9.76, 9.96, 8.92, 16.57, 18.04, 10.77, 4.71]], dtype="float32")
pred_keras = float(modelo.predict(ejemplo, verbose=0)[0, 0])
pred_onnx = float(sesion.run(None, {nombre_entrada: ejemplo})[0][0, 0])

print("Prediccion con Keras:", round(pred_keras, 4))
print("Prediccion con ONNX: ", round(pred_onnx, 4))
print("Modelo ONNX guardado en:", ARCHIVO_ONNX)
