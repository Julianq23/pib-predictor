from pathlib import Path
import os
import secrets
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_pib.onnx"
CLAVE_API = os.environ.get("API_KEY", "")

sesion = ort.InferenceSession(str(RUTA_MODELO))
NOMBRE_ENTRADA = sesion.get_inputs()[0].name

app = FastAPI(title="Servicio Cloud - Prediccion del PIB")


class Variables(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    x7: float
    x8: float


@app.get("/")
def estado():
    return {"servicio": "Prediccion del PIB", "estado": "activo", "entradas": 8, "salidas": 1}


@app.post("/predecir")
def predecir(datos: Variables, x_api_key: str = Header(default="")):
    if CLAVE_API == "" or not secrets.compare_digest(x_api_key, CLAVE_API):
        raise HTTPException(status_code=401, detail="Clave de API invalida o ausente")

    entrada = np.array([[datos.x1, datos.x2, datos.x3, datos.x4,
                         datos.x5, datos.x6, datos.x7, datos.x8]], dtype="float32")
    prediccion = sesion.run(None, {NOMBRE_ENTRADA: entrada})[0]
    return {"pib_predicho": round(float(prediccion[0, 0]), 4)}
