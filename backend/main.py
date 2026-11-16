from pathlib import Path
import logging
import os
import time
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CARPETA = Path(__file__).resolve().parent
load_dotenv(CARPETA / ".env")

CLOUD_URL = os.environ.get("CLOUD_URL", "")
CLOUD_API_KEY = os.environ.get("CLOUD_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(CARPETA / "solicitudes.log", encoding="utf-8"),
              logging.StreamHandler()],
)
registro = logging.getLogger("pib-backend")

app = FastAPI(title="API PIB Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Variables(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    x7: float
    x8: float


@app.get("/api/estado")
def estado():
    return {"api": "PIB Predictor", "estado": "activa", "servicio_cloud": CLOUD_URL}


@app.post("/api/predecir")
def predecir(datos: Variables):
    if CLOUD_URL == "" or CLOUD_API_KEY == "":
        raise HTTPException(status_code=500, detail="Falta configurar CLOUD_URL o CLOUD_API_KEY en el archivo .env")

    entrada = datos.model_dump()
    inicio = time.time()

    try:
        respuesta = requests.post(CLOUD_URL, json=entrada,
                                  headers={"x-api-key": CLOUD_API_KEY}, timeout=90)
    except requests.exceptions.Timeout:
        registro.info("ERROR tiempo de espera agotado | entrada=%s", entrada)
        raise HTTPException(status_code=504, detail="El servicio Cloud tardo demasiado en responder")
    except requests.exceptions.RequestException as error:
        registro.info("ERROR de conexion: %s | entrada=%s", error, entrada)
        raise HTTPException(status_code=502, detail="No se pudo conectar con el servicio Cloud")

    duracion = round(time.time() - inicio, 2)

    if respuesta.status_code != 200:
        registro.info("ERROR Cloud respondio %s en %ss | entrada=%s", respuesta.status_code, duracion, entrada)
        raise HTTPException(status_code=502, detail="El servicio Cloud respondio con error " + str(respuesta.status_code))

    pib = respuesta.json()["pib_predicho"]
    registro.info("OK Cloud respondio 200 en %ss | entrada=%s | pib_predicho=%s", duracion, entrada, pib)
    return {"pib_predicho": pib, "origen": "Servicio Cloud", "tiempo_segundos": duracion}
