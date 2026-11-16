from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO_EXCEL = RAIZ / "datos" / "raw" / "base_conocimiento_pib.xls"
ARCHIVO_LIMPIO = RAIZ / "datos" / "procesados" / "pib_limpio.csv"

COLUMNAS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "Y"]

df = pd.read_excel(ARCHIVO_EXCEL, header=None, skiprows=7, nrows=11, usecols="B:J")
df.columns = COLUMNAS

print("Filas y columnas:", df.shape)
print("Celdas vacias:", df.isna().sum().sum())
print("Filas duplicadas:", df.duplicated().sum())
print("Tipos de dato:", df.dtypes.unique())

assert df.shape == (11, 9), "Se esperaban 11 filas y 9 columnas"
assert df.isna().sum().sum() == 0, "Hay celdas vacias"

print(df.round(2))

df.to_csv(ARCHIVO_LIMPIO, index=False)
print("Archivo guardado en:", ARCHIVO_LIMPIO)
