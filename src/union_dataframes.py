#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import pandas as pd
from glob import glob

# Agregar src al path para importar config
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATOS_RESULTADOS_MODULARIZADO,
    ensure_dir_exists
)

# Ruta de los resultados
carpeta_resultados = DATOS_RESULTADOS_MODULARIZADO

# Buscar archivos que coincidan con el patrón correcto
archivos_prev = glob(os.path.join(str(carpeta_resultados), "hdbscan_ra*_dec*_prev.csv"))

if not archivos_prev:
    print("[ERROR] No se encontraron archivos con el patron 'hdbscan_ra*_dec*_prev.csv'.")
    exit()

print(f"[INFO] Se encontraron {len(archivos_prev)} archivos _prev.")

# Cargar y concatenar
dataframes = []
for archivo in archivos_prev:
    print(f"[INFO] Leyendo: {archivo}")
    df = pd.read_csv(archivo)

    # Extraer la regi�n del nombre del archivo
    nombre_archivo = os.path.basename(archivo)
    region = nombre_archivo.replace("hdbscan_", "").replace("_prev.csv", "")
    df["region"] = region  # A�adir columna con la regi�n

    dataframes.append(df)

df_concatenado = pd.concat(dataframes, ignore_index=True)

# Guardar archivo final
ensure_dir_exists(carpeta_resultados)
archivo_salida = carpeta_resultados / "prev_concatenados.csv"
df_concatenado.to_csv(str(archivo_salida), index=False)

print(f"[FINALIZADO] Archivo combinado guardado en: {archivo_salida}")