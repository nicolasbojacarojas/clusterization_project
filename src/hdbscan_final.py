#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd
import glob
import re
from sklearn.cluster import DBSCAN,HDBSCAN
import os
import logging

# Agregar src al path para importar config
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATOS_RESULTADOS_MODULARIZADO,
    ensure_dir_exists
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ruta donde están los archivos CSV
ruta_archivos = str(DATOS_RESULTADOS_MODULARIZADO)

# Buscar todos los archivos que cumplen con el patrón
archivos = glob.glob(os.path.join(ruta_archivos, "hdbscan_ra*_dec*_prev.csv"))
logging.info(f"Se encontraron {len(archivos)} archivos para procesar.")
dataframes = []

# Procesar cada archivo
for archivo in archivos:
    logging.info(f"Procesando archivo: {archivo}")
    # Extraer coordenadas ra y dec usando regex
    match = re.search(r"hdbscan_ra(-?\d+)_dec(-?\d+)_prev\.csv", os.path.basename(archivo))
    if not match:
        logging.warning(f"No se pudieron extraer coordenadas de {archivo}. Saltando.")
        continue

    ra_val = int(match.group(1))
    dec_val = int(match.group(2))
    logging.info(f"Coordenadas extraídas - RA: {ra_val}, DEC: {dec_val}")

    # Cargar datos
    try:
        df = pd.read_csv(archivo)
    except Exception as e:
        logging.error(f"Error al leer {archivo}: {e}")
        continue

    # Verificar que las columnas necesarias existen
    columnas = ['pmdec_norm', 'pmra_norm']
    if not all(col in df.columns for col in columnas):
        logging.warning(f"Faltan columnas necesarias en {archivo}. Saltando.")
        continue

    # Aplicar HDBSCAN
    try:
        clusterer = HDBSCAN(min_cluster_size=60, max_cluster_size=1000)
        df['grupo'] = clusterer.fit_predict(df[columnas])
        logging.info(f"Clustering completado. Número de grupos encontrados: {df['grupo'].nunique()}")
    except Exception as e:
        logging.error(f"Error al aplicar HDBSCAN en {archivo}: {e}")
        continue

    # Agregar coordenadas
    df['coordenada_ra'] = ra_val
    df['coordenada_dec'] = dec_val
    # Agregar a la lista
    dataframes.append(df)

# Concatenar todo en un solo DataFrame
if dataframes:
    df_total = pd.concat(dataframes, ignore_index=True)
    ensure_dir_exists(DATOS_RESULTADOS_MODULARIZADO)
    output_file = DATOS_RESULTADOS_MODULARIZADO / f"datos_clusterizados_todos_{len(columnas)}d.csv"
    df_total.to_csv(str(output_file), index=False)
    logging.info(f"Archivo final guardado como {output_file}")
else:
    logging.warning("No se generaron datos. Ningún archivo fue procesado correctamente.")








