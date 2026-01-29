#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN,HDBSCAN
from astropy.coordinates import SkyCoord
from astropy import units as u
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

# Agregar src al path para importar config
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    GAIA_PARALLAX5_10,
    DATOS_RESULTADOS,
    DATOS_RESULTADOS_MODULARIZADO,
    ensure_dir_exists
)

######### variables para clusterizar - variables globales #########

VARIABLES_CLUSTER = ['pmra', 'pmdec']

######### Scaler personalizado con percentiles #########

class PercentileMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.05, upper=0.95):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.p_lower_ = np.percentile(X, self.lower * 100, axis=0)
        self.p_upper_ = np.percentile(X, self.upper * 100, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_scaled = (X - self.p_lower_) / (self.p_upper_ - self.p_lower_)
        return np.clip(X_scaled, 0, 1)


######### funcion que filtra los valores por fuera del rango intercuartilico #########

def eliminar_outliers_iqr(df, columnas):
    """
    Elimina filas del DataFrame que contengan outliers según el rango intercuartílico en las columnas indicadas.

    Parámetros:
    df : pd.DataFrame
        El DataFrame original.
    columnas : list
        Lista de nombres de columnas sobre las cuales aplicar la detección de outliers.

    Retorna:
    pd.DataFrame
        Un nuevo DataFrame sin las filas con outliers en las columnas especificadas.
    """
    df_filtrado = df.copy()
    
    for col in columnas:
        Q1 = df_filtrado[col].quantile(0.25)
        Q3 = df_filtrado[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        df_filtrado = df_filtrado[
            (df_filtrado[col] >= limite_inferior) & (df_filtrado[col] <= limite_superior)
        ]

    return df_filtrado
    
        
######### División del cielo en una malla #########

def dividir_por_cuadros(df, tam_ra=10, tam_dec=10):
    bloques = []
    for ra_min in range(0, 360, tam_ra):
        for dec_min in range(-90, 90, tam_dec):
            ra_max = ra_min + tam_ra
            dec_max = dec_min + tam_dec
            filtro = (
                (df['ra'] >= ra_min) & (df['ra'] < ra_max) &
                (df['dec'] >= dec_min) & (df['dec'] < dec_max)
            )
            bloque = df[filtro].copy()
            if not bloque.empty:
                bloques.append((ra_min, dec_min, bloque))
    return bloques

######### Funcion que carga y preprocesa el archivo #########

def cargar_y_preprocesar(path):
    print(f"[INFO] Cargando datos desde: {path}")
    # lectura de archivo csv
    df = pd.read_csv(path)
    # definicion de escalamiento a aplicar
    scaler = PercentileMinMaxScaler(lower=0.05, upper=0.95)
    # aplicacion de escalamiento en tres dimensiones
    # df[['pmra_norm','pmdec_norm','parallax_norm']] = scaler.fit_transform(df[['pmra','pmdec','parallax']])
    df[['pmra_norm','pmdec_norm','parallax_norm','ra_norm','dec_norm']] = scaler.fit_transform(df[['pmra','pmdec','parallax','ra','dec']])
    print(f"[INFO] Datos cargados y normalizados. Filas: {len(df)}")
    return df

######### Funcion encargada de perturbar las variables a usar #########
    
def perturbar_variables(df):
    df_perturbado = df.copy()

    for col in ['pmra', 'pmdec', 'parallax', 'ra', 'dec']:
        err_col = f"{col}_error"
        ruido = np.random.uniform(-1, 1, size=len(df)) * df[err_col].values
        df_perturbado[col] = df[col] + ruido

    # Correcciones específicas
    df_perturbado['parallax'] = df_perturbado['parallax'].clip(lower=0)
    df_perturbado['ra'] = df_perturbado['ra'].clip(lower=0)

    return df_perturbado
    
######### Funcion que se encarga de aplicar hdbscan #########
    
def procesar_muestra(df, min_pts, max_pts, contador):

    print(f"[DEBUG] Iniciando clustering HDBSCAN en iteración {contador}")
    # definimos hdbscan con los parametros ajustados
    hdb = HDBSCAN(min_cluster_size=min_pts, max_cluster_size=max_pts)
    # nombre de las colummnas normalizadas
    columnas_norm = [f"{v}_norm" for v in VARIABLES_CLUSTER]
    # paso por el algoritmo de hdbscan
    hdb.fit(df[columnas_norm])
    # asignamos las etiquetas de los grupos
    etiquetas = hdb.labels_
    # imprimimos el numero de clusters encontrados aislando el ruido
    n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    print(f"[DEBUG] Clustering terminado. Clusters encontrados: {n_clusters}")
    # incluimos los valores de probabilidades
    probabilidades = hdb.probabilities_
    # seleccionamos los clusters validos al aislar el ruido
    valid_labels = set(etiquetas[etiquetas != -1])
    
    # lista que almacena los resultados de clusterizacion
    resultados = []
    # bucle para empezar a poblar la lista
    for idx, sid, etq, prob in zip(df.index, df.source_id, etiquetas, probabilidades):
        # validacion sobre clusters encontrados y validos
        if etq in valid_labels:
            resultados.append((idx, sid, prob))
    return resultados
    
######### Funcion para actualizar los resultados #########        
def actualizar_datos_acumulados(datos_id, resultados):
    # bucle para poblar con los resultados que se obtienen
    for idx, sid, prob in resultados:
        datos_id[idx]["source_id"] = sid
        datos_id[idx]["conteo"] += 1
        datos_id[idx]["sum_probs"] += prob
        datos_id[idx]["count_probs"] += 1
        
######### Funcion que genera un dataframe a partir de un diccionario #########
        
def construir_dataframe_conteo(datos_id, df_base, contador):
    print(f"[INFO] Construyendo DataFrame de resultados para iteración {contador}")
    # creacion y poblacion del dataframe
    df_conteo = pd.DataFrame({
        "ID": list(datos_id.keys()),
        "source_id": [datos["source_id"] for datos in datos_id.values()],
        "MediaProbabilidad": [datos["sum_probs"] / datos["count_probs"] if datos["count_probs"] > 0 else 0 for datos in datos_id.values()],
        "ConteoAgrupaciones": [datos["conteo"] for datos in datos_id.values()],
        "PresenteEnMuestras": [datos["presente"] for datos in datos_id.values()]
    })
    # union de los resultados con el dataframe inicial
    df_final = df_base.reset_index().merge(df_conteo, how='inner', on='source_id')
    df_final['iteraciones'] = contador
    print(f"[INFO] Registros acumulados hasta ahora: {len(df_final)}")
    return df_final
    
######### Funcion que divide el dataset en muestras de 10 con el 80 porciento de los datos #########
    
def procesar_lote(df_base, min_pts, max_pts, contador, n_muestras, n_cpus):
    # ideal para procesamiento con slurm para usar los 10 nucleos
    print(f"[INFO] Procesando lote {contador} con {n_muestras} muestras en paralelo ({n_cpus} núcleos)")
    # muestreo con el 80 porciento de los datos en total n_muestras dataframes nuevos
    muestras = [df_base.sample(frac=0.8) for _ in range(n_muestras)]
    # paralelizamos la ejecucion sobre cada n_cpus
    resultados = Parallel(n_jobs=n_cpus)(
        delayed(procesar_muestra)(m, min_pts, max_pts, contador) for m in muestras
    )
    return resultados

######## Funcion que en lugar de hacer el muestreo, perturba las variables 
    
def procesar_lote(df_base, min_pts, max_pts, contador, n_muestras, n_cpus, modo="combinado"):
    print(f"[INFO] Procesando lote {contador} | Modo: {modo} | Muestras: {n_muestras} | Núcleos: {n_cpus}")

    def perturbar(df):
        df_pert = df.copy()
        for col in ['pmra', 'pmdec', 'parallax', 'ra', 'dec']:
            err_col = f"{col}_error"
            ruido = np.random.uniform(-1, 1, size=len(df)) * df[err_col].values
            df_pert[col] = df[col] + ruido
        df_pert['parallax'] = df_pert['parallax'].clip(lower=0)
        df_pert['ra'] = df_pert['ra'].clip(lower=0)
        return df_pert

    muestras = []
    for _ in range(n_muestras):
        if modo == "muestreo":
            muestra = df_base.sample(frac=0.8)
        elif modo == "perturbacion":
            muestra = perturbar(df_base)
        elif modo == "combinado":
            muestra = perturbar(df_base.sample(frac=0.8))
        else:
            raise ValueError(f"[ERROR] Modo desconocido: {modo}")
        muestras.append(muestra)

    # Normalización por muestra
    scaler = PercentileMinMaxScaler(lower=0.05, upper=0.95)
    for i in range(len(muestras)):
        cols_original = VARIABLES_CLUSTER
        cols_norm = [f"{v}_norm" for v in cols_original]
        muestras[i][cols_norm] = scaler.fit_transform(muestras[i][cols_original])

    resultados = Parallel(n_jobs=n_cpus)(
        delayed(procesar_muestra)(m, min_pts, max_pts, contador) for m in muestras
    )

    return resultados

    
# def procesar_lote(df_base, min_pts, max_pts, contador, n_muestras, n_cpus):
#     print(f"[INFO] Procesando lote {contador} con {n_muestras} muestras en paralelo ({n_cpus} núcleos)")
#     muestras = [df_base.sample(frac=0.8) for _ in range(n_muestras)]
#     resultados = Parallel(n_jobs=n_cpus)(
#         delayed(procesar_muestra)(m, min_pts, max_pts, contador) for m in muestras
#     )
#     # También regresamos los source_id de las muestras
#     muestras_ids = [set(m['source_id']) for m in muestras]
#     return resultados, muestras_ids

######### Funcion main para ejecutar #########    
def main():
    df_completo = pd.read_csv(str(GAIA_PARALLAX5_10))
    df_completo['parallax_over_error'] = df_completo['parallax']/df_completo['parallax_error']
    df_completo['pmra_over_error'] = df_completo['pmra']/df_completo['pmra_error']
    df_completo['pmdec_over_error'] = df_completo['pmdec']/df_completo['pmdec_error']
    df_completo['ra_over_error'] = df_completo['ra']/df_completo['ra_error']
    df_completo['dec_over_error'] = df_completo['dec']/df_completo['dec_error']
    df_completo['ra_relative_error'] = abs(df_completo['ra_error']/df_completo['ra'])
    df_completo['dec_relative_error'] = abs(df_completo['dec_error']/df_completo['dec'])
    df_completo['pmra_relative_error'] = abs(df_completo['pmra_error']/df_completo['pmra'])
    df_completo['pmdec_relative_error'] = abs(df_completo['pmdec_error']/df_completo['pmdec'])
    df_completo['parallax_relative_error'] = abs(df_completo['parallax_error']/df_completo['parallax'])
    print(len(df_completo))
    df_completo = eliminar_outliers_iqr(df_completo, ['ra_relative_error','dec_relative_error','pmra_relative_error','pmdec_relative_error','parallax_relative_error'])
    print(len(df_completo))
    bloques = dividir_por_cuadros(df_completo, tam_ra=60, tam_dec=60)
    

    min_pts = 40
    max_pts = 1500
    n_muestras = 10
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    for ra_min, dec_min, df_bloque in bloques:
        print(f"\n[RA: {ra_min}° - {ra_min+60}°, DEC: {dec_min}° - {dec_min+60}°]\n longitud de {len(df_bloque)}")
        
        if len(df_bloque) < min_pts:
            print(f"[WARN] Bloque muy pequeño. Se omite.")
            continue

        # Normalización local por bloque
        scaler = PercentileMinMaxScaler(lower=0.05, upper=0.95)
        cols_original = VARIABLES_CLUSTER
        cols_norm = [f"{v}_norm" for v in cols_original]
        df_bloque[cols_norm] = scaler.fit_transform(df_bloque[cols_original])

        # datos_id = defaultdict(lambda: {"source_id": 0, "sum_probs": 0, "count_probs": 0, "conteo": 0})
        datos_id = defaultdict(lambda: {"source_id": 0, "sum_probs": 0, "count_probs": 0, "conteo": 0, "presente": 0})

        lista_datos = []

        for contador in range(1, 41):
            resultados_lote = procesar_lote(
                                                df_bloque, 
                                                min_pts, 
                                                max_pts, 
                                                contador, 
                                                n_muestras, 
                                                n_cpus,
                                                modo="combinado"  # o "muestreo" o "perturbacion"
                                            )

            for resultado in resultados_lote:
                actualizar_datos_acumulados(datos_id, resultado)
            df_final = construir_dataframe_conteo(datos_id, df_bloque, contador)
            
            # resultados_lote, muestras_ids = procesar_lote(df_bloque, min_pts, max_pts, contador, n_muestras, n_cpus)

            # for ids_muestra in muestras_ids:
            #   for sid in ids_muestra:
            #     idx = df_bloque[df_bloque['source_id'] == sid].index
            #     for i in idx:
            #       datos_id[i]["source_id"] = sid
            #       datos_id[i]["presente"] += 1
            # df_final = construir_dataframe_conteo(datos_id, df_bloque, contador)
            ensure_dir_exists(DATOS_RESULTADOS)
            output_prev = DATOS_RESULTADOS / f"hdbscan_ra{ra_min}_dec{dec_min}_prev.csv"
            df_final.to_csv(str(output_prev), index=False)
            lista_datos.append(df_final)

        df_master = pd.concat(lista_datos)
        ensure_dir_exists(DATOS_RESULTADOS)
        output_master = DATOS_RESULTADOS / f"hdbscan_ra{ra_min}_dec{dec_min}.csv"
        df_master.to_csv(str(output_master), index=False)

    print("\n[FINALIZADO] Todos los bloques procesados.")

if __name__ == "__main__":
    main()







