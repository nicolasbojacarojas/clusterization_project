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
    # definimos hdbscan con los parametros ajustados
    hdb = HDBSCAN(min_cluster_size=min_pts, max_cluster_size=max_pts)
    # nombre de las columnas normalizadas
    columnas_norm = [f"{v}_norm" for v in VARIABLES_CLUSTER]
    # paso por el algoritmo de hdbscan
    hdb.fit(df[columnas_norm].values)
    
    # asignamos las etiquetas de los grupos
    etiquetas = hdb.labels_
    # imprimimos el numero de clusters encontrados aislando el ruido
    n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    
    # incluimos los valores de probabilidades
    probabilidades = hdb.probabilities_
    # seleccionamos los clusters validos al aislar el ruido
    valid_labels = set(etiquetas[etiquetas != -1])
    
    if not valid_labels:
        return []
    
    # Usar numpy para operaciones más rápidas
    indices = df.index.values
    source_ids = df['source_id'].values
    
    # Filtrar solo los registros en clusters válidos
    mask = np.isin(etiquetas, list(valid_labels))
    
    resultados = [
        (idx, sid, prob) 
        for idx, sid, prob in zip(indices[mask], source_ids[mask], probabilidades[mask])
    ]
    return resultados
    
######### Funcion para actualizar los resultados #########        
def actualizar_datos_acumulados(datos_arrays, resultados):
    """Actualiza arrays numpy con resultados de clustering (más eficiente para 100k-1M registros)."""
    for idx, sid, prob in resultados:
        datos_arrays['source_id'][idx] = sid
        datos_arrays['conteo'][idx] += 1
        datos_arrays['sum_probs'][idx] += prob
        datos_arrays['count_probs'][idx] += 1
        
######### Funcion que genera un dataframe a partir de arrays numpy #########
        
def construir_dataframe_conteo(datos_arrays, df_base, contador):
    """Construye DataFrame desde arrays numpy (optimizado para 100k-1M registros)."""
    
    # Calcular MediaProbabilidad directamente con numpy
    media_prob = np.divide(
        datos_arrays['sum_probs'], 
        datos_arrays['count_probs'],
        where=(datos_arrays['count_probs'] > 0),
        out=np.zeros_like(datos_arrays['sum_probs'], dtype=np.float32)
    )
    
    df_conteo = pd.DataFrame({
        "source_id": datos_arrays['source_id'],
        "MediaProbabilidad": media_prob,
        "ConteoAgrupaciones": datos_arrays['conteo'],
        "PresenteEnMuestras": datos_arrays['presente']
    })
    
    # Merge más eficiente: usar merge on index
    df_final = df_base.reset_index(drop=True).copy()
    df_final = pd.concat([df_final, df_conteo], axis=1)
    
    df_final['iteraciones'] = contador
    media_presente = datos_arrays['presente'].mean()
    print(f"[INFO] Registros: {len(df_final)} | Media presencia: {media_presente:.1f} muestras")
    
    return df_final
    
######### Funcion que divide el dataset en muestras de 10 con el 80 porciento de los datos #########

def procesar_lote(df_base, min_pts, max_pts, contador, n_muestras, n_cpus, modo="combinado"):
    """
    Procesa un lote con perturbación, muestreo o combinación.
    Retorna tupla (resultados_clustering, muestras_source_ids)
    """
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
    muestras_indices = []  # Para rastrear los índices originales
    
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
        muestras_indices.append(set(muestra.index.values))  # Guardar índices presentes

    # Normalización por muestra
    scaler = PercentileMinMaxScaler(lower=0.05, upper=0.95)
    for i in range(len(muestras)):
        cols_norm = [f"{v}_norm" for v in VARIABLES_CLUSTER]
        muestras[i][cols_norm] = scaler.fit_transform(muestras[i][VARIABLES_CLUSTER])

    resultados = Parallel(n_jobs=n_cpus)(
        delayed(procesar_muestra)(m, min_pts, max_pts, contador) for m in muestras
    )

    return resultados, muestras_indices

######### Funcion para construir dataframe final con razón de exclusión #########

def construir_resultado_final(df_completo, df_procesado, indices_removidos_iqr, indices_removidos_bloque_pequeño):
    """
    Construye dataframe final incluyendo registros no procesados con razón de exclusión.
    """
    # Asegurar que df_procesado tiene índices
    df_procesado = df_procesado.copy()
    df_procesado['razon_exclusion'] = 'Procesado'
    
    # Identificar registros no procesados
    indices_procesados = set(df_procesado.index)
    
    # Dataframe para excluidos
    registros_excluidos = []
    
    # Registros removidos por IQR
    for idx in indices_removidos_iqr:
        if idx in df_completo.index:
            fila = df_completo.loc[idx].copy()
            fila['razon_exclusion'] = 'Removido por IQR (error relativo alto)'
            fila['ConteoAgrupaciones'] = 0
            fila['PresenteEnMuestras'] = 0
            fila['MediaProbabilidad'] = 0.0
            registros_excluidos.append(fila)
    
    # Registros en bloques pequeños
    for idx in indices_removidos_bloque_pequeño:
        if idx in df_completo.index:
            fila = df_completo.loc[idx].copy()
            fila['razon_exclusion'] = 'Bloque muy pequeño (< 40 registros)'
            fila['ConteoAgrupaciones'] = 0
            fila['PresenteEnMuestras'] = 0
            fila['MediaProbabilidad'] = 0.0
            registros_excluidos.append(fila)
    
    # Combinar procesados y excluidos
    if registros_excluidos:
        df_excluidos = pd.DataFrame(registros_excluidos)
        df_final = pd.concat([df_procesado, df_excluidos], ignore_index=False)
    else:
        df_final = df_procesado.copy()
    
    # Asegurar columna razon_exclusion existe
    if 'razon_exclusion' not in df_final.columns:
        df_final['razon_exclusion'] = 'Procesado'
    
######### Funcion main para ejecutar #########    
def main():
    df_completo = pd.read_csv(str(GAIA_PARALLAX5_10))
    df_completo_backup = df_completo.copy()  # Backup para rastrear exclusiones
    
    # Rastrear índices originales
    indices_originales = set(df_completo.index)
    
    # Vectorización: calcular errores relativos de una vez
    error_cols = ['ra', 'dec', 'pmra', 'pmdec', 'parallax']
    for col in error_cols:
        df_completo[f'{col}_relative_error'] = np.abs(df_completo[f'{col}_error'] / df_completo[col])
    
    print(f"[INFO] Registros iniciales: {len(df_completo)}")
    
    # Filtrar outliers IQR y rastrear removidos
    df_filtrado = eliminar_outliers_iqr(df_completo, ['ra_relative_error','dec_relative_error','pmra_relative_error','pmdec_relative_error','parallax_relative_error'])
    indices_removidos_iqr = indices_originales - set(df_filtrado.index)
    print(f"[INFO] Registros tras IQR: {len(df_filtrado)} (removidos: {len(indices_removidos_iqr)})")
    
    bloques = dividir_por_cuadros(df_filtrado, tam_ra=60, tam_dec=60)
    
    # Rastrear registros en bloques pequeños
    indices_en_bloques = set()
    for ra_min, dec_min, df_bloque in bloques:
        indices_en_bloques.update(df_bloque.index)
    
    indices_removidos_bloque_pequeño = set(df_filtrado.index) - indices_en_bloques
    print(f"[INFO] Registros en bloques procesables: {len(indices_en_bloques)} (bloques pequeños: {len(indices_removidos_bloque_pequeño)})")

    min_pts = 40
    max_pts = 1500
    n_muestras = 10
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    
    # Dataframe para acumular resultados procesados
    lista_datos_procesados = []

    for ra_min, dec_min, df_bloque in bloques:
        n_records = len(df_bloque)
        print(f"\n[RA: {ra_min}°-{ra_min+60}°, DEC: {dec_min}°-{dec_min+60}°] Registros: {n_records}")
        
        if n_records < min_pts:
            print(f"[WARN] Bloque omitido (demasiado pequeño)")
            continue

        # Normalización local por bloque
        scaler = PercentileMinMaxScaler(lower=0.05, upper=0.95)
        cols_norm = [f"{v}_norm" for v in VARIABLES_CLUSTER]
        df_bloque[cols_norm] = scaler.fit_transform(df_bloque[VARIABLES_CLUSTER])

        # Inicializar arrays numpy para mejor rendimiento con 100k-1M registros
        n_records = len(df_bloque)
        datos_arrays = {
            'source_id': np.zeros(n_records, dtype=np.int64),
            'conteo': np.zeros(n_records, dtype=np.int16),
            'presente': np.zeros(n_records, dtype=np.int16),
            'sum_probs': np.zeros(n_records, dtype=np.float32),
            'count_probs': np.zeros(n_records, dtype=np.int16)
        }
        
        # Pre-poblar source_id desde df_bloque
        datos_arrays['source_id'] = df_bloque['source_id'].values.copy()

        lista_datos = []

        for contador in range(1, 41):
            resultados_lote, muestras_indices = procesar_lote(df_bloque, min_pts, max_pts, contador, n_muestras, n_cpus, modo="combinado")

            # Actualizar con resultados de clustering y presencia en muestras
            for resultado, indices_presentes in zip(resultados_lote, muestras_indices):
                # Primero actualizar presencia (índices están en un set)
                for idx in indices_presentes:
                    datos_arrays['presente'][idx] += 1
                
                # Luego actualizar agrupaciones
                actualizar_datos_acumulados(datos_arrays, resultado)
            
            df_final = construir_dataframe_conteo(datos_arrays, df_bloque, contador)
            
            ensure_dir_exists(DATOS_RESULTADOS)
            output_prev = DATOS_RESULTADOS / f"hdbscan_ra{ra_min}_dec{dec_min}_prev.csv"
            df_final.to_csv(str(output_prev), index=False)
            lista_datos.append(df_final)

        # Concatenar y guardar con copy=False para optimizar memoria
        df_master = pd.concat(lista_datos, ignore_index=True, copy=False)
        ensure_dir_exists(DATOS_RESULTADOS)
        output_master = DATOS_RESULTADOS / f"hdbscan_ra{ra_min}_dec{dec_min}.csv"
        df_master.to_csv(str(output_master), index=False)
        
        # Acumular para resultado final global
        lista_datos_procesados.append(df_master)
        
        # Limpiar memoria explícitamente
        del df_master, lista_datos, datos_arrays, df_bloque
        
    # Construir resultado final con razones de exclusión
    if lista_datos_procesados:
        df_procesado_global = pd.concat(lista_datos_procesados, ignore_index=False, copy=False)
    else:
        df_procesado_global = pd.DataFrame()
    
    # Agregar razones de exclusión
    df_resultado_final = construir_resultado_final(
        df_completo_backup, 
        df_procesado_global, 
        indices_removidos_iqr, 
        indices_removidos_bloque_pequeño
    )
    
    # Guardar resultado final combinado
    ensure_dir_exists(DATOS_RESULTADOS)
    output_final = DATOS_RESULTADOS / "hdbscan_resultado_final_completo.csv"
    df_resultado_final.to_csv(str(output_final), index=False)
    
    print(f"\n[INFO] RESUMEN FINAL:")
    print(f"  - Registros iniciales: {len(df_completo_backup)}")
    print(f"  - Removidos por IQR: {len(indices_removidos_iqr)}")
    print(f"  - Removidos por bloque pequeño: {len(indices_removidos_bloque_pequeño)}")
    print(f"  - Procesados: {len(df_procesado_global)}")
    print(f"  - Total en resultado final: {len(df_resultado_final)}")
    print(f"\n[FINALIZADO] Resultado guardado en: {output_final}")

if __name__ == "__main__":
    main()







