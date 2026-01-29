# 📊 Carpeta de Datos

Esta carpeta contiene todos los datos del proyecto. **Los archivos de datos NO se suben al repositorio** (protegidos por `.gitignore`).

## 📁 Estructura de Directorios

### DatosTotales/
Dataset principal de GAIA Parallax.

```
gaia_parallax5.csv          # Dataset principal (NO se sube)
```

### datos_shell/
Datos filtrados por rango de parallax.

```
gaia_parallax5_5.csv        # Datos con parallax 5-7
gaia_parallax5_7.csv        # Datos con parallax 7-10
gaia_parallax5_10.csv       # Datos con parallax >= 10 (USADO EN BATCH)
```

### datos_resultados/
Resultados del clustering batch.

```
hdbscan_ra*_dec*_prev.csv   # Resultados previos por bloque
hdbscan_ra*_dec*.csv        # Resultados finales por bloque
```

### datos_resultados_modularizado/
Resultados procesados con HDBSCAN.

```
hdbscan_ra*_dec*_prev.csv   # Datos previos modularizados
datos_clusterizados_todos_2d.csv    # Clustering final 2D
datos_clusterizados_todos_3d.csv    # Clustering final 3D
datos_clusterizados_todos_5d.csv    # Clustering final 5D
```

### datos_resultados_union/
Resultados unidos y consolidados.

```
prev_concatenados.csv       # Todos los datos previos unidos
```

## ⚙️ Cómo Obtener los Datos

1. **Dataset principal**: Coloca `gaia_parallax5.csv` en `DatosTotales/`
2. **Ejecuta**: `python src/base_maestra.ipynb` para generar `datos_shell/`
3. **Ejecuta**: `sbatch src/run_hdbscan.sh` para clustering en HPC
4. **Ejecuta**: `sbatch src/final_run.sh` para resultados finales

## 🔐 Política de Datos

- ✅ Los datos CSV se ignoran en git
- ✅ Las imágenes PNG/JPG se ignoran en git
- ✅ Los pickles y otros binarios se ignoran en git
- ✅ La estructura de directorios SÍ se controla con git (`.gitkeep`)

## 📖 Acceso desde Python

```python
from config import (
    GAIA_PARALLAX5,
    GAIA_PARALLAX5_10,
    DATOS_RESULTADOS,
    DATOS_RESULTADOS_MODULARIZADO
)

# Cargar datos
df = pd.read_csv(str(GAIA_PARALLAX5))
```

## ⚠️ Importante

- No subas archivos CSV grandes a GitHub
- Usa `.gitignore` para proteger datos
- Documenta dónde obtener los datos originales
- Para colaboradores: instrucciones en README.md principal

---

Última actualización: 29 de Enero de 2026
