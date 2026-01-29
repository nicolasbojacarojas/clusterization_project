# Ejemplos de Uso - Rutas y Configuración

## 🚀 Inicio Rápido

### Para Notebooks

**Celda 1 - Setup de rutas**:
```python
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup automático de rutas
sys.path.insert(0, str(Path.cwd().parent / "src"))
from config import (
    GAIA_PARALLAX5,
    GAIA_PARALLAX5_10,
    DATOS_RESULTADOS,
    DATOS_RESULTADOS_MODULARIZADO,
    get_data_path,
    get_output_path
)

print(f"✓ Proyecto: {Path.cwd().parent}")
print(f"✓ GAIA Parallax 5: {GAIA_PARALLAX5.exists()}")
print(f"✓ GAIA Parallax 5-10: {GAIA_PARALLAX5_10.exists()}")
```

**Celda 2 - Cargar datos**:
```python
# Opción A: Usar directamente la constante
df_gaia = pd.read_csv(str(GAIA_PARALLAX5))

# Opción B: Usar get_data_path
df_filtered = pd.read_csv(str(get_data_path("gaia_parallax5_10.csv", "datos_shell")))

print(f"Filas cargadas: {len(df_gaia)}")
```

**Celda 3 - Guardar resultados**:
```python
# Opción A: Usar get_output_path
output = get_output_path("mi_resultado.csv", "datos_resultados")
df_resultado.to_csv(str(output), index=False)

# Opción B: Usar notebook_helper (más simple)
from notebook_helper import save_result
save_result(df_resultado, "mi_resultado.csv", "datos_resultados")
```

---

## 📊 Ejemplos de Casos de Uso

### Caso 1: Análisis en Notebook (dev/analisis_basico.ipynb)

```python
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent / "src"))
from config import GAIA_PARALLAX5, DATOS_RESULTADOS, get_output_path

# Cargar datos
df = pd.read_csv(str(GAIA_PARALLAX5))
print(f"Dataset: {len(df)} filas, {len(df.columns)} columnas")

# Análisis
df_filtrado = df[df['parallax'] > 5]

# Visualización
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(df_filtrado['ra'], df_filtrado['dec'], alpha=0.5)
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
plt.savefig(str(get_output_path("distribucion_RA_DEC.png", "datos_resultados")))

# Guardar datos procesados
df_filtrado.to_csv(str(get_output_path("gaia_parallax5_filtrado.csv", "datos_resultados")), index=False)
```

### Caso 2: Script Batch (src/hdbscan_batch.py)

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd
from sklearn.cluster import HDBSCAN

# Importar config
sys.path.insert(0, str(Path(__file__).parent))
from config import GAIA_PARALLAX5_10, DATOS_RESULTADOS, ensure_dir_exists

def process_data():
    # Cargar datos
    df = pd.read_csv(str(GAIA_PARALLAX5_10))
    
    # Procesar
    clusterer = HDBSCAN(min_cluster_size=50)
    df['cluster'] = clusterer.fit_predict(df[['pmra', 'pmdec']])
    
    # Asegurar que el directorio existe
    ensure_dir_exists(DATOS_RESULTADOS)
    
    # Guardar
    output_path = DATOS_RESULTADOS / "clusters_hdbscan.csv"
    df.to_csv(str(output_path), index=False)
    print(f"✓ Guardado en: {output_path}")

if __name__ == "__main__":
    process_data()
```

### Caso 3: Script SLURM (src/run_hdbscan.sh)

```bash
#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=168:00:0

module load anaconda
source activate cluster_env

# Navegar a src/ (automático con realpath)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "PWD: $(pwd)"
python -u hdbscan_batch.py

echo "✓ Job completado"
```

---

## 🔍 Referencia Rápida de Rutas

### Variables Principales

| Variable | Ruta | Archivo |
|----------|------|--------|
| `PROJECT_ROOT` | `/clusterization_project/` | src/config.py |
| `DATA_DIR` | `data/` | src/config.py |
| `SRC_DIR` | `src/` | src/config.py |
| `DEV_DIR` | `dev/` | src/config.py |
| `GAIA_PARALLAX5` | `data/DatosTotales/gaia_parallax5.csv` | src/config.py |
| `GAIA_PARALLAX5_10` | `data/datos_shell/gaia_parallax5_10.csv` | src/config.py |
| `DATOS_RESULTADOS` | `data/datos_resultados/` | src/config.py |
| `DATOS_RESULTADOS_MODULARIZADO` | `data/datos_resultados_modularizado/` | src/config.py |

### Funciones Principales

| Función | Propósito | Ejemplo |
|---------|-----------|---------|
| `get_data_path(filename)` | Obtiene ruta de archivo en data/ | `get_data_path("archivo.csv", "DatosTotales")` |
| `get_output_path(filename, subdir)` | Obtiene ruta para guardar | `get_output_path("resultado.csv", "datos_resultados")` |
| `ensure_dir_exists(path)` | Crea directorio si no existe | `ensure_dir_exists(DATOS_RESULTADOS)` |

---

## ⚙️ Configuración Avanzada

### Cambiar Ruta de Datos (No recomendado)

Si necesitas cambiar la ruta del proyecto:

```python
# En config.py, modifica:
PROJECT_ROOT = Path(__file__).parent.parent

# O establece una ruta absoluta:
PROJECT_ROOT = Path("/ruta/personalizada/proyecto")
```

### Agregar Nuevos Directorios

En `config.py`:

```python
# Nuevo directorio de datos
DATOS_CUSTOM = DATA_DIR / "datos_custom"

# Agregar al setup:
for dir_path in [..., DATOS_CUSTOM]:
    ensure_dir_exists(dir_path)
```

### Logging Centralizado

En un script:

```python
import logging
from config import get_output_path

# Setup logging
log_file = get_output_path("proceso.log", "datos_resultados")
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Iniciando proceso...")
```

---

## 🐛 Troubleshooting

### Error: "No se encontró GAIA_PARALLAX5"

**Verificación**:
```python
from config import GAIA_PARALLAX5
print(GAIA_PARALLAX5.exists())  # Debe ser True
print(GAIA_PARALLAX5)  # Imprime la ruta exacta
```

**Solución**: Verifica que el archivo existe en `data/DatosTotales/gaia_parallax5.csv`

### Error: "No se pudo encontrar la carpeta 'src'"

**En notebook**:
```python
import sys
from pathlib import Path

# Debugging
print(Path.cwd())  # Muestra dónde estás
print(list(Path.cwd().iterdir()))  # Lista los archivos

# Si estás en dev/, esto debería funcionar:
sys.path.insert(0, str(Path.cwd().parent / "src"))
```

### Error: "Permiso denegado" al guardar

```python
from config import ensure_dir_exists, DATOS_RESULTADOS

# Asegurar que el directorio existe
ensure_dir_exists(DATOS_RESULTADOS)

# Luego guardar
df.to_csv(str(DATOS_RESULTADOS / "archivo.csv"))
```

---

## 📝 Checklist para Nuevos Notebooks

- [ ] Ubicado en `dev/`
- [ ] Celda 1: Setup de rutas (`sys.path.insert...`)
- [ ] Celda 2: Import de config
- [ ] Celda 3: Cargar datos con `str(PATH_VARIABLE)`
- [ ] Guardar con `get_output_path()`
- [ ] Documentado con markdown entre celdas
- [ ] No hardcodear rutas

---

## 📝 Checklist para Scripts Python en `src/`

- [ ] Import: `from config import ...`
- [ ] No `sys.path.insert` (ya está en src/)
- [ ] Usar `ensure_dir_exists()` antes de guardar
- [ ] Rutas con `str(PATH_VARIABLE)`
- [ ] Logging en lugar de print
- [ ] Type hints en funciones
- [ ] Docstrings

---

**Última actualización**: 29 de Enero de 2026
