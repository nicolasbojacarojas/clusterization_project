# Guía de Actualización de Rutas en Notebooks

## Resumen de Cambios

El proyecto ha sido restructurado de forma modular. Se han creado dos módulos centralizados para gestionar rutas:

1. **`src/config.py`**: Módulo principal de configuración de rutas
2. **`src/notebook_helper.py`**: Helper especializado para notebooks

## Para Notebooks en `dev/`

### Opción 1: Usar el Helper (Recomendado)

En la **primera celda** de tu notebook, agrega:

```python
import sys
from pathlib import Path

# Setup automático de rutas
sys.path.insert(0, str(Path.cwd().parent / "src"))
from notebook_helper import setup_notebook_paths, load_data, save_result

# Configura las rutas del proyecto
project_root = setup_notebook_paths()
from config import GAIA_PARALLAX5, DATOS_RESULTADOS, get_data_path
```

Luego usa:

```python
# Para cargar datos
df = load_data("gaia_parallax5.csv", "DatosTotales")
# O directamente con config
df = pd.read_csv(GAIA_PARALLAX5)

# Para guardar resultados
save_result(df, "mi_archivo.csv", "datos_resultados")
```

### Opción 2: Usar config.py directamente

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
from config import (
    GAIA_PARALLAX5,
    GAIA_PARALLAX5_5,
    GAIA_PARALLAX5_7,
    GAIA_PARALLAX5_10,
    DATOS_RESULTADOS,
    DATOS_RESULTADOS_MODULARIZADO,
    get_data_path,
    get_output_path
)

# Cargar datos
df = pd.read_csv(str(GAIA_PARALLAX5))

# Guardar en la salida
output_path = get_output_path("mi_archivo.csv", "datos_resultados")
df.to_csv(str(output_path), index=False)
```

## Cambios en Scripts `.py` en `src/`

Los scripts principales ya han sido actualizados:
- `hdbscan_batch.py` ✓
- `hdbscan_final.py` ✓
- `union_dataframes.py` ✓

Ahora usan `config.py` para rutas relativas correctas.

## Cambios en Scripts `.sh` en `src/`

Los scripts SLURM ahora:
- Se navegan a la carpeta `src/` automáticamente
- Usan rutas relativas correctas en los scripts Python
- Logs de debug mejorados

Archivos actualizados:
- `run_hdbscan.sh` ✓
- `final_run.sh` ✓

## Estructura de Carpetas Actualizada

```
clusterization_project/
├── data/                                  # Todos los datos
│   ├── DatosTotales/                     # Dataset principal
│   │   └── gaia_parallax5.csv
│   ├── datos_shell/                      # Datos filtrados
│   │   ├── gaia_parallax5_5.csv
│   │   ├── gaia_parallax5_7.csv
│   │   └── gaia_parallax5_10.csv
│   ├── datos_resultados/                 # Resultados de clustering
│   ├── datos_resultados_modularizado/    # Resultados modularizados
│   └── datos_resultados_union/           # Resultados unidos
├── dev/                                  # Desarrollo
│   ├── *.ipynb                          # Notebooks (exploración)
│   └── ruta_imagenes_evolucion/         # Imágenes generadas
├── src/                                  # Código principal
│   ├── config.py                        # ✓ Rutas centralizadas
│   ├── notebook_helper.py               # ✓ Helper para notebooks
│   ├── hdbscan_batch.py                 # ✓ Script principal
│   ├── hdbscan_final.py                 # ✓ Script de agregación
│   ├── union_dataframes.py              # ✓ Script de unión
│   ├── run_hdbscan.sh                   # ✓ SLURM batch
│   └── final_run.sh                     # ✓ SLURM final
├── app/                                 # Dashboards (Streamlit)
├── docs/                                # Documentación
├── test/                                # Tests
└── README.md                            # Descripción del proyecto
```

## Próximos Pasos

1. **Actualiza tus notebooks**: Agrupa el código de setup de rutas en una celda
2. **Prueba los scripts**: Ejecuta `python -u src/hdbscan_batch.py` desde la raíz
3. **Verifica las rutas**: Los prints de debug muestran las rutas usadas
4. **Documentación**: Agrega docstrings a nuevas funciones

## Troubleshooting

### Error: "No se pudo encontrar la carpeta 'src'"
- Asegúrate de ejecutar el notebook desde `dev/` o la raíz del proyecto
- Verifica que tu notebook está en el lugar correcto

### Error: "No such file or directory"
- Usa `str()` al pasar Path a funciones que esperen strings
- Verifica que `config.py` está en `src/`

### Rutas relativas en HPC
- Los scripts `.sh` navegan a `src/` automáticamente
- Los scripts Python usan rutas relativas desde `src/`
- Los datos están en `../data/` desde `src/`
