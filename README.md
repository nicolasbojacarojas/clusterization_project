# Clusterización de Datos Astronómicos - GAIA

Proyecto de investigación para el análisis de clusterización de datos astronómicos usando HDBSCAN y algoritmos de muestreo robusto.

## Estructura del Proyecto

```
clusterization_project/
├── data/                              # Datos del proyecto
│   ├── DatosTotales/                 # Dataset principal GAIA
│   ├── datos_shell/                  # Datos filtrados por parallax
│   ├── datos_resultados/             # Resultados de clustering
│   ├── datos_resultados_modularizado/# Resultados modularizados
│   └── datos_resultados_union/       # Resultados unificados
├── dev/                              # Desarrollo y exploración
│   ├── *.ipynb                       # Notebooks Jupyter (análisis exploratorio)
│   └── ruta_imagenes_evolucion/      # Imágenes y visualizaciones
├── src/                              # Código fuente principal
│   ├── config.py                     # Configuración centralizada de rutas
│   ├── notebook_helper.py            # Helper para notebooks
│   ├── hdbscan_batch.py             # Script principal de clustering
│   ├── hdbscan_final.py             # Agregación de resultados
│   ├── union_dataframes.py          # Unión de resultados
│   ├── run_hdbscan.sh               # Script SLURM para batch
│   └── final_run.sh                 # Script SLURM para agregación final
├── app/                              # Dashboards y aplicaciones
├── docs/                             # Documentación
│   └── RUTAS_Y_CONFIGURACION.md     # Guía de rutas y setup
├── test/                             # Tests unitarios
├── AGENTS.md                         # Estándares de código
├── README.md                         # Este archivo
└── .python-version                   # Versión de Python 3.12
```

## Instalación y Setup

### Requisitos

- Python 3.12+
- UV (gestor de paquetes)
- Conda (para HPC)

### Instalación Local

```bash
# Instalar UV (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar el repositorio
git clone <repositorio>
cd clusterization_project

# Instalar dependencias
uv sync

# Activar el entorno virtual
source .venv/bin/activate  # En Linux/Mac
# o
.venv\Scripts\activate     # En Windows
```

### En HPC (SLURM)

```bash
# Los scripts SLURM ya incluyen la carga de módulos
# Solo necesitas ejecutar:
sbatch src/run_hdbscan.sh    # Para clustering batch
sbatch src/final_run.sh       # Para agregación final
```

## Uso

### Scripts Principales

#### 1. hdbscan_batch.py
Realiza clustering en bloques de coordenadas usando HDBSCAN con muestreo robusto.

```bash
python src/hdbscan_batch.py
```

**Entrada**: `data/datos_shell/gaia_parallax5_10.csv`
**Salida**: `data/datos_resultados/hdbscan_ra*.csv`

#### 2. hdbscan_final.py
Aplica HDBSCAN final a los resultados del clustering batch.

```bash
python src/hdbscan_final.py
```

**Entrada**: `data/datos_resultados_modularizado/hdbscan_ra*_dec*_prev.csv`
**Salida**: `data/datos_resultados_modularizado/datos_clusterizados_todos_*.csv`

#### 3. union_dataframes.py
Une los resultados previos.

```bash
python src/union_dataframes.py
```

### Notebooks de Análisis

Los notebooks en `dev/` están organizados por fase de análisis:
- `base_maestra.ipynb`: Preparación de datos
- `analisis_completo_modularizado_*.ipynb`: Análisis multidimensional
- `analisis_resultados_*.ipynb`: Análisis de resultados

Para usar en notebooks:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
from config import GAIA_PARALLAX5, get_data_path

# Cargar datos
df = pd.read_csv(str(GAIA_PARALLAX5))
```

## Configuración de Rutas

El proyecto usa un sistema centralizado de rutas en `src/config.py`:

- **GAIA_PARALLAX5**: `data/DatosTotales/gaia_parallax5.csv`
- **GAIA_PARALLAX5_10**: `data/datos_shell/gaia_parallax5_10.csv`
- **DATOS_RESULTADOS**: `data/datos_resultados/`
- **DATOS_RESULTADOS_MODULARIZADO**: `data/datos_resultados_modularizado/`

Ver [RUTAS_Y_CONFIGURACION.md](docs/RUTAS_Y_CONFIGURACION.md) para más detalles.

## Estándares de Código

El proyecto sigue los estándares definidos en [AGENTS.md](AGENTS.md):

- ✓ Funciones pequeñas y enfocadas (máx 20-30 líneas)
- ✓ Type hints obligatorios
- ✓ Docstrings PEP 257
- ✓ Logging centralizado (no print)
- ✓ Tests con pytest
- ✓ Rutas centralizadas

## Versionamiento y Git

Ramas por feature vinculadas a Jira:

```bash
git checkout -b feature/DADL-XXX  # Tu número de ticket
```

Commits:
```bash
git commit -m ":pencil: DADL-XXX: Descripción clara y concisa"
```

## Troubleshooting

### Rutas no encontradas
- Verifica que `data/` está en la raíz del proyecto
- Usa `from config import ...` en lugar de rutas hardcodeadas

### Scripts SLURM fallando
- Revisa los logs: `cat logs/hdbscan_*.err`
- Asegúrate de que el entorno conda está activado

### Rutas en notebooks
- Los notebooks deben estar en `dev/`
- Agrupa el setup de rutas en una celda al inicio

## Documentación Adicional

- [RUTAS_Y_CONFIGURACION.md](docs/RUTAS_Y_CONFIGURACION.md): Guía detallada de rutas
- [AGENTS.md](AGENTS.md): Estándares de código y mejores prácticas
- `docs/`: Documentación general del proyecto

## Licencia

Ver archivo [LICENSE](LICENSE)

## Contacto

Nicolás Bojacá Rojas
Email: nd.bojaca@uniandes.edu.co
