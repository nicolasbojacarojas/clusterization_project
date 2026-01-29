"""
Módulo de configuración centralizado para rutas del proyecto.

Este módulo gestiona todas las rutas del proyecto de manera centralizada,
permitiendo que los scripts y notebooks usen rutas relativas correctas
sin importar dónde se ejecuten.
"""

from pathlib import Path
import os

# Obtener el directorio raíz del proyecto
# Asume que este archivo está en src/ y el proyecto está en ../
PROJECT_ROOT = Path(__file__).parent.parent

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
DEV_DIR = PROJECT_ROOT / "dev"
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"
DOCS_DIR = PROJECT_ROOT / "docs"
TEST_DIR = PROJECT_ROOT / "test"

# Subdirectorios de datos
DATOS_TOTALES = DATA_DIR / "DatosTotales"
DATOS_SHELL = DATA_DIR / "datos_shell"
DATOS_RESULTADOS = DATA_DIR / "datos_resultados"
DATOS_RESULTADOS_MODULARIZADO = DATA_DIR / "datos_resultados_modularizado"
DATOS_RESULTADOS_UNION = DATA_DIR / "datos_resultados_union"
RUTA_IMAGENES_EVOLUCION = DEV_DIR / "ruta_imagenes_evolucion"

# Archivos de datos principales
GAIA_PARALLAX5 = DATOS_TOTALES / "gaia_parallax5.csv"
GAIA_PARALLAX5_5 = DATOS_SHELL / "gaia_parallax5_5.csv"
GAIA_PARALLAX5_7 = DATOS_SHELL / "gaia_parallax5_7.csv"
GAIA_PARALLAX5_10 = DATOS_SHELL / "gaia_parallax5_10.csv"

# Funciones útiles
def get_project_root() -> Path:
    """Retorna la ruta raíz del proyecto."""
    return PROJECT_ROOT

def ensure_dir_exists(directory: Path) -> Path:
    """Crea el directorio si no existe y retorna la ruta."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def get_data_path(filename: str) -> Path:
    """
    Obtiene la ruta de un archivo en la carpeta data.
    
    Args:
        filename: Nombre del archivo o ruta relativa dentro de data/
    
    Returns:
        Path: Ruta completa al archivo
    """
    return DATA_DIR / filename

def get_output_path(filename: str, subdirectory: str = "") -> Path:
    """
    Obtiene la ruta para guardar un archivo de salida.
    
    Args:
        filename: Nombre del archivo a guardar
        subdirectory: Subdirectorio dentro de data/ (ej: 'datos_resultados')
    
    Returns:
        Path: Ruta completa para el archivo
    """
    if subdirectory:
        output_dir = DATA_DIR / subdirectory
    else:
        output_dir = DATA_DIR
    ensure_dir_exists(output_dir)
    return output_dir / filename

# Verificar que los directorios existan
for dir_path in [DATOS_TOTALES, DATOS_SHELL, DATOS_RESULTADOS, 
                 DATOS_RESULTADOS_MODULARIZADO, DATOS_RESULTADOS_UNION]:
    ensure_dir_exists(dir_path)

if __name__ == "__main__":
    print(f"Raíz del proyecto: {PROJECT_ROOT}")
    print(f"Directorio data: {DATA_DIR}")
    print(f"Gaia Parallax 5: {GAIA_PARALLAX5}")
    print(f"Gaia Parallax 5-10: {GAIA_PARALLAX5_10}")
