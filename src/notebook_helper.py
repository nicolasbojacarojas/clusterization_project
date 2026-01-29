"""
Helper de rutas para Jupyter Notebooks.

Este módulo facilita el acceso a las rutas del proyecto desde notebooks,
sin importar la ruta de ejecución.

Uso en notebooks:
    from sys import path
    path.insert(0, '../src')  # Ajusta según la ubicación del notebook
    from config import GAIA_PARALLAX5, DATOS_RESULTADOS, get_data_path
    
    df = pd.read_csv(GAIA_PARALLAX5)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Auto-detección de la ruta del proyecto basada en la ubicación del notebook
def setup_notebook_paths():
    """
    Configura automáticamente las rutas en un notebook.
    Debe llamarse al principio de cada notebook.
    """
    # Obtener el directorio actual del notebook
    notebook_dir = Path.cwd()
    
    # Determinar si estamos en dev/ o en la raíz
    if (notebook_dir / "src").exists():
        # El notebook se ejecuta desde la raíz del proyecto
        src_path = notebook_dir / "src"
    elif (notebook_dir.parent / "src").exists():
        # El notebook se ejecuta desde dev/
        src_path = notebook_dir.parent / "src"
    elif (notebook_dir.parent.parent / "src").exists():
        # El notebook se ejecuta desde un subdirectorio de dev/
        src_path = notebook_dir.parent.parent / "src"
    else:
        raise ValueError("No se pudo encontrar la carpeta 'src' del proyecto")
    
    # Agregar src al path
    sys.path.insert(0, str(src_path))
    
    return src_path.parent  # Retorna la raíz del proyecto

# Funciones útiles para notebooks
def load_data(filename: str, subdirectory: str = "") -> pd.DataFrame:
    """
    Carga un archivo CSV desde la carpeta data/.
    
    Args:
        filename: Nombre del archivo (ej: 'gaia_parallax5.csv')
        subdirectory: Subdirectorio dentro de data/ (ej: 'DatosTotales')
    
    Returns:
        DataFrame cargado
    """
    from config import get_data_path
    
    if subdirectory:
        path = Path(__file__).parent.parent / "data" / subdirectory / filename
    else:
        path = get_data_path(filename)
    
    return pd.read_csv(str(path))

def save_result(df: pd.DataFrame, filename: str, subdirectory: str = "datos_resultados"):
    """
    Guarda un DataFrame en la carpeta data/.
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo
        subdirectory: Subdirectorio dentro de data/ 
    """
    from config import get_output_path
    
    output_path = get_output_path(filename, subdirectory)
    df.to_csv(str(output_path), index=False)
    print(f"✓ Archivo guardado en: {output_path}")

if __name__ == "__main__":
    print("Este módulo debe importarse en un notebook.")
