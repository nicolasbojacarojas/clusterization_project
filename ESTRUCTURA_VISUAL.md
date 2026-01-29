# 🏗️ Estructura Final del Proyecto

```
clusterization_project/
│
├── 📄 README.md                          ← COMIENZA AQUÍ
├── 📄 AGENTS.md                          ← Estándares de código
├── 📄 RESUMEN_RESTRUCTURACION.md         ← Resumen de cambios
├── .git/                                 ← Repository git
│
├── 📂 src/  [Código Principal]
│   ├── 🐍 config.py                     ← ✨ RUTAS CENTRALIZADAS
│   ├── 🐍 notebook_helper.py            ← ✨ Helper para notebooks
│   ├── 🐍 hdbscan_batch.py              ← Clustering batch
│   ├── 🐍 hdbscan_final.py              ← Agregación final
│   ├── 🐍 union_dataframes.py           ← Unión de resultados
│   ├── 📜 run_hdbscan.sh                ← SLURM batch
│   ├── 📜 final_run.sh                  ← SLURM final
│   └── 📄 README.md                     ← Docs de src/
│
├── 📂 dev/  [Notebooks - Exploración]
│   ├── 📓 base_maestra.ipynb
│   ├── 📓 analisis_completo_modularizado_2d.ipynb
│   ├── 📓 analisis_completo_modularizado_3d.ipynb
│   ├── 📓 analisis_completo_modularizado_5d.ipynb
│   ├── 📓 analisis_completo_modularizado_5dc2.ipynb
│   ├── 📓 analisis_completo_modularizado_5dc3.ipynb
│   ├── 📓 analisis_completo_modularizado_5derr.ipynb
│   ├── 📓 analisis_resultados_modularizados.ipynb
│   ├── 📓 analisis_resultados_modularizados_hyades.ipynb
│   ├── 📓 analisis_xyz_division_cumulos_hyades.ipynb
│   ├── 📓 analisisparametross.ipynb
│   ├── 📓 division_resultados_modularizados.ipynb
│   ├── 📓 division_resultados_modularizados_coma.ipynb
│   ├── 📓 division_resultados_modularizados_hyades.ipynb
│   ├── 📓 evolucion_probabilidad.ipynb
│   ├── 📓 ultima_division_3d.ipynb
│   ├── 📓 union_registros.ipynb
│   ├── 📄 notebooks_overview.md
│   └── 📂 ruta_imagenes_evolucion/     ← Imágenes generadas
│
├── 📂 data/  [Datos - Centro de Gravedad]
│   ├── 📂 DatosTotales/
│   │   └── 📊 gaia_parallax5.csv       ← Dataset principal
│   │
│   ├── 📂 datos_shell/
│   │   ├── 📊 gaia_parallax5_5.csv
│   │   ├── 📊 gaia_parallax5_7.csv
│   │   └── 📊 gaia_parallax5_10.csv    ← Used in hdbscan_batch
│   │
│   ├── 📂 datos_resultados/
│   │   ├── 📊 hdbscan_ra*.csv
│   │   └── 📊 hdbscan_ra*_prev.csv
│   │
│   ├── 📂 datos_resultados_modularizado/
│   │   ├── 📊 hdbscan_ra*_dec*_prev.csv
│   │   └── 📊 datos_clusterizados_todos_*.csv
│   │
│   ├── 📂 datos_resultados_union/
│   │   └── 📊 prev_concatenados.csv
│   │
│   └── 📄 data_overview.md
│
├── 📂 docs/  [Documentación]
│   ├── 📄 RUTAS_Y_CONFIGURACION.md     ← Guía de rutas (LEER)
│   ├── 📄 EJEMPLOS_USO.md               ← Ejemplos prácticos
│   ├── 📄 VERIFICACION_ESTRUCTURA.md    ← Checklist de cambios
│   └── 📄 [Documentación adicional]
│
├── 📂 app/   [Dashboards - Para luego]
│   └── (Streamlit, Plotly Dash, etc.)
│
└── 📂 test/  [Tests]
    └── (pytest, unittest, etc.)
```

---

## 🎯 Punto de Entrada por Rol

### 👨‍💻 Para Desarrolladores (Scripts)

```
1. Lee: README.md
2. Lee: AGENTS.md (estándares)
3. Ubica código en: src/
4. Rutas: Importa de src/config.py
5. Ejemplo: python src/hdbscan_batch.py
```

### 📊 Para Análisis Exploratorio (Notebooks)

```
1. Lee: README.md
2. Abre: dev/[tu-notebook].ipynb
3. Setup: sys.path.insert(0, str(Path.cwd().parent / "src"))
4. Importa: from config import GAIA_PARALLAX5, get_output_path
5. Datos: pd.read_csv(str(GAIA_PARALLAX5))
6. Salida: save_result(df, "tu-archivo.csv", "datos_resultados")
```

### 🔬 Para HPC (SLURM)

```
1. Navega a: clusterization_project/
2. Ejecuta: sbatch src/run_hdbscan.sh
3. O: sbatch src/final_run.sh
4. Logs: cat logs/hdbscan_*.err
5. Scripts usan config.py automáticamente
```

### 📚 Para Documentación

```
1. Lee: docs/RUTAS_Y_CONFIGURACION.md
2. Lee: docs/EJEMPLOS_USO.md
3. Para cambios: lee AGENTS.md
4. Para validación: docs/VERIFICACION_ESTRUCTURA.md
```

---

## 🔄 Flujo de Trabajo Típico

### Flujo 1: Análisis Exploratorio
```
1. Crea notebook en dev/mi_analisis.ipynb
2. Setup rutas (primera celda)
3. Importa config.py
4. Carga datos de data/
5. Procesa y visualiza
6. Guarda resultados en data/datos_resultados/
```

### Flujo 2: Producción (Scripts)
```
1. Crea script en src/mi_script.py
2. Importa from config import ...
3. Lee datos de config
4. Procesa
5. Guarda en data/
6. Crea script SLURM en src/
7. Ejecuta en HPC: sbatch src/mi_script.sh
```

### Flujo 3: Análisis de Resultados
```
1. Notebooks lean de data/datos_resultados/
2. Usan config.DATOS_RESULTADOS
3. Visualizan y documentan hallazgos
4. Guardan análisis en docs/
```

---

## 📊 Estadísticas

- **Total archivos**: 20+ movidos
- **Notebooks**: 15 en dev/
- **Scripts Python**: 3 en src/
- **Scripts Shell**: 2 en src/
- **Carpetas de datos**: 6 en data/
- **Módulos config**: 2 (config.py + notebook_helper.py)
- **Documentación**: 5 archivos

---

## ✅ Checklist de Configuración

| Item | Status | Acción |
|------|--------|--------|
| Estructura de carpetas | ✅ | Completada |
| Movimiento de archivos | ✅ | Completado |
| Módulo config.py | ✅ | Creado |
| Módulo notebook_helper.py | ✅ | Creado |
| Scripts actualizados | ✅ | 5 scripts |
| Documentación | ✅ | 5 docs |
| Notebooks actualizados | ⏳ | Ver docs/RUTAS_Y_CONFIGURACION.md |
| Tests | ⏳ | Próximo paso |

---

## 🔑 Variables Claves en config.py

```python
# Rutas principales
PROJECT_ROOT        → c:\...\clusterization_project\
DATA_DIR           → ./data/
SRC_DIR            → ./src/
DEV_DIR            → ./dev/

# Datos específicos
GAIA_PARALLAX5     → data/DatosTotales/gaia_parallax5.csv
GAIA_PARALLAX5_10  → data/datos_shell/gaia_parallax5_10.csv
DATOS_RESULTADOS   → data/datos_resultados/
DATOS_RESULTADOS_MODULARIZADO → data/datos_resultados_modularizado/

# Funciones
get_data_path()         → Obtiene ruta en data/
get_output_path()       → Obtiene ruta para guardar
ensure_dir_exists()     → Crea directorio si no existe
```

---

## 🚀 Próximos Pasos Recomendados

1. **Actualizar notebooks** (opcional)
   - Agregar setup de rutas automático
   - Usar `config.py` en lugar de hardcodear

2. **Crear unit tests**
   - Tests para scripts en `test/`
   - Validar rutas en `test/test_config.py`

3. **Documentación adicional**
   - API docs para módulos en `src/`
   - Guías de contribución en `docs/`

4. **Optimización**
   - Performance profiling
   - Logging centralizado
   - Métricas de datos

---

## 📞 Quick Links

| Documento | Propósito |
|-----------|-----------|
| [README.md](README.md) | 📖 Visión general |
| [AGENTS.md](AGENTS.md) | 📋 Estándares |
| [docs/RUTAS_Y_CONFIGURACION.md](docs/RUTAS_Y_CONFIGURACION.md) | 🗺️ Setup de rutas |
| [docs/EJEMPLOS_USO.md](docs/EJEMPLOS_USO.md) | 📚 Ejemplos |
| [src/config.py](src/config.py) | ⚙️ Configuración central |

---

**Versión**: 1.0
**Fecha**: 29 de Enero de 2026
**Estado**: ✅ Completado y Documentado
