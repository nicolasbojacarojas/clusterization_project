#!/bin/bash
#SBATCH --job-name=hdbscan_job
#SBATCH --output=logs/hdbscan_%j.out
#SBATCH --error=logs/hdbscan_%j.err
#SBATCH -n 1  #tasks paralelos
#SBATCH -N 1  #nodos requeridos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=168:00:0
#SBATCH -p medium
#SBATCH --mail-user=nd.bojaca@uniandes.edu.co
#SBATCH --mail-type=ALL

# Carga miniconda (ajusta si usas otro)
module load anaconda

# Activa el entorno conda
source activate cluster_env

# Navega a la carpeta src del proyecto
# Asume que los archivos están en la carpeta src/
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[$(date)] PWD: $(pwd)" >&2
echo "[$(date)] Listing files in src:" >&2
ls -l . >&2
echo "[$(date)] Listing data folder:" >&2
ls -l ../data/ >&2

echo "[$(date)] START SLURM JOB" >&2

# Ejecuta el script Python desde la carpeta src
python -u hdbscan_final.py >&2

echo "[$(date)] END SLURM JOB" >&2