"""Utilities for stellar apex analysis on a single cluster.

Version ICRS / Gaia:
    - Positions: ra, dec
    - Proper motions: pmra, pmdec
    - Gaia convention: pmra = mu_alpha* = mu_alpha cos(dec)

This module implements the polar representation of the Convergent Point
Method / Herschel method directly in equatorial coordinates, without
transforming to Galactic coordinates.

Example:
    from vector_director_icrs import run_cluster_analysis

    result = run_cluster_analysis(
        data_path="data/datos_clusterizados.csv",
        cluster_id="8_123",
    )

    print(result["cluster_result"]["apex_refined"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation


MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)
KM_S_PER_AU_YR = 4.74047


# ============================================================
# Basic utilities
# ============================================================

def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = set(required_columns).difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )


def create_cluster_id(
    df: pd.DataFrame,
    group_col: str = "grupo",
    ra_col: str = "coordenada_ra",
    dec_col: str = "coordenada_dec",
    cluster_col: str = "cluster_id",
    division_col: str = "division",
    numeracion_col: str = "numeracion",
) -> pd.DataFrame:
    """
    Crea una columna cluster_id combinando el grupo de HDBSCAN/DBSCAN
    con la división espacial usada en el barrido.

    Por defecto usa coordenada_ra y coordenada_dec porque esas columnas
    suelen representar la región o celda del cielo, no la RA/Dec real
    de cada estrella.
    """

    validate_columns(df, [group_col, ra_col, dec_col])

    result = df.copy()

    result[division_col] = (
        result[ra_col].astype(str) + "_" + result[dec_col].astype(str)
    )

    unique_divisions = pd.Index(result[division_col].unique())
    division_to_index = dict(zip(unique_divisions, range(len(unique_divisions))))

    result[numeracion_col] = result[division_col].map(division_to_index)

    result[cluster_col] = (
        result[group_col].astype(str)
        + "_"
        + result[numeracion_col].astype(str)
    )

    return result


# ============================================================
# Equatorial / ICRS geometry
# ============================================================

def equatorial_radec_to_unit_vector(
    ra_deg: float,
    dec_deg: float,
) -> np.ndarray:
    """
    Convierte RA, Dec en grados a vector unitario cartesiano ICRS.
    """

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    return np.array(
        [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ],
        dtype=float,
    )

# def unit_vector_to_equatorial_radec(vector: np.ndarray) -> Dict[str, float]:
#     """
#     Convierte vector cartesiano ICRS a RA, Dec.
#     """

#     vector = np.asarray(vector, dtype=float)
#     norm = np.linalg.norm(vector)

#     if norm == 0.0 or not np.isfinite(norm):
#         raise ValueError("Input vector has zero or invalid norm.")

#     x, y, z = vector / norm

#     cartesian = CartesianRepresentation(
#         x=x * u.one,
#         y=y * u.one,
#         z=z * u.one,
#     )

#     coord = SkyCoord(
#         cartesian,
#         frame="icrs",
#     )

#     ra_deg = coord.ra.to_value(u.deg)
#     dec_deg = coord.dec.to_value(u.deg)

#     ra_rad = coord.ra.to_value(u.rad)
#     dec_rad = coord.dec.to_value(u.rad)

#     return {
#         "ra_deg": float(ra_deg),
#         "dec_deg": float(dec_deg),
#         "ra_rad": float(ra_rad),
#         "dec_rad": float(dec_rad),
#     }

def unit_vector_to_equatorial_radec(vector: np.ndarray) -> Dict[str, float]:
    """
    Convierte vector cartesiano ICRS a RA, Dec.
    """

    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Input vector has zero or invalid norm.")

    x, y, z = vector / norm

    ra_rad = np.arctan2(y, x) % (2.0 * np.pi)
    dec_rad = np.arcsin(np.clip(z, -1.0, 1.0))

    return {
        "ra_deg": float(np.degrees(ra_rad)),
        "dec_deg": float(np.degrees(dec_rad)),
        "ra_rad": float(ra_rad),
        "dec_rad": float(dec_rad),
    }

# ============================================================
# Apex error metrics against a known/injected apex
# ============================================================

def resolve_apex_vector(
    apex_vector: Optional[np.ndarray] = None,
    apex_ra_deg: Optional[float] = None,
    apex_dec_deg: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Construye/resuelve un vector unitario de ápex.

    Se puede pasar:
        - apex_vector=[x, y, z]
        - apex_ra_deg y apex_dec_deg

    Si no se pasa nada, devuelve None.
    """

    if apex_vector is not None:
        vector = np.asarray(apex_vector, dtype=float)
    elif apex_ra_deg is not None and apex_dec_deg is not None:
        vector = equatorial_radec_to_unit_vector(
            ra_deg=float(apex_ra_deg),
            dec_deg=float(apex_dec_deg),
        )
    else:
        return None

    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("The reference apex vector has zero or invalid norm.")

    return vector / norm


def angular_distance_between_vectors_deg(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
) -> float:
    """
    Distancia angular orientada entre dos vectores unitarios.

    Devuelve un ángulo en [0, 180] grados.
    Aquí ápex y antápex NO se consideran equivalentes.
    """

    v1 = np.asarray(vector_1, dtype=float)
    v2 = np.asarray(vector_2, dtype=float)

    norm_1 = np.linalg.norm(v1)
    norm_2 = np.linalg.norm(v2)

    if (
        not np.isfinite(norm_1)
        or not np.isfinite(norm_2)
        or norm_1 == 0.0
        or norm_2 == 0.0
    ):
        return np.nan

    v1 = v1 / norm_1
    v2 = v2 / norm_2

    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def angular_axis_error_between_vectors_deg(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
) -> float:
    """
    Error angular de eje entre dos vectores.

    Devuelve min(theta, 180 - theta), es decir, un ángulo en [0, 90] grados.

    Esta métrica es útil porque el ajuste del plano de polos tiene una
    degeneración natural ápex/antápex. Si quieres evaluar estrictamente si
    recuperaste el ápex y no el antápex, usa la métrica orientada.
    """

    theta = angular_distance_between_vectors_deg(vector_1, vector_2)

    if not np.isfinite(theta):
        return np.nan

    return float(min(theta, 180.0 - theta))

def apex_bootstrap_stability_metrics(
    df: pd.DataFrame,
    reference_apex_vector: np.ndarray,
    n_bootstrap: int = 200,
    sample_fraction: float = 0.8,
    random_state: int = 42,
    weight_col: Optional[str] = None,
    min_sources: int = 3,
    orient_with_motion: bool = True,
) -> Dict[str, Any]:
    """
    Mide la estabilidad angular del ápex mediante bootstrap.

    Esta métrica NO evalúa el residuo de los polos respecto al ápex.
    Evalúa qué tanto cambia el ápex estimado cuando se cambia la muestra
    de estrellas del cúmulo.

    Métrica principal:
        apex_bootstrap_rms_axis_deg

    Interpretación:
        valor pequeño  -> ápex estable
        valor grande   -> ápex sensible a la membresía / outliers / ruido
    """

    if df is None or len(df) < min_sources:
        return {
            "apex_bootstrap_rms_axis_deg": np.nan,
            "apex_bootstrap_median_axis_deg": np.nan,
            "apex_bootstrap_p16_axis_deg": np.nan,
            "apex_bootstrap_p84_axis_deg": np.nan,
            "apex_bootstrap_p95_axis_deg": np.nan,
            "apex_bootstrap_n_success": 0,
            "apex_bootstrap_n_failed": int(n_bootstrap),
        }

    reference_apex_vector = np.asarray(reference_apex_vector, dtype=float)
    reference_norm = np.linalg.norm(reference_apex_vector)

    if not np.isfinite(reference_norm) or reference_norm == 0.0:
        return {
            "apex_bootstrap_rms_axis_deg": np.nan,
            "apex_bootstrap_median_axis_deg": np.nan,
            "apex_bootstrap_p16_axis_deg": np.nan,
            "apex_bootstrap_p84_axis_deg": np.nan,
            "apex_bootstrap_p95_axis_deg": np.nan,
            "apex_bootstrap_n_success": 0,
            "apex_bootstrap_n_failed": int(n_bootstrap),
        }

    reference_apex_vector = reference_apex_vector / reference_norm

    rng = np.random.default_rng(random_state)

    n_rows = len(df)
    sample_size = max(min_sources, int(np.ceil(sample_fraction * n_rows)))

    axis_errors_deg = []
    n_failed = 0

    for _ in range(n_bootstrap):
        sample_index = rng.choice(
            np.arange(n_rows),
            size=sample_size,
            replace=True,
        )

        df_boot = df.iloc[sample_index].copy()

        try:
            boot_result = estimate_apex_and_antapex(
                df_boot,
                weight_col=weight_col,
                orient_with_motion=orient_with_motion,
                min_sources=min_sources,
            )

            boot_apex_vector = boot_result["apex_vector"]

            axis_error_deg = angular_axis_error_between_vectors_deg(
                boot_apex_vector,
                reference_apex_vector,
            )

            if np.isfinite(axis_error_deg):
                axis_errors_deg.append(axis_error_deg)
            else:
                n_failed += 1

        except Exception:
            n_failed += 1

    axis_errors_deg = np.asarray(axis_errors_deg, dtype=float)

    if len(axis_errors_deg) == 0:
        return {
            "apex_bootstrap_rms_axis_deg": np.nan,
            "apex_bootstrap_median_axis_deg": np.nan,
            "apex_bootstrap_p16_axis_deg": np.nan,
            "apex_bootstrap_p84_axis_deg": np.nan,
            "apex_bootstrap_p95_axis_deg": np.nan,
            "apex_bootstrap_n_success": 0,
            "apex_bootstrap_n_failed": int(n_failed),
        }

    return {
        "apex_bootstrap_rms_axis_deg": float(
            np.sqrt(np.mean(axis_errors_deg**2))
        ),
        "apex_bootstrap_median_axis_deg": float(
            np.median(axis_errors_deg)
        ),
        "apex_bootstrap_p16_axis_deg": float(
            np.percentile(axis_errors_deg, 16)
        ),
        "apex_bootstrap_p84_axis_deg": float(
            np.percentile(axis_errors_deg, 84)
        ),
        "apex_bootstrap_p95_axis_deg": float(
            np.percentile(axis_errors_deg, 95)
        ),
        "apex_bootstrap_n_success": int(len(axis_errors_deg)),
        "apex_bootstrap_n_failed": int(n_failed),
    }

def add_true_apex_error_to_apex_result(
    apex_result: Optional[Dict[str, Any]],
    true_apex_vector: Optional[np.ndarray],
    prefix: str,
) -> Dict[str, float]:
    """
    Calcula errores entre un resultado de ápex y el ápex verdadero.

    Devuelve dos métricas:
        - {prefix}_oriented_error_true_deg:
            error angular directo, en [0, 180].
        - {prefix}_axis_error_true_deg:
            error de eje, en [0, 90], con degeneración ápex/antápex.
    """

    if apex_result is None or true_apex_vector is None:
        return {
            f"{prefix}_oriented_error_true_deg": np.nan,
            f"{prefix}_axis_error_true_deg": np.nan,
        }

    estimated_vector = apex_result.get("apex_vector")

    if estimated_vector is None:
        return {
            f"{prefix}_oriented_error_true_deg": np.nan,
            f"{prefix}_axis_error_true_deg": np.nan,
        }

    return {
        f"{prefix}_oriented_error_true_deg": angular_distance_between_vectors_deg(
            estimated_vector,
            true_apex_vector,
        ),
        f"{prefix}_axis_error_true_deg": angular_axis_error_between_vectors_deg(
            estimated_vector,
            true_apex_vector,
        ),
    }

def add_initial_final_equatorial_vectors(
    df: pd.DataFrame,
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    time_years: float = 1.0,
    angles_in_degrees: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Construye los vectores inicial y final sobre la esfera celeste usando
    directamente coordenadas Gaia.

    Gaia reporta:
        pmra = mu_alpha* = mu_alpha cos(dec)

    Por tanto, para desplazar la coordenada RA se usa:
        delta_alpha = pmra / cos(dec)

    pero para el módulo del movimiento propio se usa directamente:
        mu_total = sqrt(pmra^2 + pmdec^2)
    """

    validate_columns(df, [ra_col, dec_col, pmra_col, pmdec_col])

    result = df.copy() if copy else df

    if angles_in_degrees:
        ra_rad = np.deg2rad(result[ra_col].to_numpy(dtype=float))
        dec_rad = np.deg2rad(result[dec_col].to_numpy(dtype=float))
    else:
        ra_rad = result[ra_col].to_numpy(dtype=float)
        dec_rad = result[dec_col].to_numpy(dtype=float)

    pmra_values = result[pmra_col].to_numpy(dtype=float)
    pmdec_values = result[pmdec_col].to_numpy(dtype=float)

    cos_dec = np.cos(dec_rad)

    near_pole = np.abs(cos_dec) < 1e-12
    if near_pole.any():
        raise ValueError(
            "Some sources have cos(dec) too close to zero. "
            "Cannot safely compute delta_ra = pmra / cos(dec)."
        )

    delta_ra_rad = (pmra_values / cos_dec) * MAS_TO_RAD * time_years
    delta_dec_rad = pmdec_values * MAS_TO_RAD * time_years

    ra_final_rad = (ra_rad + delta_ra_rad) % (2.0 * np.pi)
    dec_final_rad = dec_rad + delta_dec_rad

    result["x_initial"] = np.cos(dec_rad) * np.cos(ra_rad)
    result["y_initial"] = np.cos(dec_rad) * np.sin(ra_rad)
    result["z_initial"] = np.sin(dec_rad)

    result["x_final"] = np.cos(dec_final_rad) * np.cos(ra_final_rad)
    result["y_final"] = np.cos(dec_final_rad) * np.sin(ra_final_rad)
    result["z_final"] = np.sin(dec_final_rad)

    result["delta_ra_rad"] = delta_ra_rad
    result["delta_dec_rad"] = delta_dec_rad
    result["ra_final_rad"] = ra_final_rad
    result["dec_final_rad"] = dec_final_rad

    return result


def add_cross_product_poles(
    df: pd.DataFrame,
    x_initial_col: str = "x_initial",
    y_initial_col: str = "y_initial",
    z_initial_col: str = "z_initial",
    x_final_col: str = "x_final",
    y_final_col: str = "y_final",
    z_final_col: str = "z_final",
    normalize: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Calcula el polo de cada círculo máximo:

        p = x_initial x x_final
    """

    required_columns = [
        x_initial_col,
        y_initial_col,
        z_initial_col,
        x_final_col,
        y_final_col,
        z_final_col,
    ]

    validate_columns(df, required_columns)

    result = df.copy() if copy else df

    initial_vectors = result[
        [x_initial_col, y_initial_col, z_initial_col]
    ].to_numpy(dtype=float)

    final_vectors = result[
        [x_final_col, y_final_col, z_final_col]
    ].to_numpy(dtype=float)

    pole_vectors = np.cross(initial_vectors, final_vectors)
    pole_norm = np.linalg.norm(pole_vectors, axis=1)

    result["pole_x"] = pole_vectors[:, 0]
    result["pole_y"] = pole_vectors[:, 1]
    result["pole_z"] = pole_vectors[:, 2]
    result["pole_norm"] = pole_norm

    if normalize:
        normalized = np.full_like(pole_vectors, np.nan)
        valid_norm = pole_norm > 0.0

        normalized[valid_norm] = (
            pole_vectors[valid_norm] / pole_norm[valid_norm, None]
        )

        result["pole_x_unit"] = normalized[:, 0]
        result["pole_y_unit"] = normalized[:, 1]
        result["pole_z_unit"] = normalized[:, 2]

    return result

def add_pole_apex_error_comparison(
    df: pd.DataFrame,
    apex_vector: np.ndarray,
    pole_x_col: str = "pole_x_unit",
    pole_y_col: str = "pole_y_unit",
    pole_z_col: str = "pole_z_unit",
    sin_lambda_col: str = "sin_lambda",
    prefix: str = "refined",
    min_sin_lambda: float = 1e-6,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Agrega dos errores para comparar:

    1. Error geométrico:
        arcsin(polo · apex)

    2. Error direccional corregido:
        arcsin((polo · apex) / sin(lambda))

    Ambos se guardan con signo y en valor absoluto.
    """

    validate_columns(df, [pole_x_col, pole_y_col, pole_z_col, sin_lambda_col])

    result = df.copy() if copy else df

    apex_vector = np.asarray(apex_vector, dtype=float)
    apex_norm = np.linalg.norm(apex_vector)

    if not np.isfinite(apex_norm) or apex_norm == 0.0:
        raise ValueError("Invalid apex_vector: zero or non-finite norm.")

    apex_unit = apex_vector / apex_norm

    poles = result[[pole_x_col, pole_y_col, pole_z_col]].to_numpy(dtype=float)
    pole_norms = np.linalg.norm(poles, axis=1)

    valid_poles = (
        np.isfinite(poles).all(axis=1)
        & np.isfinite(pole_norms)
        & (pole_norms > 0.0)
    )

    poles_unit = np.full_like(poles, np.nan, dtype=float)
    poles_unit[valid_poles] = poles[valid_poles] / pole_norms[valid_poles, None]

    dot = np.full(len(result), np.nan, dtype=float)
    dot[valid_poles] = poles_unit[valid_poles] @ apex_unit
    dot = np.clip(dot, -1.0, 1.0)

    # 1. Error geométrico: arcsin(p · apex)
    geometric_signed_deg = np.degrees(np.arcsin(dot))
    geometric_abs_deg = np.degrees(np.arcsin(np.abs(dot)))

    # 2. Error direccional corregido por sin(lambda)
    sin_lambda = result[sin_lambda_col].to_numpy(dtype=float)

    valid_direction = (
        np.isfinite(dot)
        & np.isfinite(sin_lambda)
        & (sin_lambda > min_sin_lambda)
    )

    ratio = np.full(len(result), np.nan, dtype=float)
    ratio[valid_direction] = dot[valid_direction] / sin_lambda[valid_direction]
    ratio = np.clip(ratio, -1.0, 1.0)

    direction_signed_deg = np.degrees(np.arcsin(ratio))
    direction_abs_deg = np.degrees(np.arcsin(np.abs(ratio)))

    valid_inverse = (
        np.isfinite(dot)
        & np.isfinite(sin_lambda)
    )

    ratio_inverse = np.full(len(result), np.nan, dtype=float)
    ratio_inverse[valid_inverse] = dot[valid_inverse] * sin_lambda[valid_inverse]
    ratio_inverse = np.clip(ratio_inverse, -1.0, 1.0)

    direction_signed_inverse_deg = np.degrees(np.arcsin(ratio_inverse))
    direction_abs_inverse_deg = np.degrees(np.arcsin(np.abs(ratio_inverse)))

    result[f"pole_apex_dot_{prefix}"] = dot
    result[f"pole_apex_error_geom_signed_{prefix}_deg"] = geometric_signed_deg
    result[f"pole_apex_error_geom_abs_{prefix}_deg"] = geometric_abs_deg

    result[f"pole_apex_error_direction_ratio_{prefix}"] = ratio
    result[f"pole_apex_error_direction_signed_{prefix}_deg"] = direction_signed_deg
    result[f"pole_apex_error_direction_abs_{prefix}_deg"] = direction_abs_deg

    result[f"pole_apex_error_direction_inverse_ratio_{prefix}"] = ratio_inverse
    result[f"pole_apex_error_direction_inverse_signed_{prefix}_deg"] = direction_signed_inverse_deg
    result[f"pole_apex_error_direction_inverse_abs_{prefix}_deg"] = direction_abs_inverse_deg

    return result

def add_perfect_poles_and_theta_from_apex(
    df: pd.DataFrame,
    apex_vector: np.ndarray,
    x_initial_col: str = "x_initial",
    y_initial_col: str = "y_initial",
    z_initial_col: str = "z_initial",
    pole_x_col: str = "pole_x_unit",
    pole_y_col: str = "pole_y_unit",
    pole_z_col: str = "pole_z_unit",
    prefix: str = "apex",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Agrega al DataFrame el polo perfecto de cada estrella y el ángulo theta
    entre ese polo perfecto y el polo observado de la estrella.

    Definición:

        perfect_pole_i = r_i x apex

    donde:
        r_i   = posición inicial unitaria de la estrella
        apex  = vector unitario del ápex del cúmulo

    Luego calcula:

        theta = angle(perfect_pole_i, observed_pole_i)

    Columnas agregadas:
        perfect_pole_{prefix}_x
        perfect_pole_{prefix}_y
        perfect_pole_{prefix}_z
        perfect_pole_{prefix}_norm

        perfect_pole_{prefix}_x_unit
        perfect_pole_{prefix}_y_unit
        perfect_pole_{prefix}_z_unit

        theta_pole_{prefix}_deg
        theta_pole_axis_{prefix}_deg

    Nota:
        theta_pole_{prefix}_deg es el ángulo orientado en [0, 180].
        theta_pole_axis_{prefix}_deg trata p y -p como equivalentes,
        por tanto queda en [0, 90].
    """

    required_columns = [
        x_initial_col,
        y_initial_col,
        z_initial_col,
        pole_x_col,
        pole_y_col,
        pole_z_col,
    ]

    validate_columns(df, required_columns)

    result = df.copy() if copy else df

    apex_vector = np.asarray(apex_vector, dtype=float)
    apex_norm = np.linalg.norm(apex_vector)

    if apex_norm == 0.0 or not np.isfinite(apex_norm):
        raise ValueError("Invalid apex_vector: zero or non-finite norm.")

    apex_unit = apex_vector / apex_norm

    initial_vectors = result[
        [x_initial_col, y_initial_col, z_initial_col]
    ].to_numpy(dtype=float)

    initial_norms = np.linalg.norm(initial_vectors, axis=1)

    initial_valid = (
        np.isfinite(initial_vectors).all(axis=1)
        & np.isfinite(initial_norms)
        & (initial_norms > 0.0)
    )

    initial_unit = np.full_like(initial_vectors, np.nan, dtype=float)
    initial_unit[initial_valid] = (
        initial_vectors[initial_valid] / initial_norms[initial_valid, None]
    )

    # Polo perfecto: posición inicial x ápex del cúmulo
    perfect_poles = np.cross(initial_unit, apex_unit)
    perfect_norms = np.linalg.norm(perfect_poles, axis=1)

    perfect_valid = (
        np.isfinite(perfect_poles).all(axis=1)
        & np.isfinite(perfect_norms)
        & (perfect_norms > 0.0)
    )

    perfect_unit = np.full_like(perfect_poles, np.nan, dtype=float)
    perfect_unit[perfect_valid] = (
        perfect_poles[perfect_valid] / perfect_norms[perfect_valid, None]
    )

    result[f"perfect_pole_{prefix}_x"] = perfect_poles[:, 0]
    result[f"perfect_pole_{prefix}_y"] = perfect_poles[:, 1]
    result[f"perfect_pole_{prefix}_z"] = perfect_poles[:, 2]
    result[f"perfect_pole_{prefix}_norm"] = perfect_norms

    result[f"perfect_pole_{prefix}_x_unit"] = perfect_unit[:, 0]
    result[f"perfect_pole_{prefix}_y_unit"] = perfect_unit[:, 1]
    result[f"perfect_pole_{prefix}_z_unit"] = perfect_unit[:, 2]

    observed_poles = result[
        [pole_x_col, pole_y_col, pole_z_col]
    ].to_numpy(dtype=float)

    observed_norms = np.linalg.norm(observed_poles, axis=1)

    observed_valid = (
        np.isfinite(observed_poles).all(axis=1)
        & np.isfinite(observed_norms)
        & (observed_norms > 0.0)
    )

    observed_unit = np.full_like(observed_poles, np.nan, dtype=float)
    observed_unit[observed_valid] = (
        observed_poles[observed_valid] / observed_norms[observed_valid, None]
    )

    valid_theta = perfect_valid & observed_valid

    dot = np.full(len(result), np.nan, dtype=float)
    dot[valid_theta] = np.sum(
        perfect_unit[valid_theta] * observed_unit[valid_theta],
        axis=1,
    )

    dot = np.clip(dot, -1.0, 1.0)

    theta_deg = np.degrees(np.arccos(dot))

    # Versión axial: útil si el signo del polo puede invertirse.
    theta_axis_deg = np.degrees(np.arccos(np.abs(dot)))

    result[f"theta_pole_{prefix}_deg"] = theta_deg
    result[f"theta_pole_axis_{prefix}_deg"] = theta_axis_deg

    return result

def sigma_clip_by_theta_pole(
    df: pd.DataFrame,
    theta_col: str = "theta_pole_axis_initial_deg",
    nsigma: float = 3.0,
    min_remaining: int = 5,
    prefix: str = "theta_clip",
    copy: bool = True,
):
    """
    Aplica sigma clipping sobre la distribución de theta entre el polo observado
    y el polo perfecto.

    Usa una estimación robusta:

        sigma = 1.4826 * MAD

    y conserva las estrellas con:

        theta <= median(theta) + nsigma * sigma

    Retorna:
        df_marked : dataframe con columnas de diagnóstico
        df_inliers : dataframe solo con estrellas aceptadas
        clip_info : diccionario con resumen del clipping
    """

    validate_columns(df, [theta_col])

    result = df.copy() if copy else df

    theta = result[theta_col].to_numpy(dtype=float)
    finite = np.isfinite(theta)

    if finite.sum() == 0:
        result[f"{prefix}_is_inlier"] = False
        result[f"{prefix}_is_outlier"] = True

        clip_info = {
            "theta_clip_column": theta_col,
            "theta_clip_nsigma": nsigma,
            "theta_clip_center_deg": np.nan,
            "theta_clip_sigma_deg": np.nan,
            "theta_clip_threshold_deg": np.nan,
            "theta_clip_n_total": int(len(result)),
            "theta_clip_n_finite": 0,
            "theta_clip_n_inliers": 0,
            "theta_clip_n_outliers": int(len(result)),
            "theta_clip_used": False,
        }

        return result, result.iloc[0:0].copy(), clip_info

    theta_finite = theta[finite]

    center = float(np.median(theta_finite))
    mad = float(np.median(np.abs(theta_finite - center)))
    sigma = 1.4826 * mad

    # Fallback si el MAD da cero.
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = float(np.std(theta_finite))

    # Si incluso std da cero, no hay dispersión medible: se conservan todas.
    if not np.isfinite(sigma) or sigma == 0.0:
        threshold = center
        inlier = finite
    else:
        threshold = center + nsigma * sigma
        inlier = finite & (theta <= threshold)

    outlier = ~inlier

    # Seguridad: si el clipping deja muy pocas estrellas, no se aplica.
    clipping_used = True

    if inlier.sum() < min_remaining:
        inlier = finite
        outlier = ~inlier
        clipping_used = False

    result[f"{prefix}_is_inlier"] = inlier
    result[f"{prefix}_is_outlier"] = outlier
    result[f"{prefix}_theta_center_deg"] = center
    result[f"{prefix}_theta_sigma_deg"] = sigma
    result[f"{prefix}_theta_threshold_deg"] = threshold
    result[f"{prefix}_theta_nsigma"] = nsigma

    df_inliers = result.loc[inlier].copy()

    clip_info = {
        "theta_clip_column": theta_col,
        "theta_clip_nsigma": float(nsigma),
        "theta_clip_center_deg": float(center),
        "theta_clip_sigma_deg": float(sigma),
        "theta_clip_threshold_deg": float(threshold),
        "theta_clip_n_total": int(len(result)),
        "theta_clip_n_finite": int(finite.sum()),
        "theta_clip_n_inliers": int(inlier.sum()),
        "theta_clip_n_outliers": int(outlier.sum()),
        "theta_clip_used": bool(clipping_used),
    }

    return result, df_inliers, clip_info

def robust_center_sigma_mad(values: np.ndarray) -> Dict[str, float]:
    """
    Calcula centro y sigma robustos usando mediana y MAD.

    sigma = 1.4826 * MAD

    Si MAD = 0, usa desviación estándar como fallback.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "center": np.nan,
            "sigma": np.nan,
            "mad": np.nan,
            "n": 0,
        }

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    sigma = float(1.4826 * mad)

    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = float(np.std(values))

    return {
        "center": center,
        "sigma": sigma,
        "mad": mad,
        "n": int(len(values)),
    }


def iterative_theta_sigma_clip_refinement(
    df: pd.DataFrame,
    initial_apex_result: Dict[str, Any],
    weight_col: Optional[str] = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
    nsigma: float = 3.0,
    max_iter: int = 5,
    min_sources: int = 3,
    min_remaining: Optional[int] = None,
    min_sin_lambda: float = 0.15,
    min_pole_norm: float = 0.0,
    apex_tolerance_deg: float = 1e-3,
    threshold_tolerance_deg: float = 1e-4,
    orient_with_motion: bool = True,
    prefix: str = "theta_iter",
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Refinamiento iterativo del ápex usando sigma clipping sobre theta.

    En cada iteración:
        1. estima el ápex con los miembros actuales,
        2. calcula el polo perfecto de cada estrella,
        3. calcula theta_pole_axis,
        4. conserva estrellas con:

            theta <= median(theta) + nsigma * sigma_theta

    donde sigma_theta se estima robustamente con MAD.

    También conserva los cortes geométricos:
        sin(lambda) >= min_sin_lambda
        pole_norm > min_pole_norm

    Retorna:
        df_marked:
            DataFrame original con columnas de diagnóstico del clipping final.

        df_refined:
            DataFrame solo con los miembros finales aceptados.

        apex_refined:
            resultado final de estimate_apex_and_antapex.

        clip_info:
            resumen global del clipping iterativo.

        clip_history:
            lista con el historial de cada iteración.
    """

    if min_remaining is None:
        min_remaining = min_sources

    min_remaining = max(int(min_remaining), int(min_sources))

    if df is None or len(df) < min_sources:
        clip_info = {
            "theta_clip_used": False,
            "theta_clip_stop_reason": "not_enough_input_sources",
            "theta_clip_n_iterations": 0,
            "theta_clip_nsigma": float(nsigma),
            "theta_clip_max_iter": int(max_iter),
            "theta_clip_n_initial": 0 if df is None else int(len(df)),
            "theta_clip_n_final": 0 if df is None else int(len(df)),
            "theta_clip_n_rejected": 0,
        }

        return df.copy(), df.copy(), initial_apex_result, clip_info, []

    current_index = pd.Index(df.index)
    history: List[Dict[str, Any]] = []

    previous_apex_vector = None
    previous_threshold = np.nan

    stop_reason = "max_iter_reached"
    last_apex_result = initial_apex_result

    for iteration in range(int(max_iter)):
        df_current_base = df.loc[current_index].copy()

        if len(df_current_base) < min_sources:
            stop_reason = "not_enough_sources_before_iteration"
            break

        if iteration == 0:
            apex_result = initial_apex_result
        else:
            try:
                apex_result = estimate_apex_and_antapex(
                    df_current_base,
                    weight_col=weight_col,
                    orient_with_motion=orient_with_motion,
                    min_sources=min_sources,
                )
            except ValueError:
                stop_reason = "apex_estimation_failed"
                break

        last_apex_result = apex_result

        apex_vector = apex_result["apex_vector"]

        df_current = add_lambda_angle_from_apex(
            df_current_base,
            apex_vector=apex_vector,
            ra_col=ra_col,
            dec_col=dec_col,
        )

        iter_prefix = f"{prefix}_{iteration}"

        df_current = add_perfect_poles_and_theta_from_apex(
            df_current,
            apex_vector=apex_vector,
            prefix=iter_prefix,
        )

        theta_col = f"theta_pole_axis_{iter_prefix}_deg"

        theta = df_current[theta_col].to_numpy(dtype=float)

        valid_for_stats = np.isfinite(theta)

        if min_sin_lambda is not None:
            valid_for_stats &= (
                df_current["sin_lambda"].notna().to_numpy()
                & np.isfinite(df_current["sin_lambda"].to_numpy(dtype=float))
                & (df_current["sin_lambda"].to_numpy(dtype=float) >= min_sin_lambda)
            )

        if min_pole_norm is not None:
            valid_for_stats &= (
                df_current["pole_norm"].notna().to_numpy()
                & np.isfinite(df_current["pole_norm"].to_numpy(dtype=float))
                & (df_current["pole_norm"].to_numpy(dtype=float) > min_pole_norm)
            )

        theta_for_stats = theta[valid_for_stats]

        robust_stats = robust_center_sigma_mad(theta_for_stats)

        center = robust_stats["center"]
        sigma = robust_stats["sigma"]
        mad = robust_stats["mad"]
        n_stats = robust_stats["n"]

        if n_stats == 0:
            stop_reason = "no_valid_theta_values"
            break

        if not np.isfinite(sigma) or sigma == 0.0:
            threshold = center
            inlier_mask_local = valid_for_stats
        else:
            threshold = center + nsigma * sigma
            inlier_mask_local = valid_for_stats & (theta <= threshold)

        new_index = pd.Index(df_current.index[inlier_mask_local])

        n_input_iter = int(len(df_current))
        n_inliers_iter = int(len(new_index))
        n_outliers_iter = int(n_input_iter - n_inliers_iter)

        apex_change_deg = np.nan

        if previous_apex_vector is not None:
            apex_change_deg = angular_axis_error_between_vectors_deg(
                apex_vector,
                previous_apex_vector,
            )

        threshold_change_deg = np.nan

        if np.isfinite(previous_threshold) and np.isfinite(threshold):
            threshold_change_deg = float(abs(threshold - previous_threshold))

        members_stable = new_index.equals(current_index)

        history_row = {
            "iteration": int(iteration),
            "n_input": n_input_iter,
            "n_valid_for_stats": int(n_stats),
            "n_inliers": n_inliers_iter,
            "n_outliers": n_outliers_iter,
            "theta_center_deg": float(center) if np.isfinite(center) else np.nan,
            "theta_mad_deg": float(mad) if np.isfinite(mad) else np.nan,
            "theta_sigma_deg": float(sigma) if np.isfinite(sigma) else np.nan,
            "theta_threshold_deg": float(threshold)
            if np.isfinite(threshold)
            else np.nan,
            "apex_change_axis_deg": float(apex_change_deg)
            if np.isfinite(apex_change_deg)
            else np.nan,
            "threshold_change_deg": float(threshold_change_deg)
            if np.isfinite(threshold_change_deg)
            else np.nan,
            "members_stable": bool(members_stable),
        }

        history.append(history_row)

        if n_inliers_iter < min_remaining:
            stop_reason = "too_few_sources_after_clipping"
            break

        if members_stable:
            current_index = new_index
            stop_reason = "members_stable"
            break

        current_index = new_index

        if (
            iteration > 0
            and np.isfinite(apex_change_deg)
            and np.isfinite(threshold_change_deg)
            and apex_change_deg < apex_tolerance_deg
            and threshold_change_deg < threshold_tolerance_deg
        ):
            stop_reason = "apex_and_threshold_stable"
            break

        previous_apex_vector = apex_vector.copy()
        previous_threshold = threshold

    df_final_base = df.loc[current_index].copy()

    if len(df_final_base) >= min_sources:
        try:
            apex_refined = estimate_apex_and_antapex(
                df_final_base,
                weight_col=weight_col,
                orient_with_motion=orient_with_motion,
                min_sources=min_sources,
            )
        except ValueError:
            apex_refined = last_apex_result
            stop_reason = "final_apex_estimation_failed_using_last"
    else:
        apex_refined = last_apex_result
        stop_reason = "final_sample_too_small_using_last"

    final_inlier_index = pd.Index(df_final_base.index)

    df_marked = df.copy()

    df_marked[f"{prefix}_final_is_inlier"] = df_marked.index.isin(
        final_inlier_index
    )
    df_marked[f"{prefix}_final_is_outlier"] = ~df_marked[
        f"{prefix}_final_is_inlier"
    ]

    df_marked = add_perfect_poles_and_theta_from_apex(
        df_marked,
        apex_vector=apex_refined["apex_vector"],
        prefix=f"{prefix}_final",
    )

    df_refined = df_marked.loc[final_inlier_index].copy()

    df_refined = add_lambda_angle_from_apex(
        df_refined,
        apex_vector=apex_refined["apex_vector"],
        ra_col=ra_col,
        dec_col=dec_col,
    )

    df_refined = add_perfect_poles_and_theta_from_apex(
        df_refined,
        apex_vector=apex_refined["apex_vector"],
        prefix="refined",
    )

    df_refined["lambda_deg_refined"] = df_refined["lambda_deg"]
    df_refined["sin_lambda_refined"] = df_refined["sin_lambda"]
    df_refined["cos_lambda_refined"] = df_refined["cos_lambda"]

    df_refined = add_pole_apex_error_comparison(
        df_refined,
        apex_vector=apex_refined["apex_vector"],
        sin_lambda_col="sin_lambda_refined",
        prefix="refined",
    )

    last_history = history[-1] if len(history) > 0 else {}

    clip_info = {
        "theta_clip_used": bool(len(history) > 0),
        "theta_clip_stop_reason": stop_reason,
        "theta_clip_n_iterations": int(len(history)),
        "theta_clip_nsigma": float(nsigma),
        "theta_clip_max_iter": int(max_iter),
        "theta_clip_min_remaining": int(min_remaining),
        "theta_clip_min_sin_lambda": float(min_sin_lambda),
        "theta_clip_min_pole_norm": float(min_pole_norm),
        "theta_clip_apex_tolerance_deg": float(apex_tolerance_deg),
        "theta_clip_threshold_tolerance_deg": float(threshold_tolerance_deg),
        "theta_clip_n_initial": int(len(df)),
        "theta_clip_n_final": int(len(df_refined)),
        "theta_clip_n_rejected": int(len(df) - len(df_refined)),
        "theta_clip_final_center_deg": last_history.get(
            "theta_center_deg",
            np.nan,
        ),
        "theta_clip_final_sigma_deg": last_history.get(
            "theta_sigma_deg",
            np.nan,
        ),
        "theta_clip_final_threshold_deg": last_history.get(
            "theta_threshold_deg",
            np.nan,
        ),
        "theta_clip_final_apex_change_axis_deg": last_history.get(
            "apex_change_axis_deg",
            np.nan,
        ),
        "theta_clip_final_threshold_change_deg": last_history.get(
            "threshold_change_deg",
            np.nan,
        ),
    }

    return df_marked, df_refined, apex_refined, clip_info, history


# ============================================================
# Apex estimation
# ============================================================

def estimate_apex_and_antapex(
    df: pd.DataFrame,
    pole_x_col: str = "pole_x_unit",
    pole_y_col: str = "pole_y_unit",
    pole_z_col: str = "pole_z_unit",
    x_initial_col: str = "x_initial",
    y_initial_col: str = "y_initial",
    z_initial_col: str = "z_initial",
    x_final_col: str = "x_final",
    y_final_col: str = "y_final",
    z_final_col: str = "z_final",
    weight_col: Optional[str] = None,
    orient_with_motion: bool = True,
    min_sources: int = 3,
) -> Dict[str, Any]:
    """
    Estima ápex y antápex ajustando el plano de polos.

    Si los polos p_i están sobre un gran círculo, entonces el ápex
    es el vector normal a ese plano. El vector normal se estima como
    el autovector asociado al menor autovalor de la matriz de covarianza
    de los polos.
    """

    required = [
        pole_x_col,
        pole_y_col,
        pole_z_col,
        x_initial_col,
        y_initial_col,
        z_initial_col,
        x_final_col,
        y_final_col,
        z_final_col,
    ]

    if weight_col is not None:
        required.append(weight_col)

    validate_columns(df, required)

    poles_all = df[[pole_x_col, pole_y_col, pole_z_col]].to_numpy(dtype=float)

    valid_mask = np.isfinite(poles_all).all(axis=1)
    pole_norms = np.linalg.norm(poles_all, axis=1)
    valid_mask &= pole_norms > 0.0

    if weight_col is not None:
        weights_all = df[weight_col].to_numpy(dtype=float)
        valid_mask &= np.isfinite(weights_all)
        valid_mask &= weights_all > 0.0
    else:
        weights_all = np.ones(len(df), dtype=float)

    if valid_mask.sum() < min_sources:
        raise ValueError(
            "Not enough valid poles to estimate apex. "
            f"Found {valid_mask.sum()}, required {min_sources}."
        )

    poles = poles_all[valid_mask] / pole_norms[valid_mask, None]
    weights = weights_all[valid_mask]
    weights_norm = weights / np.sum(weights)

    covariance_matrix = (poles * weights_norm[:, None]).T @ poles

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    apex_vector = eigenvectors[:, 0]
    apex_vector = apex_vector / np.linalg.norm(apex_vector)

    oriented = False
    toward_apex_fraction = np.nan

    if orient_with_motion:
        initial_vectors = df.loc[
            valid_mask, [x_initial_col, y_initial_col, z_initial_col]
        ].to_numpy(dtype=float)

        final_vectors = df.loc[
            valid_mask, [x_final_col, y_final_col, z_final_col]
        ].to_numpy(dtype=float)

        initial_norms = np.linalg.norm(initial_vectors, axis=1)
        final_norms = np.linalg.norm(final_vectors, axis=1)

        vector_valid = (
            np.isfinite(initial_vectors).all(axis=1)
            & np.isfinite(final_vectors).all(axis=1)
            & (initial_norms > 0.0)
            & (final_norms > 0.0)
        )

        if vector_valid.sum() >= min_sources:
            orientation_weights = weights[vector_valid]

            initial_vectors = (
                initial_vectors[vector_valid]
                / initial_norms[vector_valid, None]
            )

            final_vectors = (
                final_vectors[vector_valid]
                / final_norms[vector_valid, None]
            )

            initial_projection = initial_vectors @ apex_vector
            final_projection = final_vectors @ apex_vector

            moving_toward_apex = final_projection > initial_projection

            toward_apex_fraction = np.average(
                moving_toward_apex.astype(float),
                weights=orientation_weights,
            )

            if toward_apex_fraction < 0.5:
                apex_vector = -apex_vector
                toward_apex_fraction = 1.0 - toward_apex_fraction

            oriented = True

    antapex_vector = -apex_vector

    apex_coordinates = unit_vector_to_equatorial_radec(apex_vector)
    antapex_coordinates = unit_vector_to_equatorial_radec(antapex_vector)

    pole_plane_residual = np.clip(poles @ apex_vector, -1.0, 1.0)
    pole_residual_deg = np.degrees(np.arcsin(pole_plane_residual))

    return {
        "apex_vector": apex_vector,
        "antapex_vector": antapex_vector,

        "apex_ra_deg": apex_coordinates["ra_deg"],
        "apex_dec_deg": apex_coordinates["dec_deg"],
        "apex_ra_rad": apex_coordinates["ra_rad"],
        "apex_dec_rad": apex_coordinates["dec_rad"],

        "antapex_ra_deg": antapex_coordinates["ra_deg"],
        "antapex_dec_deg": antapex_coordinates["dec_deg"],
        "antapex_ra_rad": antapex_coordinates["ra_rad"],
        "antapex_dec_rad": antapex_coordinates["dec_rad"],

        "n_sources": int(valid_mask.sum()),
        "eigenvalues": eigenvalues,

        "rms_pole_residual_deg": float(
            np.sqrt(np.average(pole_residual_deg**2, weights=weights))
        ),
        "median_pole_residual_deg": float(np.median(pole_residual_deg)),

        "oriented_with_motion": bool(oriented),
        "toward_apex_fraction": float(toward_apex_fraction)
        if np.isfinite(toward_apex_fraction)
        else np.nan,
    }


def apex_from_pole_cross_products(
    df: pd.DataFrame,
    reference_apex_vector: Optional[np.ndarray] = None,
    pole_x_col: str = "pole_x_unit",
    pole_y_col: str = "pole_y_unit",
    pole_z_col: str = "pole_z_unit",
    weight_col: Optional[str] = None,
    pole_norm_min: float = 0.0,
    min_cross_norm: float = 1e-12,
    min_sources: int = 3,
    max_pairs: int = 200_000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """

    Estima el ápex usando intersecciones entre pares de polos.

    Cada par de polos p_i, p_j define una intersección:

        z_ij = p_i x p_j

    La dirección z_ij tiene degeneración ápex/antápex. Se orienta usando
    reference_apex_vector.
    """

    required = [pole_x_col, pole_y_col, pole_z_col]

    if weight_col is not None:
        required.append(weight_col)

    validate_columns(df, required)

    poles_all = df[[pole_x_col, pole_y_col, pole_z_col]].to_numpy(dtype=float)

    valid = np.isfinite(poles_all).all(axis=1)
    pole_norms = np.linalg.norm(poles_all, axis=1)
    valid &= pole_norms > pole_norm_min

    if weight_col is not None:
        weights_all = df[weight_col].to_numpy(dtype=float)
        valid &= np.isfinite(weights_all)
        valid &= weights_all > 0.0
    else:
        weights_all = np.ones(len(df), dtype=float)

    if valid.sum() < min_sources:
        raise ValueError(
            "Not enough valid pole vectors to compute cross-product apex. "
            f"Found {valid.sum()}, required {min_sources}."
        )

    poles = poles_all[valid] / pole_norms[valid, None]
    weights = weights_all[valid]

    if reference_apex_vector is None:
        weights_norm = weights / np.sum(weights)
        covariance_matrix = (poles * weights_norm[:, None]).T @ poles
        _, eigenvectors = np.linalg.eigh(covariance_matrix)
        reference_apex_vector = eigenvectors[:, 0]

    reference_apex_vector = np.asarray(reference_apex_vector, dtype=float)
    reference_norm = np.linalg.norm(reference_apex_vector)

    if reference_norm == 0.0 or not np.isfinite(reference_apex_vector).all():
        raise ValueError("Invalid reference_apex_vector.")

    reference_apex_vector = reference_apex_vector / reference_norm

    n_poles = len(poles)

    pair_i, pair_j = np.triu_indices(n_poles, k=1)

    n_pairs_total = len(pair_i)

    if n_pairs_total == 0:
        raise ValueError("No pole pairs available.")

    if n_pairs_total > max_pairs:
        rng = np.random.default_rng(random_state)
        selected = rng.choice(n_pairs_total, size=max_pairs, replace=False)
        pair_i = pair_i[selected]
        pair_j = pair_j[selected]

    cross_vectors = np.cross(poles[pair_i], poles[pair_j])
    cross_norms = np.linalg.norm(cross_vectors, axis=1)

    valid_cross = np.isfinite(cross_vectors).all(axis=1)
    valid_cross &= cross_norms > min_cross_norm

    if valid_cross.sum() < min_sources:
        raise ValueError(
            "Not enough valid pole intersections. "
            f"Found {valid_cross.sum()}, required {min_sources}."
        )

    cross_vectors = cross_vectors[valid_cross] / cross_norms[valid_cross, None]

    pair_weights = weights[pair_i[valid_cross]] * weights[pair_j[valid_cross]]

    orientation = np.sign(cross_vectors @ reference_apex_vector)
    orientation[orientation == 0.0] = 1.0

    cross_vectors = cross_vectors * orientation[:, None]

    weighted_mean = np.average(cross_vectors, axis=0, weights=pair_weights)
    mean_norm = np.linalg.norm(weighted_mean)

    if mean_norm == 0.0 or not np.isfinite(mean_norm):
        raise ValueError("Cross-product apex has zero or invalid norm.")

    apex_vector = weighted_mean / mean_norm
    antapex_vector = -apex_vector

    apex_coordinates = unit_vector_to_equatorial_radec(apex_vector)
    antapex_coordinates = unit_vector_to_equatorial_radec(antapex_vector)

    cos_angles = np.clip(cross_vectors @ apex_vector, -1.0, 1.0)
    angular_residuals_deg = np.degrees(np.arccos(cos_angles))

    return {
        "apex_vector": apex_vector,
        "antapex_vector": antapex_vector,

        "apex_ra_deg": apex_coordinates["ra_deg"],
        "apex_dec_deg": apex_coordinates["dec_deg"],
        "apex_ra_rad": apex_coordinates["ra_rad"],
        "apex_dec_rad": apex_coordinates["dec_rad"],

        "antapex_ra_deg": antapex_coordinates["ra_deg"],
        "antapex_dec_deg": antapex_coordinates["dec_deg"],
        "antapex_ra_rad": antapex_coordinates["ra_rad"],
        "antapex_dec_rad": antapex_coordinates["dec_rad"],

        "n_sources": int(n_poles),
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_used": int(valid_cross.sum()),

        "rms_intersection_residual_deg": float(
            np.sqrt(np.average(angular_residuals_deg**2, weights=pair_weights))
        ),
        "median_intersection_residual_deg": float(
            np.median(angular_residuals_deg)
        ),
    }


# ============================================================
# Lambda angle relative to apex
# ============================================================

def add_lambda_angle_from_apex(
    df: pd.DataFrame,
    apex_vector: Optional[np.ndarray] = None,
    apex_ra_deg: Optional[float] = None,
    apex_dec_deg: Optional[float] = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Agrega el ángulo lambda entre cada estrella y el ápex en coordenadas ICRS.

    cos(lambda) =
        sin(dec) sin(dec_apex)
        + cos(dec) cos(dec_apex) cos(ra_apex - ra)
    """

    validate_columns(df, [ra_col, dec_col])

    result = df.copy() if copy else df

    if apex_vector is not None:
        apex_coordinates = unit_vector_to_equatorial_radec(apex_vector)
        apex_ra_deg = apex_coordinates["ra_deg"]
        apex_dec_deg = apex_coordinates["dec_deg"]

    if apex_ra_deg is None or apex_dec_deg is None:
        raise ValueError(
            "You must provide either apex_vector or both "
            "apex_ra_deg and apex_dec_deg."
        )

    ra_rad = np.deg2rad(result[ra_col].to_numpy(dtype=float))
    dec_rad = np.deg2rad(result[dec_col].to_numpy(dtype=float))

    apex_ra_rad = np.deg2rad(float(apex_ra_deg))
    apex_dec_rad = np.deg2rad(float(apex_dec_deg))

    delta_ra_rad = apex_ra_rad - ra_rad
    delta_ra_rad = (delta_ra_rad + np.pi) % (2.0 * np.pi) - np.pi

    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)

    sin_dec_apex = np.sin(apex_dec_rad)
    cos_dec_apex = np.cos(apex_dec_rad)

    cos_delta_ra = np.cos(delta_ra_rad)
    sin_delta_ra = np.sin(delta_ra_rad)

    cos_lambda = (
        sin_dec * sin_dec_apex
        + cos_dec * cos_dec_apex * cos_delta_ra
    )
    cos_lambda = np.clip(cos_lambda, -1.0, 1.0)

    tangent_ra_numerator = cos_dec_apex * sin_delta_ra

    tangent_dec_numerator = (
        cos_dec * sin_dec_apex
        - sin_dec * cos_dec_apex * cos_delta_ra
    )

    sin_lambda = np.sqrt(
        tangent_ra_numerator**2 + tangent_dec_numerator**2
    )
    sin_lambda = np.clip(sin_lambda, 0.0, 1.0)

    lambda_rad = np.arctan2(sin_lambda, cos_lambda)

    result["lambda_rad"] = lambda_rad
    result["lambda_deg"] = np.rad2deg(lambda_rad)
    result["sin_lambda"] = sin_lambda
    result["cos_lambda"] = cos_lambda

    result["delta_ra_apex_rad"] = delta_ra_rad
    result["delta_ra_apex_deg"] = np.rad2deg(delta_ra_rad)

    valid_sin_lambda = sin_lambda > 1e-15

    result["lambda_tangent_ra"] = np.where(
        valid_sin_lambda,
        tangent_ra_numerator / sin_lambda,
        np.nan,
    )

    result["lambda_tangent_dec"] = np.where(
        valid_sin_lambda,
        tangent_dec_numerator / sin_lambda,
        np.nan,
    )

    return result


# ============================================================
# Data preparation
# ============================================================

def prepare_dataframe(
    df: pd.DataFrame,
    group_col: str = "grupo",
    division_ra_col: str = "coordenada_ra",
    division_dec_col: str = "coordenada_dec",
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    parallax_col: str = "parallax",
    count_col: str = "ConteoAgrupaciones",
    present_col: str = "PresenteEnMuestras",
    cluster_col: str = "cluster_id",
    weight_col: str = "probabilidad",
    weight_fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Prepara el DataFrame para el análisis del ápex en RA/Dec.

    Crea:
        - cluster_id, si no existe.
        - distance_pc, si no existe.
        - probabilidad, si no existe.
        - vectores inicial/final ICRS.
        - polos por producto cruz.
    """

    result = df.copy()

    if cluster_col not in result.columns:
        result = create_cluster_id(
            result,
            group_col=group_col,
            ra_col=division_ra_col,
            dec_col=division_dec_col,
            cluster_col=cluster_col,
        )

    if "distance_pc" not in result.columns:
        validate_columns(result, [parallax_col])

        parallax = result[parallax_col].astype(float)

        result["distance_pc"] = np.where(
            parallax > 0.0,
            1000.0 / parallax,
            np.nan,
        )

    if weight_col not in result.columns:
        if weight_fill_value is not None:
            result[weight_col] = float(weight_fill_value)
        else:
            validate_columns(result, [count_col, present_col])

            present = result[present_col].astype(float)
            count = result[count_col].astype(float)

            result[weight_col] = np.where(
                present > 0.0,
                count / present,
                np.nan,
            )

    result = add_initial_final_equatorial_vectors(
        result,
        ra_col=ra_col,
        dec_col=dec_col,
        pmra_col=pmra_col,
        pmdec_col=pmdec_col,
    )

    result = add_cross_product_poles(result)

    return result


# ============================================================
# Cluster processing
# ============================================================

def process_single_cluster(
    df: pd.DataFrame,
    cluster_id: Union[str, int],
    cluster_col: str = "cluster_id",
    weight_col: Optional[str] = "probabilidad",
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    min_stars: int = 3,
    refine_apex: bool = True,
    min_sin_lambda: float = 0.15,
    min_pole_norm: float = 0.0,
    theta_clip_nsigma: float = 3.0,
    theta_clip_max_iter: int = 5,
    theta_clip_min_remaining: Optional[int] = None,
    theta_clip_apex_tolerance_deg: float = 1e-3,
    theta_clip_threshold_tolerance_deg: float = 1e-4,
    compute_cross_product_poles: bool = True,
    true_apex_vector: Optional[np.ndarray] = None,
    true_apex_ra_deg: Optional[float] = None,
    true_apex_dec_deg: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Procesa un único cúmulo y estima su ápex inicial y refinado.

    El refinamiento elimina estrellas cercanas al ápex o antápex usando:

        sin(lambda) >= min_sin_lambda

    Esto evita fuentes con movimiento tangencial muy pequeño, donde el polo
    del círculo máximo queda peor condicionado.
    """
    true_apex_vector_resolved = resolve_apex_vector(
        apex_vector=true_apex_vector,
        apex_ra_deg=true_apex_ra_deg,
        apex_dec_deg=true_apex_dec_deg,
    )

    validate_columns(df, [cluster_col, ra_col, dec_col, pmra_col, pmdec_col])

    df_cluster = df[df[cluster_col] == cluster_id].copy()

    if len(df_cluster) == 0:
        print(f"Cluster {cluster_id}: no data found.")
        return None

    if len(df_cluster) < min_stars:
        print(
            f"Cluster {cluster_id}: only {len(df_cluster)} stars. "
            f"Minimum required: {min_stars}."
        )
        return None

    effective_weight_col = weight_col

    if effective_weight_col is not None:
        validate_columns(df_cluster, [effective_weight_col])

        valid_weights = (
            df_cluster[effective_weight_col].notna()
            & np.isfinite(df_cluster[effective_weight_col])
            & (df_cluster[effective_weight_col] > 0)
        )

        if valid_weights.sum() < min_stars:
            print(
                f"Cluster {cluster_id}: not enough valid weights. "
                "Apex will be estimated without weights."
            )
            effective_weight_col = None

    try:
        apex_initial = estimate_apex_and_antapex(
            df_cluster,
            weight_col=effective_weight_col,
            min_sources=min_stars,
        )
    except ValueError as exc:
        print(f"Cluster {cluster_id}: error computing initial apex - {exc}")
        return None

    df_initial = add_lambda_angle_from_apex(
        df_cluster,
        apex_vector=apex_initial["apex_vector"],
        ra_col=ra_col,
        dec_col=dec_col,
    )

    df_initial = add_perfect_poles_and_theta_from_apex(
        df_initial,
        apex_vector=apex_initial["apex_vector"],
        prefix="initial",
    )

    df_initial["lambda_deg_initial"] = df_initial["lambda_deg"]
    df_initial["sin_lambda_initial"] = df_initial["sin_lambda"]
    df_initial["cos_lambda_initial"] = df_initial["cos_lambda"]

    df_initial = add_pole_apex_error_comparison(
        df_initial,
        apex_vector=apex_initial["apex_vector"],
        sin_lambda_col="sin_lambda_initial",
        prefix="initial",
    )

    df_initial["pm_total"] = np.sqrt(
        df_initial[pmra_col].astype(float) ** 2
        + df_initial[pmdec_col].astype(float) ** 2
    )

    if "distance_pc" in df_initial.columns:
        df_initial["vt_kms"] = (
            KM_S_PER_AU_YR
            * (df_initial["pm_total"] / 1000.0)
            * df_initial["distance_pc"]
        )

    cross_initial = None
    if compute_cross_product_poles:
        try:
            cross_initial = apex_from_pole_cross_products(
                df_initial,
                reference_apex_vector=apex_initial["apex_vector"],
                weight_col=effective_weight_col,
                pole_norm_min=min_pole_norm,
                min_sources=min_stars,
            )
        except ValueError as exc:
            print(
                f"Cluster {cluster_id}: cross-product initial apex failed - {exc}"
            )
            cross_initial = None

    apex_initial_bootstrap_metrics = apex_bootstrap_stability_metrics(
        df=df_initial,
        reference_apex_vector=apex_initial["apex_vector"],
        n_bootstrap=2,
        sample_fraction=0.8,
        random_state=42,
        weight_col=effective_weight_col,
        min_sources=min_stars,
        orient_with_motion=True,
    )

    df_refined = df_initial.copy()
    apex_refined = apex_initial
    cross_refined = cross_initial

    theta_clip_info = {
        "theta_clip_used": False,
        "theta_clip_stop_reason": "refine_apex_false",
        "theta_clip_n_iterations": 0,
        "theta_clip_nsigma": float(theta_clip_nsigma),
        "theta_clip_max_iter": int(theta_clip_max_iter),
        "theta_clip_n_initial": int(len(df_initial)),
        "theta_clip_n_final": int(len(df_initial)),
        "theta_clip_n_rejected": 0,
    }

    theta_clip_history: List[Dict[str, Any]] = []

    if refine_apex:
        (
            df_initial,
            df_refined,
            apex_refined,
            theta_clip_info,
            theta_clip_history,
        ) = iterative_theta_sigma_clip_refinement(
            df=df_initial,
            initial_apex_result=apex_initial,
            weight_col=effective_weight_col,
            ra_col=ra_col,
            dec_col=dec_col,
            nsigma=theta_clip_nsigma,
            max_iter=theta_clip_max_iter,
            min_sources=min_stars,
            min_remaining=theta_clip_min_remaining
            if theta_clip_min_remaining is not None
            else min_stars,
            min_sin_lambda=min_sin_lambda,
            min_pole_norm=min_pole_norm,
            apex_tolerance_deg=theta_clip_apex_tolerance_deg,
            threshold_tolerance_deg=theta_clip_threshold_tolerance_deg,
            orient_with_motion=True,
            prefix="theta_iter",
        )

    else:
        df_refined = add_lambda_angle_from_apex(
            df_refined,
            apex_vector=apex_refined["apex_vector"],
            ra_col=ra_col,
            dec_col=dec_col,
        )

        df_refined = add_perfect_poles_and_theta_from_apex(
            df_refined,
            apex_vector=apex_refined["apex_vector"],
            prefix="refined",
        )

        df_refined["lambda_deg_refined"] = df_refined["lambda_deg"]
        df_refined["sin_lambda_refined"] = df_refined["sin_lambda"]
        df_refined["cos_lambda_refined"] = df_refined["cos_lambda"]

    df_refined["pm_total"] = np.sqrt(
        df_refined[pmra_col].astype(float) ** 2
        + df_refined[pmdec_col].astype(float) ** 2
    )

    if "distance_pc" in df_refined.columns:
        df_refined["vt_kms"] = (
            KM_S_PER_AU_YR
            * (df_refined["pm_total"] / 1000.0)
            * df_refined["distance_pc"]
        )

    cross_refined = None
    if compute_cross_product_poles:
        try:
            cross_refined = apex_from_pole_cross_products(
                df_refined,
                reference_apex_vector=apex_refined["apex_vector"],
                weight_col=effective_weight_col,
                pole_norm_min=min_pole_norm,
                min_sources=min_stars,
            )
        except ValueError as exc:
            print(
                f"Cluster {cluster_id}: cross-product refined apex failed - {exc}"
            )
            cross_refined = None


    if "theta_pole_refined_deg" not in df_refined.columns:
        df_refined = add_perfect_poles_and_theta_from_apex(
            df_refined,
            apex_vector=apex_refined["apex_vector"],
            prefix="refined",
        )

    if "lambda_deg_refined" not in df_refined.columns:
        df_refined = add_lambda_angle_from_apex(
            df_refined,
            apex_vector=apex_refined["apex_vector"],
            ra_col=ra_col,
            dec_col=dec_col,
        )

        df_refined["lambda_deg_refined"] = df_refined["lambda_deg"]
        df_refined["sin_lambda_refined"] = df_refined["sin_lambda"]
        df_refined["cos_lambda_refined"] = df_refined["cos_lambda"]

    # ============================================================
    # Pole-apex residuals: geometric and direction-corrected
    # ============================================================

    if "sin_lambda_initial" not in df_initial.columns:
        df_initial = add_lambda_angle_from_apex(
            df_initial,
            apex_vector=apex_initial["apex_vector"],
            ra_col=ra_col,
            dec_col=dec_col,
        )

        df_initial["lambda_deg_initial"] = df_initial["lambda_deg"]
        df_initial["sin_lambda_initial"] = df_initial["sin_lambda"]
        df_initial["cos_lambda_initial"] = df_initial["cos_lambda"]

    df_initial = add_pole_apex_error_comparison(
        df_initial,
        apex_vector=apex_initial["apex_vector"],
        sin_lambda_col="sin_lambda_initial",
        prefix="initial",
    )

    if "sin_lambda_refined" not in df_refined.columns:
        df_refined = add_lambda_angle_from_apex(
            df_refined,
            apex_vector=apex_refined["apex_vector"],
            ra_col=ra_col,
            dec_col=dec_col,
        )

        df_refined["lambda_deg_refined"] = df_refined["lambda_deg"]
        df_refined["sin_lambda_refined"] = df_refined["sin_lambda"]
        df_refined["cos_lambda_refined"] = df_refined["cos_lambda"]

    df_refined = add_pole_apex_error_comparison(
        df_refined,
        apex_vector=apex_refined["apex_vector"],
        sin_lambda_col="sin_lambda_refined",
        prefix="refined",
    )

    true_apex_coordinates = (
        unit_vector_to_equatorial_radec(true_apex_vector_resolved)
        if true_apex_vector_resolved is not None
        else None
    )

    apex_refined_bootstrap_metrics = apex_bootstrap_stability_metrics(
        df=df_refined,
        reference_apex_vector=apex_refined["apex_vector"],
        n_bootstrap=200,
        sample_fraction=0.8,
        random_state=43,
        weight_col=effective_weight_col,
        min_sources=min_stars,
        orient_with_motion=True,
    )

    apex_initial_true_errors = add_true_apex_error_to_apex_result(
        apex_initial,
        true_apex_vector_resolved,
        prefix="apex_initial",
    )

    apex_refined_true_errors = add_true_apex_error_to_apex_result(
        apex_refined,
        true_apex_vector_resolved,
        prefix="apex_refined",
    )

    cross_initial_true_errors = add_true_apex_error_to_apex_result(
        cross_initial,
        true_apex_vector_resolved,
        prefix="cross_initial",
    )

    cross_refined_true_errors = add_true_apex_error_to_apex_result(
        cross_refined,
        true_apex_vector_resolved,
        prefix="cross_refined",
    )

    return {
        "cluster_id": cluster_id,
        "n_stars": int(len(df_cluster)),
        "n_stars_initial": int(len(df_initial)),
        "n_stars_refined": int(len(df_refined)),
        "weight_col_used": effective_weight_col,

        "apex_initial": apex_initial,
        "apex_refined": apex_refined,

        "cross_initial": cross_initial,
        "cross_refined": cross_refined,

        "apex_initial_bootstrap_metrics": apex_initial_bootstrap_metrics,
        "apex_refined_bootstrap_metrics": apex_refined_bootstrap_metrics,

        "true_apex_vector": true_apex_vector_resolved,
        "true_apex_ra_deg": (
            true_apex_coordinates["ra_deg"]
            if true_apex_coordinates is not None
            else np.nan
        ),
        "true_apex_dec_deg": (
            true_apex_coordinates["dec_deg"]
            if true_apex_coordinates is not None
            else np.nan
        ),

        **apex_initial_true_errors,
        **apex_refined_true_errors,
        **cross_initial_true_errors,
        **cross_refined_true_errors,

        "theta_clip_info": theta_clip_info,
        "theta_clip_history": theta_clip_history,

        "data_initial": df_initial,
        "data_refined": df_refined,

        "data_initial": df_initial,
        "data_refined": df_refined,
    }


def process_all_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    weight_col: Optional[str] = "probabilidad",
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    min_stars: int = 3,
    refine_apex: bool = True,
    min_sin_lambda: float = 0.15,
    min_pole_norm: float = 0.0,
    true_apex_vector: Optional[np.ndarray] = None,
    true_apex_ra_deg: Optional[float] = None,
    true_apex_dec_deg: Optional[float] = None,
    theta_clip_nsigma: float = 3.0,
    theta_clip_max_iter: int = 5,
    theta_clip_min_remaining: Optional[int] = None,
    theta_clip_apex_tolerance_deg: float = 1e-3,
    theta_clip_threshold_tolerance_deg: float = 1e-4,
    compute_cross_product_poles: bool = True,
) -> List[Dict[str, Any]]:
    """
    Procesa todos los cluster_id presentes en el DataFrame.
    """

    validate_columns(df, [cluster_col])

    results: List[Dict[str, Any]] = []

    cluster_ids = pd.Index(df[cluster_col].dropna().unique())

    for cid in cluster_ids:
        cluster_result = process_single_cluster(
            df=df,
            cluster_id=cid,
            cluster_col=cluster_col,
            weight_col=weight_col,
            ra_col=ra_col,
            dec_col=dec_col,
            pmra_col=pmra_col,
            pmdec_col=pmdec_col,
            min_stars=min_stars,
            refine_apex=refine_apex,
            min_sin_lambda=min_sin_lambda,
            min_pole_norm=min_pole_norm,
            true_apex_vector=true_apex_vector,
            true_apex_ra_deg=true_apex_ra_deg,
            true_apex_dec_deg=true_apex_dec_deg,
            theta_clip_nsigma=theta_clip_nsigma,
            theta_clip_max_iter=theta_clip_max_iter,
            theta_clip_min_remaining=theta_clip_min_remaining,
            theta_clip_apex_tolerance_deg=theta_clip_apex_tolerance_deg,
            theta_clip_threshold_tolerance_deg=theta_clip_threshold_tolerance_deg,
            compute_cross_product_poles=compute_cross_product_poles,
        )

        if cluster_result is not None:
            results.append(cluster_result)

    return results


# ============================================================
# Result summarization
# ============================================================

def _safe_get(result: Optional[Dict[str, Any]], key: str) -> Any:
    if result is None:
        return np.nan

    return result.get(key, np.nan)


def summarize_cluster_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte el resultado completo de un cúmulo en una fila plana.
    Útil para guardar en CSV.
    """

    apex_initial = result["apex_initial"]
    apex_refined = result["apex_refined"]

    cross_initial = result.get("cross_initial")
    cross_refined = result.get("cross_refined")

    apex_initial_bootstrap = result.get("apex_initial_bootstrap_metrics", {})
    apex_refined_bootstrap = result.get("apex_refined_bootstrap_metrics", {})

    theta_clip_info = result.get("theta_clip_info", {})

    data_initial = result.get("data_initial")
    data_refined = result.get("data_refined")

    def _nanmedian_col(df_data, col):
        if df_data is None or col not in df_data.columns:
            return np.nan

        values = df_data[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan

        return float(np.median(values))


    def _nanrms_col(df_data, col):
        if df_data is None or col not in df_data.columns:
            return np.nan

        values = df_data[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan

        return float(np.sqrt(np.mean(values**2)))

    def _nanmean_col(df_data, col):
        if df_data is None or col not in df_data.columns:
            return np.nan

        values = df_data[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan

        return float(np.mean(values))


    def _nanpercentile_col(df_data, col, q):
        if df_data is None or col not in df_data.columns:
            return np.nan

        values = df_data[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan

        return float(np.percentile(values, q))

    return {
        "cluster_id": result["cluster_id"],
        "n_stars": result["n_stars"],
        "n_stars_initial": result["n_stars_initial"],
        "n_stars_refined": result["n_stars_refined"],
        "true_apex_ra_deg": result.get("true_apex_ra_deg", np.nan),
        "true_apex_dec_deg": result.get("true_apex_dec_deg", np.nan),

        "apex_initial_ra_deg": apex_initial["apex_ra_deg"],
        "apex_initial_dec_deg": apex_initial["apex_dec_deg"],
        "antapex_initial_ra_deg": apex_initial["antapex_ra_deg"],
        "antapex_initial_dec_deg": apex_initial["antapex_dec_deg"],
        "rms_pole_residual_initial_deg": apex_initial[
            "rms_pole_residual_deg"
        ],
        "median_pole_residual_initial_deg": apex_initial[
            "median_pole_residual_deg"
        ],
        "toward_apex_fraction_initial": apex_initial[
            "toward_apex_fraction"
        ],
        "apex_initial_bootstrap_rms_axis_deg": apex_initial_bootstrap.get(
            "apex_bootstrap_rms_axis_deg",
            np.nan,
        ),
        "apex_initial_bootstrap_median_axis_deg": apex_initial_bootstrap.get(
            "apex_bootstrap_median_axis_deg",
            np.nan,
        ),
        "apex_initial_bootstrap_p16_axis_deg": apex_initial_bootstrap.get(
            "apex_bootstrap_p16_axis_deg",
            np.nan,
        ),
        "apex_initial_bootstrap_p84_axis_deg": apex_initial_bootstrap.get(
            "apex_bootstrap_p84_axis_deg",
            np.nan,
        ),
        "apex_initial_bootstrap_p95_axis_deg": apex_initial_bootstrap.get(
            "apex_bootstrap_p95_axis_deg",
            np.nan,
        ),
        "apex_initial_bootstrap_n_success": apex_initial_bootstrap.get(
            "apex_bootstrap_n_success",
            0,
        ),
        "apex_initial_oriented_error_true_deg": result.get(
            "apex_initial_oriented_error_true_deg",
            np.nan,
        ),
        "apex_initial_axis_error_true_deg": result.get(
            "apex_initial_axis_error_true_deg",
            np.nan,
        ),

        "apex_refined_ra_deg": apex_refined["apex_ra_deg"],
        "apex_refined_dec_deg": apex_refined["apex_dec_deg"],
        "antapex_refined_ra_deg": apex_refined["antapex_ra_deg"],
        "antapex_refined_dec_deg": apex_refined["antapex_dec_deg"],
        "rms_pole_residual_refined_deg": apex_refined[
            "rms_pole_residual_deg"
        ],
        "median_pole_residual_refined_deg": apex_refined[
            "median_pole_residual_deg"
        ],
        "toward_apex_fraction_refined": apex_refined[
            "toward_apex_fraction"
        ],
        "apex_refined_bootstrap_rms_axis_deg": apex_refined_bootstrap.get(
            "apex_bootstrap_rms_axis_deg",
            np.nan,
        ),
        "apex_refined_bootstrap_median_axis_deg": apex_refined_bootstrap.get(
            "apex_bootstrap_median_axis_deg",
            np.nan,
        ),
        "apex_refined_bootstrap_p16_axis_deg": apex_refined_bootstrap.get(
            "apex_bootstrap_p16_axis_deg",
            np.nan,
        ),
        "apex_refined_bootstrap_p84_axis_deg": apex_refined_bootstrap.get(
            "apex_bootstrap_p84_axis_deg",
            np.nan,
        ),
        "apex_refined_bootstrap_p95_axis_deg": apex_refined_bootstrap.get(
            "apex_bootstrap_p95_axis_deg",
            np.nan,
        ),
        "apex_refined_bootstrap_n_success": apex_refined_bootstrap.get(
            "apex_bootstrap_n_success",
            0,
        ),
        "apex_refined_oriented_error_true_deg": result.get(
            "apex_refined_oriented_error_true_deg",
            np.nan,
        ),
        "apex_refined_axis_error_true_deg": result.get(
            "apex_refined_axis_error_true_deg",
            np.nan,
        ),

        "cross_initial_apex_ra_deg": _safe_get(cross_initial, "apex_ra_deg"),
        "cross_initial_apex_dec_deg": _safe_get(cross_initial, "apex_dec_deg"),
        "cross_initial_rms_residual_deg": _safe_get(
            cross_initial,
            "rms_intersection_residual_deg",
        ),
        "cross_initial_n_pairs_used": _safe_get(cross_initial, "n_pairs_used"),

        "cross_refined_apex_ra_deg": _safe_get(cross_refined, "apex_ra_deg"),
        "cross_refined_apex_dec_deg": _safe_get(cross_refined, "apex_dec_deg"),
        "cross_refined_rms_residual_deg": _safe_get(
            cross_refined,
            "rms_intersection_residual_deg",
        ),
        "cross_refined_n_pairs_used": _safe_get(cross_refined, "n_pairs_used"),
        "cross_initial_oriented_error_true_deg": result.get(
            "cross_initial_oriented_error_true_deg",
            np.nan,
        ),
        "cross_initial_axis_error_true_deg": result.get(
            "cross_initial_axis_error_true_deg",
            np.nan,
        ),
        "cross_refined_oriented_error_true_deg": result.get(
            "cross_refined_oriented_error_true_deg",
            np.nan,
        ),
        "cross_refined_axis_error_true_deg": result.get(
            "cross_refined_axis_error_true_deg",
            np.nan,
        ),
        "theta_pole_initial_median_deg": _nanmedian_col(
            data_initial,
            "theta_pole_initial_deg",
        ),
        "theta_pole_initial_rms_deg": _nanrms_col(
            data_initial,
            "theta_pole_initial_deg",
        ),
        "theta_pole_axis_initial_median_deg": _nanmedian_col(
            data_initial,
            "theta_pole_axis_initial_deg",
        ),
        "theta_pole_axis_initial_rms_deg": _nanrms_col(
            data_initial,
            "theta_pole_axis_initial_deg",
        ),

        "theta_pole_refined_median_deg": _nanmedian_col(
            data_refined,
            "theta_pole_refined_deg",
        ),
        "theta_pole_refined_rms_deg": _nanrms_col(
            data_refined,
            "theta_pole_refined_deg",
        ),
        "theta_pole_axis_refined_median_deg": _nanmedian_col(
            data_refined,
            "theta_pole_axis_refined_deg",
        ),
        "theta_pole_axis_refined_rms_deg": _nanrms_col(
            data_refined,
            "theta_pole_axis_refined_deg",
        ),
        "theta_clip_used": theta_clip_info.get("theta_clip_used", False),
        "theta_clip_stop_reason": theta_clip_info.get(
            "theta_clip_stop_reason",
            None,
        ),
        "theta_clip_n_iterations": theta_clip_info.get(
            "theta_clip_n_iterations",
            0,
        ),
        "theta_clip_nsigma": theta_clip_info.get(
            "theta_clip_nsigma",
            np.nan,
        ),
        "theta_clip_n_initial": theta_clip_info.get(
            "theta_clip_n_initial",
            np.nan,
        ),
        "theta_clip_n_final": theta_clip_info.get(
            "theta_clip_n_final",
            np.nan,
        ),
        "theta_clip_n_rejected": theta_clip_info.get(
            "theta_clip_n_rejected",
            np.nan,
        ),
        "theta_clip_final_center_deg": theta_clip_info.get(
            "theta_clip_final_center_deg",
            np.nan,
        ),
        "theta_clip_final_sigma_deg": theta_clip_info.get(
            "theta_clip_final_sigma_deg",
            np.nan,
        ),
        "theta_clip_final_threshold_deg": theta_clip_info.get(
            "theta_clip_final_threshold_deg",
            np.nan,
        ),
        "theta_clip_final_apex_change_axis_deg": theta_clip_info.get(
            "theta_clip_final_apex_change_axis_deg",
            np.nan,
        ),
        "theta_clip_final_threshold_change_deg": theta_clip_info.get(
            "theta_clip_final_threshold_change_deg",
            np.nan,
        ),
        "lambda_refined_median_deg": _nanmedian_col(
            data_refined,
            "lambda_deg_refined",
        ),
        "lambda_refined_p16_deg": _nanpercentile_col(
            data_refined,
            "lambda_deg_refined",
            16,
        ),
        "lambda_refined_p84_deg": _nanpercentile_col(
            data_refined,
            "lambda_deg_refined",
            84,
        ),
        "sin_lambda_refined_median": _nanmedian_col(
            data_refined,
            "sin_lambda_refined",
        ),
        "sin_lambda_refined_mean": _nanmean_col(
            data_refined,
            "sin_lambda_refined",
        ),
        "pole_norm_refined_median": _nanmedian_col(
            data_refined,
            "pole_norm",
        ),
        "pm_total_refined_median": _nanmedian_col(
            data_refined,
            "pm_total",
        ),
        "vt_kms_refined_median": _nanmedian_col(
            data_refined,
            "vt_kms",
        ),
        # ============================================================
        # Pole-apex geometric residuals: initial
        # ============================================================

        "pole_apex_dot_initial_median": _nanmedian_col(
            data_initial,
            "pole_apex_dot_initial",
        ),
        "pole_apex_dot_initial_rms": _nanrms_col(
            data_initial,
            "pole_apex_dot_initial",
        ),

        "pole_apex_error_geom_signed_initial_mean_deg": _nanmean_col(
            data_initial,
            "pole_apex_error_geom_signed_initial_deg",
        ),
        "pole_apex_error_geom_signed_initial_median_deg": _nanmedian_col(
            data_initial,
            "pole_apex_error_geom_signed_initial_deg",
        ),
        "pole_apex_error_geom_signed_initial_rms_deg": _nanrms_col(
            data_initial,
            "pole_apex_error_geom_signed_initial_deg",
        ),

        "pole_apex_error_geom_abs_initial_median_deg": _nanmedian_col(
            data_initial,
            "pole_apex_error_geom_abs_initial_deg",
        ),
        "pole_apex_error_geom_abs_initial_rms_deg": _nanrms_col(
            data_initial,
            "pole_apex_error_geom_abs_initial_deg",
        ),
        "pole_apex_error_geom_abs_initial_p16_deg": _nanpercentile_col(
            data_initial,
            "pole_apex_error_geom_abs_initial_deg",
            16,
        ),
        "pole_apex_error_geom_abs_initial_p84_deg": _nanpercentile_col(
            data_initial,
            "pole_apex_error_geom_abs_initial_deg",
            84,
        ),

        # ============================================================
        # Pole-apex direction-corrected residuals: initial
        # ============================================================

        "pole_apex_error_direction_ratio_initial_median": _nanmedian_col(
            data_initial,
            "pole_apex_error_direction_ratio_initial",
        ),
        "pole_apex_error_direction_ratio_initial_rms": _nanrms_col(
            data_initial,
            "pole_apex_error_direction_ratio_initial",
        ),

        "pole_apex_error_direction_signed_initial_mean_deg": _nanmean_col(
            data_initial,
            "pole_apex_error_direction_signed_initial_deg",
        ),
        "pole_apex_error_direction_signed_initial_median_deg": _nanmedian_col(
            data_initial,
            "pole_apex_error_direction_signed_initial_deg",
        ),
        "pole_apex_error_direction_signed_initial_rms_deg": _nanrms_col(
            data_initial,
            "pole_apex_error_direction_signed_initial_deg",
        ),

        "pole_apex_error_direction_abs_initial_median_deg": _nanmedian_col(
            data_initial,
            "pole_apex_error_direction_abs_initial_deg",
        ),
        "pole_apex_error_direction_abs_initial_rms_deg": _nanrms_col(
            data_initial,
            "pole_apex_error_direction_abs_initial_deg",
        ),
        "pole_apex_error_direction_abs_initial_p16_deg": _nanpercentile_col(
            data_initial,
            "pole_apex_error_direction_abs_initial_deg",
            16,
        ),
        "pole_apex_error_direction_abs_initial_p84_deg": _nanpercentile_col(
            data_initial,
            "pole_apex_error_direction_abs_initial_deg",
            84,
        ),

        # ============================================================
        # Pole-apex inverse direction residuals: initial
        # ============================================================

        "pole_apex_error_direction_inverse_ratio_initial_median": _nanmedian_col(
            data_initial,
            "pole_apex_error_direction_inverse_ratio_initial",
        ),
        "pole_apex_error_direction_inverse_ratio_initial_rms": _nanrms_col(
            data_initial,
            "pole_apex_error_direction_inverse_ratio_initial",
        ),

        "pole_apex_error_direction_inverse_signed_initial_mean_deg": _nanmean_col(
            data_initial,
            "pole_apex_error_direction_inverse_signed_initial_deg",
        ),
        "pole_apex_error_direction_inverse_signed_initial_median_deg": _nanmedian_col(
            data_initial,
            "pole_apex_error_direction_inverse_signed_initial_deg",
        ),
        "pole_apex_error_direction_inverse_signed_initial_rms_deg": _nanrms_col(
            data_initial,
            "pole_apex_error_direction_inverse_signed_initial_deg",
        ),

        "pole_apex_error_direction_inverse_abs_initial_median_deg": _nanmedian_col(
            data_initial,
            "pole_apex_error_direction_inverse_abs_initial_deg",
        ),
        "pole_apex_error_direction_inverse_abs_initial_rms_deg": _nanrms_col(
            data_initial,
            "pole_apex_error_direction_inverse_abs_initial_deg",
        ),
        "pole_apex_error_direction_inverse_abs_initial_p16_deg": _nanpercentile_col(
            data_initial,
            "pole_apex_error_direction_inverse_abs_initial_deg",
            16,
        ),
        "pole_apex_error_direction_inverse_abs_initial_p84_deg": _nanpercentile_col(
            data_initial,
            "pole_apex_error_direction_inverse_abs_initial_deg",
            84,
        ),

        # ============================================================
        # Pole-apex geometric residuals: refined
        # ============================================================

        "pole_apex_dot_refined_median": _nanmedian_col(
            data_refined,
            "pole_apex_dot_refined",
        ),
        "pole_apex_dot_refined_rms": _nanrms_col(
            data_refined,
            "pole_apex_dot_refined",
        ),

        "pole_apex_error_geom_signed_refined_mean_deg": _nanmean_col(
            data_refined,
            "pole_apex_error_geom_signed_refined_deg",
        ),
        "pole_apex_error_geom_signed_refined_median_deg": _nanmedian_col(
            data_refined,
            "pole_apex_error_geom_signed_refined_deg",
        ),
        "pole_apex_error_geom_signed_refined_rms_deg": _nanrms_col(
            data_refined,
            "pole_apex_error_geom_signed_refined_deg",
        ),

        "pole_apex_error_geom_abs_refined_median_deg": _nanmedian_col(
            data_refined,
            "pole_apex_error_geom_abs_refined_deg",
        ),
        "pole_apex_error_geom_abs_refined_rms_deg": _nanrms_col(
            data_refined,
            "pole_apex_error_geom_abs_refined_deg",
        ),
        "pole_apex_error_geom_abs_refined_p16_deg": _nanpercentile_col(
            data_refined,
            "pole_apex_error_geom_abs_refined_deg",
            16,
        ),
        "pole_apex_error_geom_abs_refined_p84_deg": _nanpercentile_col(
            data_refined,
            "pole_apex_error_geom_abs_refined_deg",
            84,
        ),

        # ============================================================
        # Pole-apex direction-corrected residuals: refined
        # ============================================================

        "pole_apex_error_direction_ratio_refined_median": _nanmedian_col(
            data_refined,
            "pole_apex_error_direction_ratio_refined",
        ),
        "pole_apex_error_direction_ratio_refined_rms": _nanrms_col(
            data_refined,
            "pole_apex_error_direction_ratio_refined",
        ),

        "pole_apex_error_direction_signed_refined_mean_deg": _nanmean_col(
            data_refined,
            "pole_apex_error_direction_signed_refined_deg",
        ),
        "pole_apex_error_direction_signed_refined_median_deg": _nanmedian_col(
            data_refined,
            "pole_apex_error_direction_signed_refined_deg",
        ),
        "pole_apex_error_direction_signed_refined_rms_deg": _nanrms_col(
            data_refined,
            "pole_apex_error_direction_signed_refined_deg",
        ),

        "pole_apex_error_direction_abs_refined_median_deg": _nanmedian_col(
            data_refined,
            "pole_apex_error_direction_abs_refined_deg",
        ),
        "pole_apex_error_direction_abs_refined_rms_deg": _nanrms_col(
            data_refined,
            "pole_apex_error_direction_abs_refined_deg",
        ),
        "pole_apex_error_direction_abs_refined_p16_deg": _nanpercentile_col(
            data_refined,
            "pole_apex_error_direction_abs_refined_deg",
            16,
        ),
        "pole_apex_error_direction_abs_refined_p84_deg": _nanpercentile_col(
            data_refined,
            "pole_apex_error_direction_abs_refined_deg",
            84,
        ),

        # ============================================================
        # Pole-apex inverse direction residuals: refined
        # ============================================================

        "pole_apex_error_direction_inverse_ratio_refined_median": _nanmedian_col(
            data_refined,
            "pole_apex_error_direction_inverse_ratio_refined",
        ),
        "pole_apex_error_direction_inverse_ratio_refined_rms": _nanrms_col(
            data_refined,
            "pole_apex_error_direction_inverse_ratio_refined",
        ),

        "pole_apex_error_direction_inverse_signed_refined_mean_deg": _nanmean_col(
            data_refined,
            "pole_apex_error_direction_inverse_signed_refined_deg",
        ),
        "pole_apex_error_direction_inverse_signed_refined_median_deg": _nanmedian_col(
            data_refined,
            "pole_apex_error_direction_inverse_signed_refined_deg",
        ),
        "pole_apex_error_direction_inverse_signed_refined_rms_deg": _nanrms_col(
            data_refined,
            "pole_apex_error_direction_inverse_signed_refined_deg",
        ),

        "pole_apex_error_direction_inverse_abs_refined_median_deg": _nanmedian_col(
            data_refined,
            "pole_apex_error_direction_inverse_abs_refined_deg",
        ),
        "pole_apex_error_direction_inverse_abs_refined_rms_deg": _nanrms_col(
            data_refined,
            "pole_apex_error_direction_inverse_abs_refined_deg",
        ),
        "pole_apex_error_direction_inverse_abs_refined_p16_deg": _nanpercentile_col(
            data_refined,
            "pole_apex_error_direction_inverse_abs_refined_deg",
            16,
        ),
        "pole_apex_error_direction_inverse_abs_refined_p84_deg": _nanpercentile_col(
            data_refined,
            "pole_apex_error_direction_inverse_abs_refined_deg",
            84,
        ),
    }


def summarize_all_cluster_results(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Convierte una lista de resultados en una tabla resumen.
    """

    rows = [summarize_cluster_result(result) for result in results]
    return pd.DataFrame(rows)

def run_cluster_analysis_from_dataframe(
    df: pd.DataFrame,
    cluster_id: Optional[Union[str, int]] = None,
    group_col: str = "grupo",
    division_ra_col: str = "coordenada_ra",
    division_dec_col: str = "coordenada_dec",
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    parallax_col: str = "parallax",
    count_col: str = "ConteoAgrupaciones",
    present_col: str = "PresenteEnMuestras",
    cluster_col: str = "cluster_id",
    weight_col: Optional[str] = "probabilidad",
    weight_fill_value: Optional[float] = None,
    min_stars: int = 3,
    refine_apex: bool = True,
    min_sin_lambda: float = 0.15,
    min_pole_norm: float = 0.0,
    true_apex_vector: Optional[np.ndarray] = None,
    true_apex_ra_deg: Optional[float] = None,
    true_apex_dec_deg: Optional[float] = None,
    theta_clip_nsigma: float = 3.0,
    theta_clip_max_iter: int = 5,
    theta_clip_min_remaining: Optional[int] = None,
    theta_clip_apex_tolerance_deg: float = 1e-3,
    theta_clip_threshold_tolerance_deg: float = 1e-4,
    compute_cross_product_poles: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta el análisis completo desde un DataFrame de pandas ya cargado.

    Si cluster_id es None, procesa todos los cúmulos.
    Si cluster_id se proporciona, procesa solo ese cúmulo.

    El DataFrame debe contener, como mínimo:
        ra, dec, pmra, pmdec, parallax

    Y además, si no existe cluster_id:
        grupo, coordenada_ra, coordenada_dec

    Si weight_col='probabilidad' no existe, se calcula como:
        ConteoAgrupaciones / PresenteEnMuestras

    Si no quieres usar pesos, pasa:
        weight_col=None
    """

    prepared = prepare_dataframe(
        df,
        group_col=group_col,
        division_ra_col=division_ra_col,
        division_dec_col=division_dec_col,
        ra_col=ra_col,
        dec_col=dec_col,
        pmra_col=pmra_col,
        pmdec_col=pmdec_col,
        parallax_col=parallax_col,
        count_col=count_col,
        present_col=present_col,
        cluster_col=cluster_col,
        weight_col=weight_col if weight_col is not None else "probabilidad",
        weight_fill_value=weight_fill_value,
    )

    effective_weight_col = weight_col

    if weight_col is None:
        effective_weight_col = None

    if cluster_id is not None:
        cluster_result = process_single_cluster(
            df=prepared,
            cluster_id=cluster_id,
            cluster_col=cluster_col,
            weight_col=effective_weight_col,
            ra_col=ra_col,
            dec_col=dec_col,
            pmra_col=pmra_col,
            pmdec_col=pmdec_col,
            min_stars=min_stars,
            refine_apex=refine_apex,
            min_sin_lambda=min_sin_lambda,
            min_pole_norm=min_pole_norm,
            true_apex_vector=true_apex_vector,
            true_apex_ra_deg=true_apex_ra_deg,
            true_apex_dec_deg=true_apex_dec_deg,
            theta_clip_nsigma=theta_clip_nsigma,
            theta_clip_max_iter=theta_clip_max_iter,
            theta_clip_min_remaining=theta_clip_min_remaining,
            theta_clip_apex_tolerance_deg=theta_clip_apex_tolerance_deg,
            theta_clip_threshold_tolerance_deg=theta_clip_threshold_tolerance_deg,
            compute_cross_product_poles=compute_cross_product_poles,
        )

        summary = (
            summarize_cluster_result(cluster_result)
            if cluster_result is not None
            else None
        )

        return {
            "data": prepared,
            "cluster_result": cluster_result,
            "summary": summary,
        }

    cluster_results = process_all_clusters(
        df=prepared,
        cluster_col=cluster_col,
        weight_col=effective_weight_col,
        ra_col=ra_col,
        dec_col=dec_col,
        pmra_col=pmra_col,
        pmdec_col=pmdec_col,
        min_stars=min_stars,
        refine_apex=refine_apex,
        min_sin_lambda=min_sin_lambda,
        min_pole_norm=min_pole_norm,
        true_apex_vector=true_apex_vector,
        true_apex_ra_deg=true_apex_ra_deg,
        true_apex_dec_deg=true_apex_dec_deg,
        theta_clip_nsigma=theta_clip_nsigma,
        theta_clip_max_iter=theta_clip_max_iter,
        theta_clip_min_remaining=theta_clip_min_remaining,
        theta_clip_apex_tolerance_deg=theta_clip_apex_tolerance_deg,
        theta_clip_threshold_tolerance_deg=theta_clip_threshold_tolerance_deg,
        compute_cross_product_poles=compute_cross_product_poles,
    )

    summary_df = summarize_all_cluster_results(cluster_results)

    return {
        "data": prepared,
        "cluster_results": cluster_results,
        "summary": summary_df,
    }


# ============================================================
# High-level runner
# ============================================================

def run_cluster_analysis(
    data_path: Union[str, Path],
    cluster_id: Optional[Union[str, int]] = None,
    group_col: str = "grupo",
    division_ra_col: str = "coordenada_ra",
    division_dec_col: str = "coordenada_dec",
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    parallax_col: str = "parallax",
    count_col: str = "ConteoAgrupaciones",
    present_col: str = "PresenteEnMuestras",
    cluster_col: str = "cluster_id",
    weight_col: Optional[str] = "probabilidad",
    weight_fill_value: Optional[float] = None,
    min_stars: int = 3,
    refine_apex: bool = True,
    min_sin_lambda: float = 0.15,
    min_pole_norm: float = 0.0,
    true_apex_vector: Optional[np.ndarray] = None,
    true_apex_ra_deg: Optional[float] = None,
    true_apex_dec_deg: Optional[float] = None,
    theta_clip_nsigma: float = 3.0,
    theta_clip_max_iter: int = 5,
    theta_clip_min_remaining: Optional[int] = None,
    theta_clip_apex_tolerance_deg: float = 1e-3,
    theta_clip_threshold_tolerance_deg: float = 1e-4,
) -> Dict[str, Any]:
    """
    Ejecuta el análisis completo desde un archivo CSV.
    """

    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    return run_cluster_analysis_from_dataframe(
        df=df,
        cluster_id=cluster_id,
        group_col=group_col,
        division_ra_col=division_ra_col,
        division_dec_col=division_dec_col,
        ra_col=ra_col,
        dec_col=dec_col,
        pmra_col=pmra_col,
        pmdec_col=pmdec_col,
        parallax_col=parallax_col,
        count_col=count_col,
        present_col=present_col,
        cluster_col=cluster_col,
        weight_col=weight_col,
        weight_fill_value=weight_fill_value,
        min_stars=min_stars,
        refine_apex=refine_apex,
        min_sin_lambda=min_sin_lambda,
        min_pole_norm=min_pole_norm,
        true_apex_vector=true_apex_vector,
        true_apex_ra_deg=true_apex_ra_deg,
        true_apex_dec_deg=true_apex_dec_deg,
        theta_clip_nsigma=theta_clip_nsigma,
        theta_clip_max_iter=theta_clip_max_iter,
        theta_clip_min_remaining=theta_clip_min_remaining,
        theta_clip_apex_tolerance_deg=theta_clip_apex_tolerance_deg,
        theta_clip_threshold_tolerance_deg=theta_clip_threshold_tolerance_deg,
        compute_cross_product_poles=True,
    )


# ============================================================
# Optional CLI-style execution
# ============================================================

if __name__ == "__main__":
    # Ejemplo mínimo. Edita estas rutas/IDs según tu caso.
    #
    # result = run_cluster_analysis(
    #     data_path="data/datos_clusterizados.csv",
    #     cluster_id="8_123",
    # )
    #
    # print(result["summary"])
    pass