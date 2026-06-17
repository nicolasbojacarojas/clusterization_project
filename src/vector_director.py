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

    pole_plane_residual = np.clip(np.abs(poles @ apex_vector), 0.0, 1.0)
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
) -> Optional[Dict[str, Any]]:
    """
    Procesa un único cúmulo y estima su ápex inicial y refinado.

    El refinamiento elimina estrellas cercanas al ápex o antápex usando:

        sin(lambda) >= min_sin_lambda

    Esto evita fuentes con movimiento tangencial muy pequeño, donde el polo
    del círculo máximo queda peor condicionado.
    """

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

    df_initial["lambda_deg_initial"] = df_initial["lambda_deg"]
    df_initial["sin_lambda_initial"] = df_initial["sin_lambda"]
    df_initial["cos_lambda_initial"] = df_initial["cos_lambda"]

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

    df_refined = df_initial.copy()
    apex_refined = apex_initial
    cross_refined = cross_initial

    if refine_apex:
        refinement_mask = (
            df_initial["sin_lambda"].notna()
            & np.isfinite(df_initial["sin_lambda"])
            & (df_initial["sin_lambda"] >= min_sin_lambda)
            & (df_initial["pole_norm"] > min_pole_norm)
        )

        df_refined_candidate = df_initial[refinement_mask].copy()

        if len(df_refined_candidate) >= min_stars:
            try:
                apex_refined = estimate_apex_and_antapex(
                    df_refined_candidate,
                    weight_col=effective_weight_col,
                    min_sources=min_stars,
                )

                df_refined = add_lambda_angle_from_apex(
                    df_refined_candidate,
                    apex_vector=apex_refined["apex_vector"],
                    ra_col=ra_col,
                    dec_col=dec_col,
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
                        f"Cluster {cluster_id}: cross-product refined apex "
                        f"failed - {exc}"
                    )
                    cross_refined = None

            except ValueError as exc:
                print(
                    f"Cluster {cluster_id}: refined apex failed - {exc}. "
                    "Using initial apex."
                )

        else:
            print(
                f"Cluster {cluster_id}: only {len(df_refined_candidate)} "
                "stars after refinement cut. Using initial apex."
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

    return {
        "cluster_id": result["cluster_id"],
        "n_stars": result["n_stars"],
        "n_stars_initial": result["n_stars_initial"],
        "n_stars_refined": result["n_stars_refined"],

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