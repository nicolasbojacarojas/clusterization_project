"""Utilities for stellar apex analysis on a single cluster.

This module extracts the core preprocessing and analysis steps from the
notebook, but in a reusable Python library.

Example:
    from src.vector_director import run_cluster_analysis

    result = run_cluster_analysis(
        data_path="data/datos_resultados_modularizado/datos_clusterizados_todos_5d_f00.csv",
        cluster_id="8_123",
    )

    print(result["apex_result"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)


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


def add_galactic_coordinates(
    df: pd.DataFrame,
    ra_col: str = "ra",
    dec_col: str = "dec",
    copy: bool = True,
) -> pd.DataFrame:
    validate_columns(df, [ra_col, dec_col])
    result = df.copy() if copy else df

    coords = SkyCoord(
        ra=result[ra_col].to_numpy() * u.deg,
        dec=result[dec_col].to_numpy() * u.deg,
        frame="icrs",
    )
    galactic_coords = coords.galactic

    result["l"] = galactic_coords.l.degree
    result["b"] = galactic_coords.b.degree
    result["l_rad"] = np.radians(result["l"].to_numpy() - 180.0)
    result["b_rad"] = np.radians(result["b"].to_numpy())

    return result


def add_galactic_proper_motions(
    df: pd.DataFrame,
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    copy: bool = True,
    include_mu_l: bool = False,
) -> pd.DataFrame:
    required_columns = [ra_col, dec_col, pmra_col, pmdec_col]
    validate_columns(df, required_columns)

    result = df.copy() if copy else df

    coords = SkyCoord(
        ra=result[ra_col].to_numpy() * u.deg,
        dec=result[dec_col].to_numpy() * u.deg,
        pm_ra_cosdec=result[pmra_col].to_numpy() * u.mas / u.yr,
        pm_dec=result[pmdec_col].to_numpy() * u.mas / u.yr,
        frame="icrs",
    )
    galactic_coords = coords.galactic

    result["pm_l_cosb"] = galactic_coords.pm_l_cosb.to_value(u.mas / u.yr)
    result["pm_b"] = galactic_coords.pm_b.to_value(u.mas / u.yr)

    if include_mu_l:
        b_rad = galactic_coords.b.to_value(u.rad)
        cos_b = np.cos(b_rad)
        result["pm_l"] = np.where(
            np.abs(cos_b) > 1e-12,
            result["pm_l_cosb"].to_numpy() / cos_b,
            np.nan,
        )

    return result


def add_initial_final_galactic_vectors(
    df: pd.DataFrame,
    l_col: str = "l",
    b_col: str = "b",
    pm_l_col: str = "pm_l_cosb",
    pm_b_col: str = "pm_b",
    time_years: float = 1.0,
    angles_in_degrees: bool = True,
    pm_l_is_cosb: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    validate_columns(df, [l_col, b_col, pm_l_col, pm_b_col])
    result = df.copy() if copy else df

    if angles_in_degrees:
        l_rad = np.deg2rad(result[l_col].to_numpy(dtype=float))
        b_rad = np.deg2rad(result[b_col].to_numpy(dtype=float))
    else:
        l_rad = result[l_col].to_numpy(dtype=float)
        b_rad = result[b_col].to_numpy(dtype=float)

    pm_l_values = result[pm_l_col].to_numpy(dtype=float)
    pm_b_values = result[pm_b_col].to_numpy(dtype=float)
    cos_b = np.cos(b_rad)

    if pm_l_is_cosb:
        near_pole = np.abs(cos_b) < 1e-12
        if near_pole.any():
            raise ValueError(
                "Some sources have cos(b) too close to zero. "
                "Cannot safely compute mu_l = pm_l_cosb / cos(b)."
            )
        mu_l_masyr = pm_l_values / cos_b
    else:
        mu_l_masyr = pm_l_values

    mu_b_masyr = pm_b_values
    delta_l_rad = mu_l_masyr * MAS_TO_RAD * time_years
    delta_b_rad = mu_b_masyr * MAS_TO_RAD * time_years

    l_final_rad = l_rad + delta_l_rad
    b_final_rad = b_rad + delta_b_rad

    result["x_initial"] = np.cos(l_rad) * np.cos(b_rad)
    result["y_initial"] = np.sin(l_rad) * np.cos(b_rad)
    result["z_initial"] = np.sin(b_rad)

    result["x_final"] = np.cos(l_final_rad) * np.cos(b_final_rad)
    result["y_final"] = np.sin(l_final_rad) * np.cos(b_final_rad)
    result["z_final"] = np.sin(b_final_rad)

    result["delta_l_rad"] = delta_l_rad
    result["delta_b_rad"] = delta_b_rad
    result["l_final_rad"] = l_final_rad
    result["b_final_rad"] = b_final_rad

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
        normalized[valid_norm] = pole_vectors[valid_norm] / pole_norm[valid_norm, None]
        result["pole_x_unit"] = normalized[:, 0]
        result["pole_y_unit"] = normalized[:, 1]
        result["pole_z_unit"] = normalized[:, 2]

    return result


def unit_vector_to_galactic_lb(vector: np.ndarray) -> Dict[str, float]:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Input vector has zero norm.")

    x, y, z = vector / norm
    l_rad = np.arctan2(y, x) % (2.0 * np.pi)
    b_rad = np.arcsin(np.clip(z, -1.0, 1.0))

    return {
        "l_deg": np.degrees(l_rad),
        "b_deg": np.degrees(b_rad),
        "l_rad": l_rad,
        "b_rad": b_rad,
    }


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
    required_columns = [pole_x_col, pole_y_col, pole_z_col]
    if weight_col is not None:
        required_columns.append(weight_col)
    if orient_with_motion:
        required_columns.extend(
            [
                x_initial_col,
                y_initial_col,
                z_initial_col,
                x_final_col,
                y_final_col,
                z_final_col,
            ]
        )
    validate_columns(df, required_columns)

    poles = df[[pole_x_col, pole_y_col, pole_z_col]].to_numpy(dtype=float)
    valid_mask = np.isfinite(poles).all(axis=1)
    pole_norms = np.linalg.norm(poles, axis=1)
    valid_mask &= pole_norms > 0.0

    if weight_col is not None:
        weights_all = df[weight_col].to_numpy(dtype=float)
        valid_mask &= np.isfinite(weights_all)
        valid_mask &= weights_all > 0.0
    else:
        weights_all = np.ones(len(df), dtype=float)

    if valid_mask.sum() < min_sources:
        raise ValueError(
            "Not enough valid sources to estimate apex. "
            f"Found {valid_mask.sum()}, required {min_sources}."
        )

    poles = poles[valid_mask] / pole_norms[valid_mask, None]
    weights = weights_all[valid_mask]
    weights = weights / np.sum(weights)

    covariance_matrix = (poles * weights[:, None]).T @ poles
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    smallest_index = np.argmin(eigenvalues)

    apex_vector = eigenvectors[:, smallest_index]
    apex_vector = apex_vector / np.linalg.norm(apex_vector)
    toward_apex_fraction = np.nan
    oriented = False

    if orient_with_motion:
        initial_vectors = df.loc[valid_mask, [x_initial_col, y_initial_col, z_initial_col]].to_numpy(dtype=float)
        final_vectors = df.loc[valid_mask, [x_final_col, y_final_col, z_final_col]].to_numpy(dtype=float)
        initial_norms = np.linalg.norm(initial_vectors, axis=1)
        final_norms = np.linalg.norm(final_vectors, axis=1)
        vector_valid = (initial_norms > 0.0) & (final_norms > 0.0)

        if vector_valid.sum() >= min_sources:
            orientation_weights = weights[vector_valid]
            initial_vectors = initial_vectors[vector_valid] / initial_norms[vector_valid, None]
            final_vectors = final_vectors[vector_valid] / final_norms[vector_valid, None]
            initial_projection = initial_vectors @ apex_vector
            final_projection = final_vectors @ apex_vector
            moving_toward_apex = final_projection > initial_projection
            toward_apex_fraction = np.average(
                moving_toward_apex.astype(float), weights=orientation_weights
            )
            if toward_apex_fraction < 0.5:
                apex_vector = -apex_vector
                toward_apex_fraction = 1.0 - toward_apex_fraction
            oriented = True

    antapex_vector = -apex_vector
    apex_coordinates = unit_vector_to_galactic_lb(apex_vector)
    antapex_coordinates = unit_vector_to_galactic_lb(antapex_vector)

    pole_plane_residual = np.clip(np.abs(poles @ apex_vector), 0.0, 1.0)
    pole_residual_deg = np.degrees(np.arcsin(pole_plane_residual))

    return {
        "apex_vector": apex_vector,
        "antapex_vector": antapex_vector,
        "apex_l_deg": apex_coordinates["l_deg"],
        "apex_b_deg": apex_coordinates["b_deg"],
        "apex_l_rad": apex_coordinates["l_rad"],
        "apex_b_rad": apex_coordinates["b_rad"],
        "antapex_l_deg": antapex_coordinates["l_deg"],
        "antapex_b_deg": antapex_coordinates["b_deg"],
        "antapex_l_rad": antapex_coordinates["l_rad"],
        "antapex_b_rad": antapex_coordinates["b_rad"],
        "n_sources": int(valid_mask.sum()),
        "eigenvalues": eigenvalues,
        "rms_pole_residual_deg": np.sqrt(np.average(pole_residual_deg ** 2, weights=weights)),
        "median_pole_residual_deg": np.median(pole_residual_deg),
        "oriented_with_motion": oriented,
        "toward_apex_fraction": toward_apex_fraction,
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
) -> Dict[str, Any]:
    """
    Estima el ápex usando intersecciones entre pares de polos.

    Cada par de polos p_i, p_j define una intersección:

        z_ij = p_i x p_j

    La dirección z_ij tiene degeneración ápex/antápex. Por eso se orienta
    usando `reference_apex_vector`.
    """

    required = [pole_x_col, pole_y_col, pole_z_col]

    if weight_col is not None:
        required.append(weight_col)

    validate_columns(df, required)

    poles_all = df[[pole_x_col, pole_y_col, pole_z_col]].to_numpy(
        dtype=float
    )

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

    intersections = []
    intersection_weights = []

    n_poles = len(poles)

    for i in range(n_poles):
        for j in range(i + 1, n_poles):
            cross = np.cross(poles[i], poles[j])
            cross_norm = np.linalg.norm(cross)

            if cross_norm <= min_cross_norm:
                continue

            unit_cross = cross / cross_norm

            if np.dot(unit_cross, reference_apex_vector) < 0.0:
                unit_cross = -unit_cross

            pair_weight = weights[i] * weights[j] * cross_norm

            intersections.append(unit_cross)
            intersection_weights.append(pair_weight)

    if len(intersections) == 0:
        raise ValueError("No valid cross-product intersections found.")

    intersections = np.vstack(intersections)
    intersection_weights = np.asarray(intersection_weights, dtype=float)

    weight_sum = np.sum(intersection_weights)

    if weight_sum <= 0.0:
        raise ValueError("Cross-product weights are all zero.")

    intersection_weights = intersection_weights / weight_sum

    mean_vector = np.sum(
        intersections * intersection_weights[:, None],
        axis=0,
    )

    mean_norm = np.linalg.norm(mean_vector)

    if mean_norm == 0.0:
        raise ValueError("Mean cross-product vector has zero norm.")

    apex_vector = mean_vector / mean_norm
    antapex_vector = -apex_vector

    apex_coordinates = unit_vector_to_galactic_lb(apex_vector)
    antapex_coordinates = unit_vector_to_galactic_lb(antapex_vector)

    angular_residuals = np.degrees(
        np.arccos(np.clip(intersections @ apex_vector, -1.0, 1.0))
    )

    return {
        "apex_vector": apex_vector,
        "antapex_vector": antapex_vector,
        "apex_l_deg": apex_coordinates["l_deg"],
        "apex_b_deg": apex_coordinates["b_deg"],
        "antapex_l_deg": antapex_coordinates["l_deg"],
        "antapex_b_deg": antapex_coordinates["b_deg"],
        "n_intersections": int(len(intersections)),
        "rms_intersection_residual_deg": float(
            np.sqrt(np.mean(angular_residuals**2))
        ),
        "median_intersection_residual_deg": float(
            np.median(angular_residuals)
        ),
    }


def galactic_lb_to_unit_vector(l_deg: float, b_deg: float) -> np.ndarray:
    l_rad = np.deg2rad(l_deg)
    b_rad = np.deg2rad(b_deg)
    return np.array(
        [
            np.cos(l_rad) * np.cos(b_rad),
            np.sin(l_rad) * np.cos(b_rad),
            np.sin(b_rad),
        ],
        dtype=float,
    )


def add_lambda_angle_from_apex(
    df: pd.DataFrame,
    apex_vector: Optional[np.ndarray] = None,
    apex_l_deg: Optional[float] = None,
    apex_b_deg: Optional[float] = None,
    l_col: str = "l",
    b_col: str = "b",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Agrega el ángulo lambda entre cada estrella y el ápex.

    Esta versión usa directamente trigonometría esférica:

        cos(lambda) =
            sin(b) sin(b_apex)
            + cos(b) cos(b_apex) cos(l_apex - l)

    y calcula lambda con:

        lambda = atan2(sin(lambda), cos(lambda))

    en vez de usar arccos directamente. Esto es más estable
    numéricamente cerca del ápex y del antápex.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada. Debe contener coordenadas galácticas.

    apex_vector : np.ndarray or None
        Vector unitario del ápex en coordenadas galácticas cartesianas.
        Si no se proporciona, deben pasarse `apex_l_deg` y `apex_b_deg`.

    apex_l_deg : float or None
        Longitud galáctica del ápex, en grados.

    apex_b_deg : float or None
        Latitud galáctica del ápex, en grados.

    l_col : str, optional
        Columna con longitud galáctica, en grados.

    b_col : str, optional
        Columna con latitud galáctica, en grados.

    copy : bool, optional
        Si True, devuelve una copia modificada.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas nuevas:

        - lambda_rad
        - lambda_deg
        - sin_lambda
        - cos_lambda
        - delta_l_apex_rad
        - delta_l_apex_deg
        - lambda_tangent_l
        - lambda_tangent_b
    """

    validate_columns(df, [l_col, b_col])

    result = df.copy() if copy else df

    if apex_vector is not None:
        apex_coordinates = unit_vector_to_galactic_lb(apex_vector)
        apex_l_deg = apex_coordinates["l_deg"]
        apex_b_deg = apex_coordinates["b_deg"]

    if apex_l_deg is None or apex_b_deg is None:
        raise ValueError(
            "You must provide either apex_vector or both "
            "apex_l_deg and apex_b_deg."
        )

    l_rad = np.deg2rad(result[l_col].to_numpy(dtype=float))
    b_rad = np.deg2rad(result[b_col].to_numpy(dtype=float))

    apex_l_rad = np.deg2rad(float(apex_l_deg))
    apex_b_rad = np.deg2rad(float(apex_b_deg))

    delta_l_rad = apex_l_rad - l_rad
    delta_l_rad = (delta_l_rad + np.pi) % (2.0 * np.pi) - np.pi

    sin_b = np.sin(b_rad)
    cos_b = np.cos(b_rad)

    sin_b_apex = np.sin(apex_b_rad)
    cos_b_apex = np.cos(apex_b_rad)

    cos_delta_l = np.cos(delta_l_rad)
    sin_delta_l = np.sin(delta_l_rad)

    cos_lambda = (
        sin_b * sin_b_apex
        + cos_b * cos_b_apex * cos_delta_l
    )
    cos_lambda = np.clip(cos_lambda, -1.0, 1.0)

    tangent_l_numerator = cos_b_apex * sin_delta_l
    tangent_b_numerator = (
        cos_b * sin_b_apex
        - sin_b * cos_b_apex * cos_delta_l
    )

    sin_lambda = np.sqrt(
        tangent_l_numerator**2 + tangent_b_numerator**2
    )
    sin_lambda = np.clip(sin_lambda, 0.0, 1.0)

    lambda_rad = np.arctan2(sin_lambda, cos_lambda)

    result["lambda_rad"] = lambda_rad
    result["lambda_deg"] = np.rad2deg(lambda_rad)
    result["sin_lambda"] = sin_lambda
    result["cos_lambda"] = cos_lambda

    result["delta_l_apex_rad"] = delta_l_rad
    result["delta_l_apex_deg"] = np.rad2deg(delta_l_rad)

    valid_sin_lambda = sin_lambda > 1e-15

    result["lambda_tangent_l"] = np.where(
        valid_sin_lambda,
        tangent_l_numerator / sin_lambda,
        np.nan,
    )

    result["lambda_tangent_b"] = np.where(
        valid_sin_lambda,
        tangent_b_numerator / sin_lambda,
        np.nan,
    )

    return result


def prepare_dataframe(
    df: pd.DataFrame,
    group_col: str = "grupo",
    ra_col: str = "coordenada_ra",
    dec_col: str = "coordenada_dec",
    parallax_col: str = "parallax",
    count_col: str = "ConteoAgrupaciones",
    present_col: str = "PresenteEnMuestras",
    cluster_col: str = "cluster_id",
    weight_col: str = "probabilidad",
    weight_fill_value: Optional[float] = None,
) -> pd.DataFrame:
    result = df.copy()

    if cluster_col not in result.columns:
        result = create_cluster_id(
            result,
            group_col=group_col,
            ra_col=ra_col,
            dec_col=dec_col,
            cluster_col=cluster_col,
        )

    if "distance_pc" not in result.columns:
        validate_columns(result, [parallax_col])
        result["distance_pc"] = 1000.0 / result[parallax_col].astype(float)

    if weight_col not in result.columns:
        # If a fallback weight value was provided, use it for all rows.
        if weight_fill_value is not None:
            result[weight_col] = float(weight_fill_value)
        else:
            # Otherwise try to compute the probability from counts/presence.
            validate_columns(result, [count_col, present_col])
            result[weight_col] = (
                result[count_col].astype(float) / result[present_col].astype(float)
            )

    result = add_galactic_coordinates(result)
    result = add_galactic_proper_motions(result)
    result = add_initial_final_galactic_vectors(result)
    result = add_cross_product_poles(result)

    return result


def process_single_cluster(
    df: pd.DataFrame,
    cluster_id: Union[str, int],
    cluster_col: str = "cluster_id",
    weight_col: Optional[str] = "probabilidad",
    min_stars: int = 3,
    refine_apex: bool = True,
    min_sin_lambda: float = 0.15,
    min_pole_norm: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Procesa un único cúmulo y estima su ápex inicial y refinado.

    El refinado elimina estrellas cercanas al ápex o antápex usando:

        sin_lambda >= min_sin_lambda

    Esto evita fuentes con movimiento tangencial muy pequeño, donde el polo
    del círculo máximo queda peor condicionado.
    """

    validate_columns(df, [cluster_col])

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
        )
    except ValueError as exc:
        print(f"Cluster {cluster_id}: error computing initial apex - {exc}")
        return None

    df_initial = add_lambda_angle_from_apex(
        df_cluster,
        apex_vector=apex_initial["apex_vector"],
    )

    df_initial["lambda_deg_initial"] = df_initial["lambda_deg"]
    df_initial["sin_lambda_initial"] = df_initial["sin_lambda"]
    df_initial["cos_lambda_initial"] = df_initial["cos_lambda"]

    df_initial["pm_total"] = np.sqrt(
        df_initial["pm_l_cosb"] ** 2 + df_initial["pm_b"] ** 2
    )

    if "distance_pc" in df_initial.columns:
        df_initial["vt_kms"] = (
            4.74
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
    except Exception as exc:
        cross_initial = {"error": str(exc)}

    apex_refined = None
    cross_refined = None
    df_refined = df_initial.copy()

    refinement_summary = {
        "enabled": bool(refine_apex),
        "min_sin_lambda": float(min_sin_lambda),
        "min_pole_norm": float(min_pole_norm),
        "n_initial": int(len(df_initial)),
        "n_refined": int(len(df_initial)),
        "fraction_kept": 1.0,
        "status": "not_applied",
    }

    if refine_apex:
        refined_mask = (
            np.isfinite(df_initial["sin_lambda_initial"])
            & (df_initial["sin_lambda_initial"] >= min_sin_lambda)
        )

        if "pole_norm" in df_initial.columns:
            refined_mask &= (
                np.isfinite(df_initial["pole_norm"])
                & (df_initial["pole_norm"] >= min_pole_norm)
            )

        df_refined = df_initial.loc[refined_mask].copy()

        refinement_summary["n_refined"] = int(len(df_refined))
        refinement_summary["fraction_kept"] = float(
            len(df_refined) / len(df_initial)
        )

        if len(df_refined) < min_stars:
            refinement_summary["status"] = "not_enough_sources_after_filter"

            apex_refined = {
                "error": (
                    "Not enough sources after refinement filter. "
                    f"Found {len(df_refined)}, required {min_stars}."
                )
            }
            cross_refined = {
                "error": "Refined apex was not computed."
            }

        else:
            try:
                apex_refined = estimate_apex_and_antapex(
                    df_refined,
                    weight_col=effective_weight_col,
                )

                df_refined = add_lambda_angle_from_apex(
                    df_refined,
                    apex_vector=apex_refined["apex_vector"],
                )

                df_refined["lambda_deg_refined"] = df_refined["lambda_deg"]
                df_refined["sin_lambda_refined"] = df_refined["sin_lambda"]
                df_refined["cos_lambda_refined"] = df_refined["cos_lambda"]

                cross_refined = apex_from_pole_cross_products(
                    df_refined,
                    reference_apex_vector=apex_refined["apex_vector"],
                    weight_col=effective_weight_col,
                    pole_norm_min=min_pole_norm,
                    min_sources=min_stars,
                )

                refinement_summary["status"] = "ok"

            except Exception as exc:
                refinement_summary["status"] = "failed"
                apex_refined = {"error": str(exc)}
                cross_refined = {"error": str(exc)}

    preferred_apex = (
        apex_refined
        if isinstance(apex_refined, dict) and "error" not in apex_refined
        else apex_initial
    )

    preferred_cross_apex = (
        cross_refined
        if isinstance(cross_refined, dict) and "error" not in cross_refined
        else cross_initial
    )

    return {
        "cluster_id": cluster_id,
        "n_stars": int(len(df_initial)),
        "n_stars_refined": int(len(df_refined)),
        "weight_col_used": effective_weight_col,
        "refinement": refinement_summary,

        # Salida recomendada para usar directamente.
        "apex_result": preferred_apex,
        "apex_from_pole_crosses": preferred_cross_apex,

        # Salidas explícitas.
        "apex_initial": apex_initial,
        "apex_refined": apex_refined,
        "apex_from_pole_crosses_initial": cross_initial,
        "apex_from_pole_crosses_refined": cross_refined,

        # DataFrames.
        "df": df_refined,
        "df_initial": df_initial,
        "df_refined": df_refined,
    }


def run_cluster_analysis_from_dataframe(
    df: pd.DataFrame,
    cluster_id: Union[str, int],
    weight_col: Optional[str] = "probabilidad",
    weight_fill_value: Optional[float] = 1.0,
    **prepare_kwargs: Any,
) -> Dict[str, Any]:
    """Run apex analysis using an already-loaded DataFrame.

    By default `weight_fill_value=1.0` so the function will not fail
    when probability/count columns are missing.
    """
    # Ensure the prepare step receives the fallback weight value.
    prepare_kwargs.setdefault("weight_fill_value", weight_fill_value)
    return run_cluster_analysis(
        df,
        cluster_id=cluster_id,
        weight_col=weight_col,
        **prepare_kwargs,
    )


def run_cluster_analysis(
    source: Union[str, Path, pd.DataFrame],
    cluster_id: Union[str, int],
    weight_col: Optional[str] = "probabilidad",
    weight_fill_value: Optional[float] = 1.0,
    **prepare_kwargs: Any,
) -> Dict[str, Any]:
    """Run apex analysis from a CSV path or a pandas DataFrame.

    Parameters
    ----------
    source : Union[str, Path, pd.DataFrame]
        Path to a CSV file or a loaded pandas DataFrame.
    cluster_id : Union[str, int]
        Cluster identifier.
    weight_col : Optional[str]
        Column used for weights.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        df = pd.read_csv(path)
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise TypeError("source must be a file path or a pandas DataFrame")

    # Pass default fallback weight into prepare step unless overridden.
    prepare_kwargs.setdefault("weight_fill_value", weight_fill_value)

    df_prepared = prepare_dataframe(df, **prepare_kwargs)
    result = process_single_cluster(
        df_prepared,
        cluster_id=cluster_id,
        weight_col=weight_col,
    )

    if result is None:
        raise ValueError(f"Cluster {cluster_id} could not be processed.")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run single-cluster apex analysis."
    )
    parser.add_argument(
        "data_path",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "cluster_id",
        help="Cluster identifier to process",
    )
    parser.add_argument(
        "--weight-col",
        default="probabilidad",
        help="Column to use for weights (default: probabilidad)",
    )
    args = parser.parse_args()

    analysis_result = run_cluster_analysis(
        args.data_path,
        cluster_id=args.cluster_id,
        weight_col=args.weight_col,
    )

    apex = analysis_result["apex_result"]
    print(f"Cluster: {analysis_result['cluster_id']}")
    print(f"Stars: {analysis_result['n_stars']}")
    print(f"Apex l,b = {apex['apex_l_deg']:.3f}, {apex['apex_b_deg']:.3f}")
    print(f"Antapex l,b = {apex['antapex_l_deg']:.3f}, {apex['antapex_b_deg']:.3f}")
    print(f"RMS residual = {apex['rms_pole_residual_deg']:.4f} deg")
