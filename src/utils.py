from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    Galactic,
    ICRS,
    SkyCoord,
)

try:
    from IPython.display import display
except ImportError:
    display = print
import importlib
import src.vector_director as vd
importlib.reload(vd)
from src.vector_director import run_cluster_analysis_from_dataframe
KM_S_PER_ARCSEC_YR_PC = 4.74047
MAS_TO_RAD = np.deg2rad(1.0 / 3_600_000.0)
KM_S_PER_AU_YR = 4.74047

def unit_vector_to_galactic_lb(vector: np.ndarray) -> dict:
    """
    Convierte un vector cartesiano galáctico 3D a coordenadas l, b.

    Parameters
    ----------
    vector : np.ndarray
        Vector 3D en coordenadas galácticas cartesianas.

    Returns
    -------
    dict
        Diccionario con l_deg, b_deg, l_rad, b_rad.
    """

    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm == 0.0:
        raise ValueError("Input vector has zero norm.")

    unit_vector = vector / norm

    x_coord, y_coord, z_coord = unit_vector

    l_rad = np.arctan2(y_coord, x_coord) % (2.0 * np.pi)
    b_rad = np.arcsin(np.clip(z_coord, -1.0, 1.0))

    return {
        "l_deg": np.degrees(l_rad),
        "b_deg": np.degrees(b_rad),
        "l_rad": l_rad,
        "b_rad": b_rad,
    }


def extract_cross_product_apex_candidates(
    analysis_result: dict,
    use_refined: bool = True,
    weight_col: str | None = None,
    pole_x_col: str = "pole_x_unit",
    pole_y_col: str = "pole_y_unit",
    pole_z_col: str = "pole_z_unit",
    min_cross_norm: float = 1e-12,
) -> pd.DataFrame:
    """
    Extrae todos los candidatos a ápex generados por productos cruzados
    entre pares de polos.

    Parameters
    ----------
    analysis_result : dict
        Resultado devuelto por `process_single_cluster`.

    use_refined : bool, optional
        Si True, usa el DataFrame refinado si existe.

    weight_col : str or None, optional
        Columna de pesos. Por ejemplo: "probabilidad".

    pole_x_col, pole_y_col, pole_z_col : str
        Columnas con los polos unitarios.

    min_cross_norm : float
        Norma mínima aceptada para el producto cruz entre dos polos.

    Returns
    -------
    pd.DataFrame
        Tabla con candidatos:

        - apex_l_deg
        - apex_b_deg
        - candidate_weight
        - cross_norm
        - i
        - j
    """

    if use_refined and "df_refined" in analysis_result:
        df = analysis_result["df_refined"].copy()
        apex_result = analysis_result.get("apex_refined")
    else:
        df = analysis_result["df_initial"].copy()
        apex_result = analysis_result.get("apex_initial")

    if apex_result is None or "error" in apex_result:
        apex_result = analysis_result["apex_result"]

    reference_apex_vector = np.asarray(
        apex_result["apex_vector"],
        dtype=float,
    )
    reference_apex_vector = (
        reference_apex_vector / np.linalg.norm(reference_apex_vector)
    )

    required_cols = [pole_x_col, pole_y_col, pole_z_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    poles = df[[pole_x_col, pole_y_col, pole_z_col]].to_numpy(dtype=float)

    valid = np.isfinite(poles).all(axis=1)
    pole_norms = np.linalg.norm(poles, axis=1)
    valid &= pole_norms > 0.0

    df_valid = df.loc[valid].copy()
    poles = poles[valid] / pole_norms[valid, None]

    if weight_col is not None and weight_col in df_valid.columns:
        weights = df_valid[weight_col].to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    else:
        weights = np.ones(len(df_valid), dtype=float)

    candidates = []

    n_poles = len(poles)

    for i in range(n_poles):
        for j in range(i + 1, n_poles):
            cross_vector = np.cross(poles[i], poles[j])
            cross_norm = np.linalg.norm(cross_vector)

            if cross_norm <= min_cross_norm:
                continue

            candidate_vector = cross_vector / cross_norm

            # Orientar hacia el mismo hemisferio que el ápex estimado.
            if np.dot(candidate_vector, reference_apex_vector) < 0.0:
                candidate_vector = -candidate_vector

            coords = unit_vector_to_galactic_lb(candidate_vector)

            candidate_weight = weights[i] * weights[j] * cross_norm

            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "apex_l_deg": coords["l_deg"],
                    "apex_b_deg": coords["b_deg"],
                    "candidate_weight": candidate_weight,
                    "cross_norm": cross_norm,
                }
            )

    if len(candidates) == 0:
        raise ValueError("No valid cross-product apex candidates found.")

    return pd.DataFrame(candidates)


def get_lambda_values_from_result(
    analysis_result: dict,
    use_refined: bool = True,
) -> np.ndarray:
    """
    Extrae los valores de lambda_deg desde el resultado del análisis.
    """

    if use_refined and "df_refined" in analysis_result:
        df = analysis_result["df_refined"]
    else:
        df = analysis_result["df_initial"]

    if "lambda_deg_refined" in df.columns:
        lambda_values = df["lambda_deg_refined"].to_numpy(dtype=float)
    elif "lambda_deg_initial" in df.columns:
        lambda_values = df["lambda_deg_initial"].to_numpy(dtype=float)
    elif "lambda_deg" in df.columns:
        lambda_values = df["lambda_deg"].to_numpy(dtype=float)
    else:
        raise ValueError(
            "No lambda column found. Expected lambda_deg, "
            "lambda_deg_initial or lambda_deg_refined."
        )

    return lambda_values[np.isfinite(lambda_values)]

def place_legend_outside(
    ax: plt.Axes,
    loc: str = "upper left",
) -> None:
    """
    Coloca la leyenda fuera del eje para evitar que se solape
    con las cajas de estadísticas o con los datos.
    """

    handles, labels = ax.get_legend_handles_labels()

    if len(handles) == 0:
        return

    ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
        fontsize=8,
    )


def wrap_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    """
    Lleva ángulos en grados al intervalo [-180, 180).
    """

    return (angle_deg + 180.0) % 360.0 - 180.0


def weighted_quantile(
    values: np.ndarray,
    quantiles: list[float],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calcula cuantiles ponderados.

    Parameters
    ----------
    values : np.ndarray
        Valores de entrada.

    quantiles : list[float]
        Cuantiles entre 0 y 1.

    weights : np.ndarray or None
        Pesos. Si es None, usa cuantiles no ponderados.

    Returns
    -------
    np.ndarray
        Cuantiles solicitados.
    """

    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    valid = np.isfinite(values)

    if weights is None:
        return np.quantile(values[valid], quantiles)

    weights = np.asarray(weights, dtype=float)
    valid &= np.isfinite(weights)
    valid &= weights > 0.0

    values = values[valid]
    weights = weights[valid]

    if len(values) == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumulative_weights = np.cumsum(weights)
    cumulative_weights /= cumulative_weights[-1]

    return np.interp(quantiles, cumulative_weights, values)


def weighted_mean_and_std(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Media y desviación estándar ponderadas.
    """

    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if weights is None:
        clean_values = values[valid]
        return float(np.mean(clean_values)), float(np.std(clean_values))

    weights = np.asarray(weights, dtype=float)
    valid &= np.isfinite(weights)
    valid &= weights > 0.0

    clean_values = values[valid]
    clean_weights = weights[valid]

    mean = np.average(clean_values, weights=clean_weights)
    variance = np.average(
        (clean_values - mean) ** 2,
        weights=clean_weights,
    )

    return float(mean), float(np.sqrt(variance))


def weighted_circular_mean_and_std_deg(
    angles_deg: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Media circular y desviación circular para ángulos en grados.

    Útil para longitud galáctica l, porque l = 359° y l = 1°
    están cerca angularmente.
    """

    angles_deg = np.asarray(angles_deg, dtype=float)
    valid = np.isfinite(angles_deg)

    if weights is None:
        clean_weights = np.ones(valid.sum(), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        valid &= np.isfinite(weights)
        valid &= weights > 0.0
        clean_weights = weights[valid]

    clean_angles_rad = np.deg2rad(angles_deg[valid])

    sin_mean = np.average(np.sin(clean_angles_rad), weights=clean_weights)
    cos_mean = np.average(np.cos(clean_angles_rad), weights=clean_weights)

    mean_rad = np.arctan2(sin_mean, cos_mean) % (2.0 * np.pi)

    resultant_length = np.sqrt(sin_mean**2 + cos_mean**2)
    resultant_length = np.clip(resultant_length, 1e-15, 1.0)

    std_rad = np.sqrt(-2.0 * np.log(resultant_length))

    return float(np.rad2deg(mean_rad)), float(np.rad2deg(std_rad))


def galactic_lb_to_unit_vectors(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
) -> np.ndarray:
    """
    Convierte coordenadas galácticas l, b a vectores cartesianos unitarios.
    """

    l_rad = np.deg2rad(l_deg)
    b_rad = np.deg2rad(b_deg)

    x_coord = np.cos(l_rad) * np.cos(b_rad)
    y_coord = np.sin(l_rad) * np.cos(b_rad)
    z_coord = np.sin(b_rad)

    return np.column_stack((x_coord, y_coord, z_coord))


def angular_separation_deg(
    vectors_a: np.ndarray,
    vector_b: np.ndarray,
) -> np.ndarray:
    """
    Calcula separación angular entre muchos vectores y un vector de referencia.
    """

    vectors_a = np.asarray(vectors_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    vectors_a = vectors_a / np.linalg.norm(vectors_a, axis=1)[:, None]
    vector_b = vector_b / np.linalg.norm(vector_b)

    cos_angle = np.clip(vectors_a @ vector_b, -1.0, 1.0)

    return np.rad2deg(np.arccos(cos_angle))


def add_stats_box(
    ax: plt.Axes,
    text: str,
    loc: str = "upper right",
) -> None:
    """
    Agrega una caja de texto con estadísticas dentro de un eje.
    """

    positions = {
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left": (0.03, 0.97, "left", "top"),
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left": (0.03, 0.03, "left", "bottom"),
    }

    x_pos, y_pos, ha_text, va_text = positions[loc]

    ax.text(
        x_pos,
        y_pos,
        text,
        transform=ax.transAxes,
        ha=ha_text,
        va=va_text,
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "gray",
        },
    )


def plot_apex_candidate_diagnostics(
    analysis_result: dict,
    true_apex_l_deg: float | None = None,
    true_apex_b_deg: float | None = None,
    use_refined: bool = True,
    weight_col: str | None = None,
    bins_l: int = 40,
    bins_b: int = 40,
    bins_lambda: int = 40,
    bins_residual: int = 40,
    figsize: tuple[float, float] = (14.0, 9.0),
) -> tuple[plt.Figure, np.ndarray, object]:
    """
    Grafica diagnósticos de candidatos a ápex por producto cruz.

    Incluye:

    1. Histograma ponderado de l de candidatos.
    2. Histograma ponderado de b de candidatos.
    3. Histograma de lambda de las estrellas.
    4. Histograma del residuo angular de candidatos respecto al ápex.

    Parameters
    ----------
    analysis_result : dict
        Salida de `process_single_cluster`.

    true_apex_l_deg : float or None
        Longitud galáctica verdadera del ápex simulado, en grados.

    true_apex_b_deg : float or None
        Latitud galáctica verdadera del ápex simulado, en grados.

    use_refined : bool
        Si True, usa el resultado refinado.

    weight_col : str or None
        Columna de pesos para los candidatos.

    Returns
    -------
    tuple
        fig, axes, candidates_df.
    """

    candidates_df = extract_cross_product_apex_candidates(
        analysis_result=analysis_result,
        use_refined=use_refined,
        weight_col=weight_col,
    )

    lambda_values = get_lambda_values_from_result(
        analysis_result=analysis_result,
        use_refined=use_refined,
    )

    if use_refined and "apex_refined" in analysis_result:
        apex_result = analysis_result["apex_refined"]
    else:
        apex_result = analysis_result["apex_initial"]

    if apex_result is None or "error" in apex_result:
        apex_result = analysis_result["apex_result"]

    estimated_l = float(apex_result["apex_l_deg"])
    estimated_b = float(apex_result["apex_b_deg"])

    weights = candidates_df["candidate_weight"].to_numpy(dtype=float)

    candidate_vectors = galactic_lb_to_unit_vectors(
        candidates_df["apex_l_deg"].to_numpy(dtype=float),
        candidates_df["apex_b_deg"].to_numpy(dtype=float),
    )

    estimated_vector = np.asarray(apex_result["apex_vector"], dtype=float)

    residual_to_estimated = angular_separation_deg(
        candidate_vectors,
        estimated_vector,
    )

    candidates_df["residual_to_estimated_deg"] = residual_to_estimated

    if true_apex_l_deg is not None and true_apex_b_deg is not None:
        true_vector = galactic_lb_to_unit_vectors(
            np.array([true_apex_l_deg]),
            np.array([true_apex_b_deg]),
        )[0]

        residual_to_true = angular_separation_deg(
            candidate_vectors,
            true_vector,
        )

        candidates_df["residual_to_true_deg"] = residual_to_true
        residual_values = residual_to_true
        residual_label = "Residuo angular respecto al ápex verdadero [deg]"
    else:
        residual_values = residual_to_estimated
        residual_label = "Residuo angular respecto al ápex estimado [deg]"

    l_values = candidates_df["apex_l_deg"].to_numpy(dtype=float)
    b_values = candidates_df["apex_b_deg"].to_numpy(dtype=float)

    l_mean, l_std = weighted_circular_mean_and_std_deg(l_values, weights)
    b_mean, b_std = weighted_mean_and_std(b_values, weights)

    residual_mean, residual_std = weighted_mean_and_std(
        residual_values,
        weights,
    )

    residual_q16, residual_q50, residual_q84 = weighted_quantile(
        residual_values,
        [0.16, 0.50, 0.84],
        weights,
    )

    lambda_mean, lambda_std = weighted_mean_and_std(lambda_values)
    lambda_q16, lambda_q50, lambda_q84 = weighted_quantile(
        lambda_values,
        [0.16, 0.50, 0.84],
    )

    l_delta_to_estimated = wrap_angle_deg(l_values - estimated_l)
    l_delta_mean, l_delta_std = weighted_mean_and_std(
        l_delta_to_estimated,
        weights,
    )

    if true_apex_l_deg is not None:
        delta_l_true = wrap_angle_deg(np.array([estimated_l - true_apex_l_deg]))
        delta_l_true = float(delta_l_true[0])
    else:
        delta_l_true = np.nan

    if true_apex_b_deg is not None:
        delta_b_true = estimated_b - true_apex_b_deg
    else:
        delta_b_true = np.nan

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        constrained_layout=True,
    )

    ax_l = axes[0, 0]
    ax_b = axes[0, 1]
    ax_lambda = axes[1, 0]
    ax_residual = axes[1, 1]

    ax_l.hist(
        l_values,
        bins=bins_l,
        weights=weights,
        color="tab:blue",
        alpha=0.75,
        edgecolor="black",
    )

    ax_l.axvline(
        estimated_l,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Estimado: {estimated_l:.2f}°",
    )

    ax_l.axvline(
        l_mean,
        color="tab:orange",
        linestyle=":",
        linewidth=2,
        label=f"Media circ.: {l_mean:.2f}°",
    )

    if true_apex_l_deg is not None:
        ax_l.axvline(
            true_apex_l_deg,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Verdadero: {true_apex_l_deg:.2f}°",
        )

    ax_l.set_xlabel("l del candidato a ápex [deg]")
    ax_l.set_ylabel("Frecuencia ponderada")
    ax_l.set_title("Distribución de l")
    ax_l.legend(loc=1)

    l_stats = (
        f"N candidatos = {len(candidates_df)}\n"
        f"media circ. = {l_mean:.2f}°\n"
        f"std circ. = {l_std:.2f}°\n"
        f"Δl vs estimado = {l_delta_mean:.2f} ± {l_delta_std:.2f}°"
    )

    if np.isfinite(delta_l_true):
        l_stats += f"\nΔl estimado-true = {delta_l_true:.2f}°"

    add_stats_box(ax_l, l_stats, loc="upper left")

    ax_b.hist(
        b_values,
        bins=bins_b,
        weights=weights,
        color="tab:green",
        alpha=0.75,
        edgecolor="black",
    )

    ax_b.axvline(
        estimated_b,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Estimado: {estimated_b:.2f}°",
    )

    ax_b.axvline(
        b_mean,
        color="tab:orange",
        linestyle=":",
        linewidth=2,
        label=f"Media: {b_mean:.2f}°",
    )

    if true_apex_b_deg is not None:
        ax_b.axvline(
            true_apex_b_deg,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Verdadero: {true_apex_b_deg:.2f}°",
        )

    ax_b.set_xlabel("b del candidato a ápex [deg]")
    ax_b.set_ylabel("Frecuencia ponderada")
    ax_b.set_title("Distribución de b")
    ax_b.legend(loc=1)

    b_stats = (
        f"media = {b_mean:.2f}°\n"
        f"std = {b_std:.2f}°\n"
        f"estimado = {estimated_b:.2f}°"
    )

    if np.isfinite(delta_b_true):
        b_stats += f"\nΔb estimado-true = {delta_b_true:.2f}°"

    add_stats_box(ax_b, b_stats, loc="upper left")

    ax_lambda.hist(
        lambda_values,
        bins=bins_lambda,
        color="tab:purple",
        alpha=0.75,
        edgecolor="black",
    )

    ax_lambda.axvline(
        lambda_q50,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {lambda_q50:.2f}°",
    )

    ax_lambda.axvspan(
        lambda_q16,
        lambda_q84,
        color="tab:red",
        alpha=0.15,
        label="p16-p84",
    )

    ax_lambda.set_xlabel(r"$\lambda$ [deg]")
    ax_lambda.set_ylabel("Número de estrellas")
    ax_lambda.set_title(r"Distribución de $\lambda$")
    ax_lambda.legend(loc = 1)

    lambda_stats = (
        f"N estrellas = {len(lambda_values)}\n"
        f"media = {lambda_mean:.2f}°\n"
        f"std = {lambda_std:.2f}°\n"
        f"p16-p50-p84 = "
        f"{lambda_q16:.2f}, {lambda_q50:.2f}, {lambda_q84:.2f}°"
    )

    add_stats_box(ax_lambda, lambda_stats, loc="upper left")

    ax_residual.hist(
        residual_values,
        bins=bins_residual,
        weights=weights,
        color="tab:brown",
        alpha=0.75,
        edgecolor="black",
    )

    ax_residual.axvline(
        residual_q50,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {residual_q50:.2f}°",
    )

    ax_residual.axvspan(
        residual_q16,
        residual_q84,
        color="tab:red",
        alpha=0.15,
        label="p16-p84",
    )

    ax_residual.set_xlabel(residual_label)
    ax_residual.set_ylabel("Frecuencia ponderada")
    ax_residual.set_title("Residuo angular de candidatos")
    ax_residual.legend(loc=2)

    residual_stats = (
        f"media = {residual_mean:.2f}°\n"
        f"std = {residual_std:.2f}°\n"
        f"mediana = {residual_q50:.2f}°\n"
        f"p16-p84 = {residual_q16:.2f}–{residual_q84:.2f}°"
    )

    add_stats_box(ax_residual, residual_stats, loc="upper right")

    title_suffix = "refinado" if use_refined else "inicial"

    fig.suptitle(
        f"Diagnóstico del ápex por producto cruz ({title_suffix})",
        fontsize=15,
    )

    return fig, axes, candidates_df

# ============================================================
# Basic vector utilities
# ============================================================

def normalize_vectors(vectors, axis=1):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=axis)

    if axis == 1:
        out = np.full_like(vectors, np.nan, dtype=float)
        valid = norms > 0
        out[valid] = vectors[valid] / norms[valid, None]
        return out, norms

    if axis == 0:
        if norms == 0:
            return np.full_like(vectors, np.nan), norms
        return vectors / norms, norms

    raise ValueError("axis must be 0 or 1.")

# ============================================================
# Core astrometric diagnostics
# ============================================================


def angular_separation_vectors_deg(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def add_cluster_apex_diagnostics_icrs(
    df,
    apex_vector_icrs,
    weight_col="probabilidad",
    min_pole_norm=0.0,
):
    """
    Diagnósticos astrométricos en sistema ICRS/Gaia.

    Required columns:
        ra, dec, parallax, pmra, pmdec

    Optional:
        weight_col
        radial_velocity_kms_true
        radial_velocity
    """

    required = ["ra", "dec", "parallax", "pmra", "pmdec"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result_df = df.copy()

    if weight_col is None or weight_col not in result_df.columns:
        result_df["_weight_used"] = np.ones(len(result_df), dtype=float)
        result_df["_weight_used"] /= result_df["_weight_used"].sum()
    else:
        weights = result_df[weight_col].to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

        if np.sum(weights) <= 0:
            weights = np.ones(len(result_df), dtype=float)

        result_df["_weight_used"] = weights / np.sum(weights)

    ra_rad = np.deg2rad(result_df["ra"].to_numpy(dtype=float))
    dec_rad = np.deg2rad(result_df["dec"].to_numpy(dtype=float))

    pmra = result_df["pmra"].to_numpy(dtype=float)
    pmdec = result_df["pmdec"].to_numpy(dtype=float)

    cos_dec = np.cos(dec_rad)
    safe_cos_dec = np.where(np.abs(cos_dec) > 1e-12, cos_dec, np.nan)

    # ------------------------------------------------------------
    # Vector unitario de posición en ICRS
    # ------------------------------------------------------------

    star_vec = np.column_stack(
        (
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        )
    )

    result_df["x_icrs"] = star_vec[:, 0]
    result_df["y_icrs"] = star_vec[:, 1]
    result_df["z_icrs"] = star_vec[:, 2]

    # ------------------------------------------------------------
    # Posición final tras 1 año de movimiento propio
    # Gaia: pmra = mu_alpha* = mu_alpha cos(dec)
    # Por tanto: delta_ra = pmra / cos(dec)
    # ------------------------------------------------------------

    delta_ra_rad = (pmra / safe_cos_dec) * MAS_TO_RAD
    delta_dec_rad = pmdec * MAS_TO_RAD

    ra2_rad = (ra_rad + delta_ra_rad) % (2.0 * np.pi)
    dec2_rad = dec_rad + delta_dec_rad

    star_vec_final = np.column_stack(
        (
            np.cos(dec2_rad) * np.cos(ra2_rad),
            np.cos(dec2_rad) * np.sin(ra2_rad),
            np.sin(dec2_rad),
        )
    )

    poles = np.cross(star_vec, star_vec_final)
    poles_hat, pole_norm = normalize_vectors(poles, axis=1)

    valid_pole = np.isfinite(pole_norm) & (pole_norm > min_pole_norm)
    valid_pole &= np.all(np.isfinite(poles_hat), axis=1)

    result_df["pole_x"] = poles_hat[:, 0]
    result_df["pole_y"] = poles_hat[:, 1]
    result_df["pole_z"] = poles_hat[:, 2]
    result_df["pole_norm"] = pole_norm
    result_df["valid_pole"] = valid_pole

    # ------------------------------------------------------------
    # Lambda respecto al ápex ICRS
    # ------------------------------------------------------------

    apex_vector_icrs = np.asarray(apex_vector_icrs, dtype=float)
    apex_vector_icrs = apex_vector_icrs / np.linalg.norm(apex_vector_icrs)

    cos_lambda = np.clip(star_vec @ apex_vector_icrs, -1.0, 1.0)
    sin_lambda = np.sqrt(np.clip(1.0 - cos_lambda**2, 0.0, 1.0))

    lambda_rad = np.arctan2(sin_lambda, cos_lambda)

    result_df["cos_lambda"] = cos_lambda
    result_df["sin_lambda"] = sin_lambda
    result_df["lambda_deg"] = np.degrees(lambda_rad)

    # ------------------------------------------------------------
    # Residuos del polo respecto al plano perpendicular al ápex
    # ------------------------------------------------------------

    dot_pole_apex = np.full(len(result_df), np.nan)
    dot_pole_apex[valid_pole] = poles_hat[valid_pole] @ apex_vector_icrs

    pole_residual_rad = np.arcsin(
        np.clip(np.abs(dot_pole_apex), 0.0, 1.0)
    )

    result_df["pole_residual_deg"] = np.degrees(pole_residual_rad)

    # ------------------------------------------------------------
    # Movimiento propio total y velocidad tangencial
    # ------------------------------------------------------------

    mu_tot = np.sqrt(pmra**2 + pmdec**2)
    result_df["mu_total_masyr"] = mu_tot

    parallax = result_df["parallax"].to_numpy(dtype=float)

    distance_pc = np.full(len(result_df), np.nan)
    valid_parallax = parallax > 0
    distance_pc[valid_parallax] = 1000.0 / parallax[valid_parallax]

    result_df["distance_pc_from_parallax"] = distance_pc

    result_df["vt_kms"] = (
        KM_S_PER_AU_YR
        * (mu_tot / 1000.0)
        * distance_pc
    )

    # ------------------------------------------------------------
    # Alineación entre movimiento propio observado y dirección al ápex
    # en el plano tangente ecuatorial
    # ------------------------------------------------------------

    e_ra = np.column_stack(
        (
            -np.sin(ra_rad),
            np.cos(ra_rad),
            np.zeros_like(ra_rad),
        )
    )

    e_dec = np.column_stack(
        (
            -np.cos(ra_rad) * np.sin(dec_rad),
            -np.sin(ra_rad) * np.sin(dec_rad),
            np.cos(dec_rad),
        )
    )

    motion_tangent = pmra[:, None] * e_ra + pmdec[:, None] * e_dec
    motion_norm = np.linalg.norm(motion_tangent, axis=1)

    apex_tangent = apex_vector_icrs[None, :] - cos_lambda[:, None] * star_vec
    apex_tangent_norm = np.linalg.norm(apex_tangent, axis=1)

    cos_align = np.full(len(result_df), np.nan)

    valid_align = (
        np.isfinite(motion_norm)
        & np.isfinite(apex_tangent_norm)
        & (motion_norm > 0)
        & (apex_tangent_norm > 0)
    )

    cos_align[valid_align] = np.sum(
        motion_tangent[valid_align] * apex_tangent[valid_align],
        axis=1,
    ) / (
        motion_norm[valid_align]
        * apex_tangent_norm[valid_align]
    )

    cos_align = np.clip(cos_align, -1.0, 1.0)

    result_df["cos_motion_apex_alignment"] = cos_align
    result_df["motion_apex_alignment_deg"] = np.degrees(np.arccos(cos_align))

    return result_df

def compute_validation_summary(result, df_diag):
    """
    Builds a compact summary table from result and diagnostics.
    """

    apex = result.get("apex_result", result.get("apex_refined", {}))
    cross = result.get("apex_from_pole_crosses", result.get("apex_from_pole_crosses_refined", {}))
    refinement = result.get("refinement", {})

    apex_vector = np.asarray(apex["apex_vector"], dtype=float)

    summary = {}

    summary["cluster_id"] = result.get("cluster_id", None)
    summary["n_stars"] = result.get("n_stars", len(df_diag))
    summary["n_stars_refined"] = result.get("n_stars_refined", np.nan)

    summary["apex_ra_deg"] = apex.get("apex_ra_deg", np.nan)
    summary["apex_dec_deg"] = apex.get("apex_dec_deg", np.nan)
    summary["antapex_ra_deg"] = apex.get("antapex_ra_deg", np.nan)
    summary["antapex_dec_deg"] = apex.get("antapex_dec_deg", np.nan)

    summary["rms_pole_residual_deg"] = apex.get("rms_pole_residual_deg", np.nan)
    summary["median_pole_residual_deg"] = apex.get("median_pole_residual_deg", np.nan)
    summary["toward_apex_fraction"] = apex.get("toward_apex_fraction", np.nan)

    summary["refinement_status"] = refinement.get("status", None)
    summary["fraction_kept"] = refinement.get("fraction_kept", np.nan)
    summary["min_sin_lambda"] = refinement.get("min_sin_lambda", np.nan)

    if "apex_vector" in cross:
        cross_vector = np.asarray(cross["apex_vector"], dtype=float)
        summary["apex_cross_ra_deg"] = cross.get("apex_ra_deg", np.nan)
        summary["apex_cross_dec_deg"] = cross.get("apex_dec_deg", np.nan)
        summary["apex_cross_separation_deg"] = angular_separation_vectors_deg(
            apex_vector,
            cross_vector,
        )
        summary["n_cross_intersections"] = cross.get("n_intersections", np.nan)
        summary["rms_cross_residual_deg"] = cross.get("rms_intersection_residual_deg", np.nan)
        summary["median_cross_residual_deg"] = cross.get("median_intersection_residual_deg", np.nan)
    else:
        summary["apex_cross_separation_deg"] = np.nan

    eigenvalues = np.asarray(apex.get("eigenvalues", [np.nan, np.nan, np.nan]), dtype=float)

    if len(eigenvalues) >= 3:
        summary["eigenvalue_1"] = eigenvalues[0]
        summary["eigenvalue_2"] = eigenvalues[1]
        summary["eigenvalue_3"] = eigenvalues[2]
        summary["eigenvalue_ratio_1_2"] = eigenvalues[0] / eigenvalues[1]
        summary["eigenvalue_ratio_2_3"] = eigenvalues[1] / eigenvalues[2]

    w = df_diag["_weight_used"].to_numpy(dtype=float)

    for col in [
        "parallax",
        "distance_pc_from_parallax",
        "pmra",
        "pmdec",
        "mu_total_masyr",
        "vt_kms",
        "lambda_deg",
        "pole_residual_deg",
        "motion_apex_alignment_deg",
    ]:
        if col in df_diag.columns:
            mean, std = weighted_mean_and_std(df_diag[col].to_numpy(dtype=float), w)
            summary[f"{col}_weighted_mean"] = mean
            summary[f"{col}_weighted_std"] = std

    residuals = df_diag["pole_residual_deg"].to_numpy(dtype=float)
    summary["fraction_pole_residual_lt_1deg"] = float(np.nanmean(residuals < 1.0))
    summary["fraction_pole_residual_lt_2deg"] = float(np.nanmean(residuals < 2.0))
    summary["fraction_motion_alignment_lt_1deg"] = float(
        np.nanmean(df_diag["motion_apex_alignment_deg"].to_numpy(dtype=float) < 1.0)
    )
    summary["fraction_motion_alignment_lt_2deg"] = float(
        np.nanmean(df_diag["motion_apex_alignment_deg"].to_numpy(dtype=float) < 2.0)
    )

    summary.update(fit_speed_from_vt_sinlambda(df_diag))

    if "radial_velocity_kms_true" in df_diag.columns:
        summary.update(fit_speed_from_vr_coslambda(df_diag, rv_col="radial_velocity_kms_true"))
    elif "radial_velocity" in df_diag.columns:
        summary.update(fit_speed_from_vr_coslambda(df_diag, rv_col="radial_velocity"))
    else:
        summary.update({
            "rv_col_used": None,
            "speed_from_vr_kms": np.nan,
            "speed_from_vr_rms_kms": np.nan,
            "n_vr_fit": 0,
        })

    return pd.DataFrame([summary])

def compute_validation_summary_icrs(result, df_diag):
    """
    Builds a compact validation summary table from an ICRS/Gaia apex result
    and the star-by-star diagnostic dataframe.

    Expected result structure:
        result["cluster_id"]
        result["n_stars"]
        result["n_stars_refined"]
        result["apex_refined"] or result["apex_initial"]
        result["cross_refined"] or result["cross_initial"]

    Expected df_diag columns:
        _weight_used
        parallax
        distance_pc_from_parallax
        pmra
        pmdec
        mu_total_masyr
        vt_kms
        lambda_deg
        pole_residual_deg
        motion_apex_alignment_deg
    """

    apex = (
        result.get("apex_refined", None)
        or result.get("apex_initial", None)
        or result.get("apex_result", None)
    )

    if apex is None:
        raise ValueError(
            "Could not find apex_refined, apex_initial, or apex_result "
            "inside result."
        )

    if "apex_vector" not in apex:
        raise ValueError("Apex dictionary does not contain 'apex_vector'.")

    cross = (
        result.get("cross_refined", None)
        or result.get("cross_initial", None)
        or result.get("apex_from_pole_crosses_refined", None)
        or result.get("apex_from_pole_crosses", None)
    )

    apex_vector = np.asarray(apex["apex_vector"], dtype=float)

    summary = {}

    # ------------------------------------------------------------
    # Basic cluster information
    # ------------------------------------------------------------

    summary["cluster_id"] = result.get("cluster_id", None)
    summary["n_stars"] = result.get("n_stars", len(df_diag))
    summary["n_stars_initial"] = result.get("n_stars_initial", np.nan)
    summary["n_stars_refined"] = result.get("n_stars_refined", np.nan)

    n_stars = summary["n_stars"]
    n_stars_refined = summary["n_stars_refined"]

    if (
        n_stars is not None
        and np.isfinite(n_stars)
        and n_stars > 0
        and np.isfinite(n_stars_refined)
    ):
        summary["fraction_kept"] = float(n_stars_refined / n_stars)
    else:
        summary["fraction_kept"] = np.nan

    summary["weight_col_used"] = result.get("weight_col_used", None)

    # ------------------------------------------------------------
    # Apex and antapex in RA/Dec
    # ------------------------------------------------------------

    summary["apex_ra_deg"] = apex.get("apex_ra_deg", np.nan)
    summary["apex_dec_deg"] = apex.get("apex_dec_deg", np.nan)

    summary["antapex_ra_deg"] = apex.get("antapex_ra_deg", np.nan)
    summary["antapex_dec_deg"] = apex.get("antapex_dec_deg", np.nan)

    summary["rms_pole_residual_deg"] = apex.get(
        "rms_pole_residual_deg",
        np.nan,
    )

    summary["median_pole_residual_deg"] = apex.get(
        "median_pole_residual_deg",
        np.nan,
    )

    summary["toward_apex_fraction"] = apex.get(
        "toward_apex_fraction",
        np.nan,
    )

    summary["oriented_with_motion"] = apex.get(
        "oriented_with_motion",
        np.nan,
    )

    # ------------------------------------------------------------
    # Cross-product apex estimator
    # ------------------------------------------------------------

    if cross is not None and "apex_vector" in cross:
        cross_vector = np.asarray(cross["apex_vector"], dtype=float)

        summary["apex_cross_ra_deg"] = cross.get("apex_ra_deg", np.nan)
        summary["apex_cross_dec_deg"] = cross.get("apex_dec_deg", np.nan)

        summary["antapex_cross_ra_deg"] = cross.get(
            "antapex_ra_deg",
            np.nan,
        )
        summary["antapex_cross_dec_deg"] = cross.get(
            "antapex_dec_deg",
            np.nan,
        )

        summary["apex_cross_separation_deg"] = angular_separation_vectors_deg(
            apex_vector,
            cross_vector,
        )

        summary["n_cross_pairs_total"] = cross.get("n_pairs_total", np.nan)
        summary["n_cross_pairs_used"] = cross.get("n_pairs_used", np.nan)

        summary["rms_cross_residual_deg"] = cross.get(
            "rms_intersection_residual_deg",
            np.nan,
        )

        summary["median_cross_residual_deg"] = cross.get(
            "median_intersection_residual_deg",
            np.nan,
        )
    else:
        summary["apex_cross_ra_deg"] = np.nan
        summary["apex_cross_dec_deg"] = np.nan
        summary["antapex_cross_ra_deg"] = np.nan
        summary["antapex_cross_dec_deg"] = np.nan
        summary["apex_cross_separation_deg"] = np.nan
        summary["n_cross_pairs_total"] = np.nan
        summary["n_cross_pairs_used"] = np.nan
        summary["rms_cross_residual_deg"] = np.nan
        summary["median_cross_residual_deg"] = np.nan

    # ------------------------------------------------------------
    # Eigenvalues of pole-plane fit
    # ------------------------------------------------------------

    eigenvalues = np.asarray(
        apex.get("eigenvalues", [np.nan, np.nan, np.nan]),
        dtype=float,
    )

    if len(eigenvalues) >= 3:
        summary["eigenvalue_1"] = eigenvalues[0]
        summary["eigenvalue_2"] = eigenvalues[1]
        summary["eigenvalue_3"] = eigenvalues[2]

        if np.isfinite(eigenvalues[1]) and eigenvalues[1] != 0:
            summary["eigenvalue_ratio_1_2"] = eigenvalues[0] / eigenvalues[1]
        else:
            summary["eigenvalue_ratio_1_2"] = np.nan

        if np.isfinite(eigenvalues[2]) and eigenvalues[2] != 0:
            summary["eigenvalue_ratio_2_3"] = eigenvalues[1] / eigenvalues[2]
        else:
            summary["eigenvalue_ratio_2_3"] = np.nan
    else:
        summary["eigenvalue_1"] = np.nan
        summary["eigenvalue_2"] = np.nan
        summary["eigenvalue_3"] = np.nan
        summary["eigenvalue_ratio_1_2"] = np.nan
        summary["eigenvalue_ratio_2_3"] = np.nan

    # ------------------------------------------------------------
    # Weighted means and weighted standard deviations
    # ------------------------------------------------------------

    if "_weight_used" in df_diag.columns:
        w = df_diag["_weight_used"].to_numpy(dtype=float)
    else:
        w = np.ones(len(df_diag), dtype=float)

    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    if np.sum(w) <= 0:
        w = np.ones(len(df_diag), dtype=float)

    for col in [
        "parallax",
        "distance_pc_from_parallax",
        "pmra",
        "pmdec",
        "mu_total_masyr",
        "vt_kms",
        "lambda_deg",
        "pole_residual_deg",
        "motion_apex_alignment_deg",
    ]:
        if col in df_diag.columns:
            mean, std = weighted_mean_and_std(
                df_diag[col].to_numpy(dtype=float),
                w,
            )

            summary[f"{col}_weighted_mean"] = mean
            summary[f"{col}_weighted_std"] = std

    # ------------------------------------------------------------
    # Fractions below residual thresholds
    # ------------------------------------------------------------

    if "pole_residual_deg" in df_diag.columns:
        residuals = df_diag["pole_residual_deg"].to_numpy(dtype=float)

        summary["fraction_pole_residual_lt_1deg"] = float(
            np.nanmean(residuals < 1.0)
        )
        summary["fraction_pole_residual_lt_2deg"] = float(
            np.nanmean(residuals < 2.0)
        )
        summary["fraction_pole_residual_lt_5deg"] = float(
            np.nanmean(residuals < 5.0)
        )
    else:
        summary["fraction_pole_residual_lt_1deg"] = np.nan
        summary["fraction_pole_residual_lt_2deg"] = np.nan
        summary["fraction_pole_residual_lt_5deg"] = np.nan

    if "motion_apex_alignment_deg" in df_diag.columns:
        alignments = df_diag["motion_apex_alignment_deg"].to_numpy(dtype=float)

        summary["fraction_motion_alignment_lt_1deg"] = float(
            np.nanmean(alignments < 1.0)
        )
        summary["fraction_motion_alignment_lt_2deg"] = float(
            np.nanmean(alignments < 2.0)
        )
        summary["fraction_motion_alignment_lt_5deg"] = float(
            np.nanmean(alignments < 5.0)
        )
    else:
        summary["fraction_motion_alignment_lt_1deg"] = np.nan
        summary["fraction_motion_alignment_lt_2deg"] = np.nan
        summary["fraction_motion_alignment_lt_5deg"] = np.nan

    # ------------------------------------------------------------
    # Speed estimates
    # ------------------------------------------------------------

    if "vt_kms" in df_diag.columns and "sin_lambda" in df_diag.columns:
        summary.update(fit_speed_from_vt_sinlambda(df_diag))
    else:
        summary.update(
            {
                "speed_from_vt_kms": np.nan,
                "speed_from_vt_rms_kms": np.nan,
                "n_vt_fit": 0,
            }
        )

    if "radial_velocity_kms_true" in df_diag.columns:
        summary.update(
            fit_speed_from_vr_coslambda(
                df_diag,
                rv_col="radial_velocity_kms_true",
            )
        )
    elif "radial_velocity" in df_diag.columns:
        summary.update(
            fit_speed_from_vr_coslambda(
                df_diag,
                rv_col="radial_velocity",
            )
        )
    else:
        summary.update(
            {
                "rv_col_used": None,
                "speed_from_vr_kms": np.nan,
                "speed_from_vr_rms_kms": np.nan,
                "n_vr_fit": 0,
            }
        )

    return pd.DataFrame([summary])

def fit_apex_from_poles_icrs(
    poles,
    weights=None,
    reference_apex=None,
):
    poles = np.asarray(poles, dtype=float)

    valid = np.isfinite(poles).all(axis=1)
    norms = np.linalg.norm(poles, axis=1)
    valid &= norms > 0

    poles = poles[valid] / norms[valid, None]

    if weights is None:
        weights = np.ones(len(poles), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)[valid]

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

    if np.sum(weights) <= 0:
        weights = np.ones(len(poles), dtype=float)

    weights_norm = weights / np.sum(weights)

    covariance_matrix = (poles * weights_norm[:, None]).T @ poles

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    apex_vector = eigenvectors[:, 0]
    apex_vector = apex_vector / np.linalg.norm(apex_vector)

    if reference_apex is not None:
        reference_apex = np.asarray(reference_apex, dtype=float)
        reference_apex = reference_apex / np.linalg.norm(reference_apex)

        if np.dot(apex_vector, reference_apex) < 0:
            apex_vector = -apex_vector

    coords = unit_vector_to_equatorial_radec(apex_vector)

    pole_plane_residual = np.clip(np.abs(poles @ apex_vector), 0.0, 1.0)
    pole_residual_deg = np.degrees(np.arcsin(pole_plane_residual))

    return {
        "apex_vector": apex_vector,
        "apex_ra_deg": coords["ra_deg"],
        "apex_dec_deg": coords["dec_deg"],
        "eigenvalues": eigenvalues,
        "rms_pole_residual_deg": float(
            np.sqrt(np.average(pole_residual_deg**2, weights=weights))
        ),
        "median_pole_residual_deg": float(np.median(pole_residual_deg)),
    }


def bootstrap_apex_uncertainty_icrs(
    df_diag,
    reference_apex_vector,
    n_bootstrap=500,
    random_seed=42,
    weight_col="_weight_used",
):
    """
    Bootstrap uncertainty estimate for the apex using ICRS pole vectors.
    """

    rng = np.random.default_rng(random_seed)

    valid = df_diag["valid_pole"].to_numpy(dtype=bool)

    poles = df_diag.loc[
        valid,
        ["pole_x", "pole_y", "pole_z"],
    ].to_numpy(dtype=float)

    if weight_col in df_diag.columns:
        weights = df_diag.loc[valid, weight_col].to_numpy(dtype=float)
    else:
        weights = np.ones(len(poles), dtype=float)

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

    if len(poles) < 3:
        return pd.DataFrame()

    if np.sum(weights) <= 0:
        probabilities = np.ones(len(poles), dtype=float) / len(poles)
    else:
        probabilities = weights / np.sum(weights)

    rows = []

    for i in range(n_bootstrap):
        idx = rng.choice(
            np.arange(len(poles)),
            size=len(poles),
            replace=True,
            p=probabilities,
        )

        boot_poles = poles[idx]
        boot_weights = weights[idx]

        try:
            fit = fit_apex_from_poles_icrs(
                boot_poles,
                weights=boot_weights,
                reference_apex=reference_apex_vector,
            )

            delta_deg = angular_separation_vectors_deg(
                reference_apex_vector,
                fit["apex_vector"],
            )

            rows.append(
                {
                    "bootstrap_id": i,
                    "apex_ra_deg": fit["apex_ra_deg"],
                    "apex_dec_deg": fit["apex_dec_deg"],
                    "delta_from_reference_deg": delta_deg,
                    "rms_pole_residual_deg": fit[
                        "rms_pole_residual_deg"
                    ],
                    "median_pole_residual_deg": fit[
                        "median_pole_residual_deg"
                    ],
                }
            )

        except Exception:
            continue

    return pd.DataFrame(rows)

def fit_speed_from_vt_sinlambda(df_diag, weight_col="_weight_used"):
    """
    Fits V_t = V * sin(lambda), through the origin.
    """

    x = df_diag["sin_lambda"].to_numpy(dtype=float)
    y = df_diag["vt_kms"].to_numpy(dtype=float)

    if weight_col in df_diag.columns:
        w = df_diag[weight_col].to_numpy(dtype=float)
    else:
        w = np.ones(len(df_diag), dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    valid &= x > 0

    if np.sum(valid) < 3:
        return {
            "speed_from_vt_kms": np.nan,
            "speed_from_vt_rms_kms": np.nan,
            "n_vt_fit": int(np.sum(valid)),
        }

    x = x[valid]
    y = y[valid]
    w = w[valid]

    if np.sum(w) <= 0:
        w = np.ones(len(x), dtype=float)

    speed = np.sum(w * x * y) / np.sum(w * x * x)
    residual = y - speed * x
    rms = np.sqrt(np.average(residual**2, weights=w))

    return {
        "speed_from_vt_kms": float(speed),
        "speed_from_vt_rms_kms": float(rms),
        "n_vt_fit": int(len(x)),
    }

def fit_speed_from_vr_coslambda(df_diag, rv_col="radial_velocity_kms_true", weight_col="_weight_used"):
    """
    Fits V_r = V * cos(lambda), through the origin.
    Useful when true or observed radial velocities are available.
    """

    if rv_col not in df_diag.columns:
        return {
            "rv_col_used": None,
            "speed_from_vr_kms": np.nan,
            "speed_from_vr_rms_kms": np.nan,
            "n_vr_fit": 0,
        }

    x = df_diag["cos_lambda"].to_numpy(dtype=float)
    y = df_diag[rv_col].to_numpy(dtype=float)

    if weight_col in df_diag.columns:
        w = df_diag[weight_col].to_numpy(dtype=float)
    else:
        w = np.ones(len(df_diag), dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)

    if np.sum(valid) < 3:
        return {
            "rv_col_used": rv_col,
            "speed_from_vr_kms": np.nan,
            "speed_from_vr_rms_kms": np.nan,
            "n_vr_fit": int(np.sum(valid)),
        }

    x = x[valid]
    y = y[valid]
    w = w[valid]

    if np.sum(w) <= 0:
        w = np.ones(len(x), dtype=float)

    speed = np.sum(w * x * y) / np.sum(w * x * x)
    residual = y - speed * x
    rms = np.sqrt(np.average(residual**2, weights=w))

    return {
        "rv_col_used": rv_col,
        "speed_from_vr_kms": float(speed),
        "speed_from_vr_rms_kms": float(rms),
        "n_vr_fit": int(len(x)),
    }



def show_cluster_validation_products_icrs(
    result,
    bootstrap_n=500,
    bootstrap_seed=42,
    figsize=(14, 9),
):
    """
    Muestra en pantalla los cálculos y gráficas de validación de un cúmulo
    usando coordenadas ICRS/Gaia:

        Posiciones       : ra, dec
        Movimientos      : pmra, pmdec
        Ápex             : apex_ra_deg, apex_dec_deg
        Antápex          : antapex_ra_deg, antapex_dec_deg

    Esta función está pensada para usarse con la salida de:

        process_single_cluster(...)
        run_cluster_analysis_from_dataframe(..., cluster_id=...)
        run_cluster_analysis(..., cluster_id=...)

    Es decir, espera un diccionario con estructura:

        result["data_refined"]
        result["data_initial"]
        result["apex_refined"]
        result["apex_initial"]
        result["cross_refined"]
        result["cross_initial"]

    Parameters
    ----------
    result : dict
        Diccionario de salida del análisis de ápex para un cúmulo.

    bootstrap_n : int
        Número de remuestreos bootstrap para estimar incertidumbre del ápex.

    bootstrap_seed : int
        Semilla del bootstrap.

    figsize : tuple
        Tamaño base de las figuras.

    Returns
    -------
    summary_df : pd.DataFrame
        Tabla resumen de métricas globales.

    df_diag : pd.DataFrame
        Tabla estrella por estrella con diagnósticos.

    boot_df : pd.DataFrame
        Tabla con resultados bootstrap.
    """

    try:
        from IPython.display import display as _display
    except Exception:
        _display = print

    # ------------------------------------------------------------
    # Selección de datos
    # ------------------------------------------------------------

    if "data_refined" in result and result["data_refined"] is not None:
        df = result["data_initial"].copy()
    elif "data_initial" in result and result["data_initial"] is not None:
        df = result["data_initial"].copy()
    elif "df" in result and result["df"] is not None:
        # Compatibilidad con versiones anteriores
        df = result["df"].copy()
    else:
        raise ValueError(
            "result must contain result['data_refined'], "
            "result['data_initial'], or result['df']."
        )

    if len(df) == 0:
        raise ValueError("The selected dataframe is empty.")

    weight_col = result.get("weight_col_used", "probabilidad")

    # ------------------------------------------------------------
    # Selección del ápex
    # ------------------------------------------------------------

    apex = (
        result.get("apex_refined", None)
        or result.get("apex_initial", None)
        or result.get("apex_result", None)
    )

    if apex is None:
        raise ValueError(
            "Could not find apex_refined, apex_initial, or apex_result "
            "in result."
        )

    if "apex_vector" not in apex:
        raise ValueError("The selected apex dictionary lacks 'apex_vector'.")

    apex_vector = np.asarray(apex["apex_vector"], dtype=float)

    if not np.isfinite(apex_vector).all() or np.linalg.norm(apex_vector) == 0:
        raise ValueError("Invalid apex_vector.")

    apex_vector = apex_vector / np.linalg.norm(apex_vector)

    cross = (
        result.get("cross_refined", None)
        or result.get("cross_initial", None)
        or result.get("apex_from_pole_crosses_refined", None)
        or result.get("apex_from_pole_crosses", None)
    )

    # ------------------------------------------------------------
    # Cálculos principales
    # ------------------------------------------------------------

    df_diag = add_cluster_apex_diagnostics_icrs(
        df=df,
        apex_vector_icrs=apex_vector,
        weight_col=weight_col,
        min_pole_norm=0.0,
    )

    summary_df = compute_validation_summary_icrs(
        result=result,
        df_diag=df_diag,
    )

    # Asegurar columnas útiles aunque compute_validation_summary_icrs
    # no las haya creado explícitamente.
    if "fraction_kept" not in summary_df.columns:
        n_stars = result.get("n_stars", len(df_diag))
        n_stars_refined = result.get("n_stars_refined", len(df_diag))

        if n_stars is not None and n_stars > 0:
            summary_df.loc[0, "fraction_kept"] = n_stars_refined / n_stars
        else:
            summary_df.loc[0, "fraction_kept"] = np.nan

    if "min_sin_lambda" not in summary_df.columns:
        summary_df.loc[0, "min_sin_lambda"] = result.get(
            "min_sin_lambda",
            np.nan,
        )

    # ------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------

    if bootstrap_n is not None and bootstrap_n > 0:
        boot_df = bootstrap_apex_uncertainty_icrs(
            df_diag=df_diag,
            reference_apex_vector=apex_vector,
            n_bootstrap=bootstrap_n,
            random_seed=bootstrap_seed,
            weight_col="_weight_used",
        )
    else:
        boot_df = pd.DataFrame()

    if boot_df is not None and len(boot_df) > 0:
        summary_df.loc[0, "bootstrap_n"] = len(boot_df)

        summary_df.loc[0, "bootstrap_apex_delta_median_deg"] = float(
            boot_df["delta_from_reference_deg"].median()
        )
        summary_df.loc[0, "bootstrap_apex_delta_p16_deg"] = float(
            boot_df["delta_from_reference_deg"].quantile(0.16)
        )
        summary_df.loc[0, "bootstrap_apex_delta_p84_deg"] = float(
            boot_df["delta_from_reference_deg"].quantile(0.84)
        )

        if "apex_ra_deg" in boot_df.columns:
            summary_df.loc[0, "bootstrap_apex_ra_std_deg"] = float(
                boot_df["apex_ra_deg"].std()
            )

        if "apex_dec_deg" in boot_df.columns:
            summary_df.loc[0, "bootstrap_apex_dec_std_deg"] = float(
                boot_df["apex_dec_deg"].std()
            )
    else:
        summary_df.loc[0, "bootstrap_n"] = 0
        summary_df.loc[0, "bootstrap_apex_delta_median_deg"] = np.nan
        summary_df.loc[0, "bootstrap_apex_delta_p16_deg"] = np.nan
        summary_df.loc[0, "bootstrap_apex_delta_p84_deg"] = np.nan
        summary_df.loc[0, "bootstrap_apex_ra_std_deg"] = np.nan
        summary_df.loc[0, "bootstrap_apex_dec_std_deg"] = np.nan

    # ------------------------------------------------------------
    # Impresión del resumen principal
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("CLUSTER VALIDATION SUMMARY — ICRS / Gaia")
    print("=" * 72)

    important_cols = [
        "cluster_id",
        "n_stars",
        "n_stars_refined",
        "fraction_kept",

        "apex_ra_deg",
        "apex_dec_deg",
        "antapex_ra_deg",
        "antapex_dec_deg",

        "rms_pole_residual_deg",
        "median_pole_residual_deg",
        "toward_apex_fraction",

        "apex_cross_ra_deg",
        "apex_cross_dec_deg",
        "apex_cross_separation_deg",

        "fraction_pole_residual_lt_1deg",
        "fraction_pole_residual_lt_2deg",
        "fraction_motion_alignment_lt_1deg",
        "fraction_motion_alignment_lt_2deg",

        "speed_from_vt_kms",
        "speed_from_vt_rms_kms",

        "speed_from_vr_kms",
        "speed_from_vr_rms_kms",
        "rv_col_used",

        "bootstrap_n",
        "bootstrap_apex_delta_median_deg",
        "bootstrap_apex_delta_p16_deg",
        "bootstrap_apex_delta_p84_deg",
        "bootstrap_apex_ra_std_deg",
        "bootstrap_apex_dec_std_deg",
    ]

    available_cols = [
        col for col in important_cols
        if col in summary_df.columns
    ]

    _display(summary_df[available_cols].T.rename(columns={0: "value"}))

    print("\n" + "=" * 72)
    print("INTERPRETATION FLAGS")
    print("=" * 72)

    def _get_summary_value(col, default=np.nan):
        if col in summary_df.columns:
            return summary_df.loc[0, col]
        return default

    def _format_float(value, ndigits=4):
        if value is None or not np.isfinite(value):
            return "nan"
        return f"{value:.{ndigits}f}"

    rms = _get_summary_value("rms_pole_residual_deg")
    med = _get_summary_value("median_pole_residual_deg")
    toward = _get_summary_value("toward_apex_fraction")
    kept = _get_summary_value("fraction_kept")
    cross_sep = _get_summary_value("apex_cross_separation_deg")

    print(f"RMS pole residual          : {_format_float(rms)} deg")
    print(f"Median pole residual       : {_format_float(med)} deg")
    print(f"Toward-apex fraction       : {_format_float(toward)}")
    print(f"Fraction kept after filter : {_format_float(kept)}")
    print(f"Apex-cross separation      : {_format_float(cross_sep)} deg")

    if np.isfinite(rms) and np.isfinite(med):
        if rms < 1.0 and med < 1.0:
            print(
                "Pole residuals             : GOOD, "
                "strong common-motion geometry."
            )
        elif rms < 2.0:
            print(
                "Pole residuals             : ACCEPTABLE, "
                "but inspect outliers."
            )
        else:
            print(
                "Pole residuals             : LARGE, "
                "possible contamination or weak coherence."
            )
    else:
        print("Pole residuals             : NOT AVAILABLE.")

    if np.isfinite(toward):
        if toward > 0.9:
            print(
                "Motion orientation         : GOOD, "
                "most stars move toward the recovered apex."
            )
        else:
            print(
                "Motion orientation         : WARNING, "
                "apex orientation may be unstable."
            )
    else:
        print("Motion orientation         : NOT AVAILABLE.")

    if np.isfinite(cross_sep):
        if cross_sep < 1.0:
            print(
                "Apex estimator agreement   : GOOD, "
                "eigen and pole-cross estimators agree."
            )
        else:
            print(
                "Apex estimator agreement   : WARNING, "
                "estimators differ significantly."
            )
    else:
        print("Apex estimator agreement   : NOT AVAILABLE.")

    # ------------------------------------------------------------
    # Figura 1: distribución en el cielo y ápex en RA/Dec
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sc0 = axes[0].scatter(
        df_diag["ra"],
        df_diag["dec"],
        c=df_diag["parallax"],
        s=25,
        cmap="viridis",
        alpha=0.85,
        edgecolor="none",
    )

    axes[0].set_xlabel("RA [deg]")
    axes[0].set_ylabel("Dec [deg]")
    axes[0].set_title("Sky distribution")
    axes[0].grid(alpha=0.3)

    cbar0 = plt.colorbar(sc0, ax=axes[0])
    cbar0.set_label("Parallax [mas]")

    sc1 = axes[1].scatter(
        df_diag["ra"],
        df_diag["dec"],
        c=df_diag["pole_residual_deg"],
        s=25,
        cmap="magma_r",
        alpha=0.85,
        edgecolor="none",
        label="Members",
    )

    axes[1].scatter(
        apex["apex_ra_deg"],
        apex["apex_dec_deg"],
        marker="*",
        s=300,
        color="tab:blue",
        edgecolor="black",
        label="Apex",
        zorder=5,
    )

    axes[1].scatter(
        apex["antapex_ra_deg"],
        apex["antapex_dec_deg"],
        marker="X",
        s=140,
        color="tab:cyan",
        edgecolor="black",
        label="Antapex",
        zorder=5,
    )

    if cross is not None and "apex_ra_deg" in cross and "apex_dec_deg" in cross:
        axes[1].scatter(
            cross["apex_ra_deg"],
            cross["apex_dec_deg"],
            marker="P",
            s=170,
            color="tab:red",
            edgecolor="black",
            label="Pole-cross apex",
            zorder=5,
        )

    axes[1].set_xlabel("RA [deg]")
    axes[1].set_ylabel("Dec [deg]")
    axes[1].set_title("ICRS distribution and apex")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    cbar1 = plt.colorbar(sc1, ax=axes[1])
    cbar1.set_label("Pole residual [deg]")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Figura 2: movimientos propios y campo vectorial
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sc0 = axes[0].scatter(
        df_diag["pmra"],
        df_diag["pmdec"],
        c=df_diag["pole_residual_deg"],
        s=25,
        cmap="magma_r",
        alpha=0.85,
        edgecolor="none",
    )

    axes[0].set_xlabel(r"$\mu_{\alpha^\ast}$ [mas yr$^{-1}$]")
    axes[0].set_ylabel(r"$\mu_{\delta}$ [mas yr$^{-1}$]")
    axes[0].set_title("Vector point diagram")
    axes[0].grid(alpha=0.3)

    cbar0 = plt.colorbar(sc0, ax=axes[0])
    cbar0.set_label("Pole residual [deg]")

    sample = df_diag.copy()

    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)

    axes[1].scatter(
        df_diag["ra"],
        df_diag["dec"],
        s=10,
        color="lightgray",
        alpha=0.6,
    )

    qscale = np.nanpercentile(
        np.sqrt(sample["pmra"]**2 + sample["pmdec"]**2),
        90,
    )

    if not np.isfinite(qscale) or qscale <= 0:
        qscale = 1.0

    axes[1].quiver(
        sample["ra"],
        sample["dec"],
        sample["pmra"],
        sample["pmdec"],
        angles="xy",
        scale_units="xy",
        scale=qscale * 2.0,
        width=0.003,
        color="tab:blue",
        alpha=0.8,
    )

    axes[1].set_xlabel("RA [deg]")
    axes[1].set_ylabel("Dec [deg]")
    axes[1].set_title("Proper-motion vector field")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Figura 3: paralaje, distancia, residuos y lambda
    # ------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].hist(
        df_diag["parallax"].dropna(),
        bins=35,
        color="tab:blue",
        alpha=0.85,
    )
    axes[0, 0].set_xlabel("Parallax [mas]")
    axes[0, 0].set_ylabel("Number of stars")
    axes[0, 0].set_title("Parallax distribution")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(
        df_diag["distance_pc_from_parallax"].dropna(),
        bins=35,
        color="tab:green",
        alpha=0.85,
    )
    axes[0, 1].set_xlabel("Distance [pc]")
    axes[0, 1].set_ylabel("Number of stars")
    axes[0, 1].set_title("Distance distribution")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(
        df_diag["pole_residual_deg"].dropna(),
        bins=35,
        color="tab:purple",
        alpha=0.85,
    )
    axes[1, 0].axvline(1.0, color="black", ls="--", lw=1, label="1 deg")
    axes[1, 0].axvline(2.0, color="black", ls=":", lw=1, label="2 deg")
    axes[1, 0].set_xlabel("Pole residual [deg]")
    axes[1, 0].set_ylabel("Number of stars")
    axes[1, 0].set_title("Pole residual distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(
        df_diag["lambda_deg"].dropna(),
        bins=35,
        color="tab:orange",
        alpha=0.85,
    )
    axes[1, 1].set_xlabel(r"$\lambda$ [deg]")
    axes[1, 1].set_ylabel("Number of stars")
    axes[1, 1].set_title("Angular distance to apex")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Figura 4: velocidad tangencial vs sin(lambda)
    # ------------------------------------------------------------

    if "vt_kms" in df_diag.columns and "sin_lambda" in df_diag.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

        valid_vt = (
            np.isfinite(df_diag["vt_kms"])
            & np.isfinite(df_diag["sin_lambda"])
            & np.isfinite(df_diag["pole_residual_deg"])
        )

        sc0 = axes[0].scatter(
            df_diag.loc[valid_vt, "sin_lambda"],
            df_diag.loc[valid_vt, "vt_kms"],
            c=df_diag.loc[valid_vt, "pole_residual_deg"],
            s=25,
            cmap="magma_r",
            alpha=0.85,
            edgecolor="none",
        )

        speed_vt = _get_summary_value("speed_from_vt_kms")

        if np.isfinite(speed_vt):
            x_line = np.linspace(0, 1, 200)
            y_line = speed_vt * x_line

            axes[0].plot(
                x_line,
                y_line,
                color="black",
                lw=2,
                label=rf"$V={speed_vt:.2f}$ km s$^{{-1}}$",
            )

            axes[0].legend()

        axes[0].set_xlabel(r"$\sin(\lambda)$")
        axes[0].set_ylabel(r"$V_t$ [km s$^{-1}$]")
        axes[0].set_title(r"Tangential velocity: $V_t = V\sin(\lambda)$")
        axes[0].grid(alpha=0.3)

        cbar0 = plt.colorbar(sc0, ax=axes[0])
        cbar0.set_label("Pole residual [deg]")

        if np.isfinite(speed_vt):
            residual_vt = (
                df_diag.loc[valid_vt, "vt_kms"]
                - speed_vt * df_diag.loc[valid_vt, "sin_lambda"]
            )

            axes[1].scatter(
                df_diag.loc[valid_vt, "lambda_deg"],
                residual_vt,
                c=df_diag.loc[valid_vt, "pole_residual_deg"],
                s=25,
                cmap="magma_r",
                alpha=0.85,
                edgecolor="none",
            )

            axes[1].axhline(0.0, color="black", ls="--", lw=1)
            axes[1].set_ylabel(r"$V_t - V\sin(\lambda)$ [km s$^{-1}$]")
        else:
            axes[1].scatter(
                df_diag.loc[valid_vt, "lambda_deg"],
                df_diag.loc[valid_vt, "vt_kms"],
                c=df_diag.loc[valid_vt, "pole_residual_deg"],
                s=25,
                cmap="magma_r",
                alpha=0.85,
                edgecolor="none",
            )
            axes[1].set_ylabel(r"$V_t$ [km s$^{-1}$]")

        axes[1].set_xlabel(r"$\lambda$ [deg]")
        axes[1].set_title("Velocity residuals vs lambda")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Figura 5: alineación movimiento-ápex
    # ------------------------------------------------------------

    if "motion_apex_alignment_deg" in df_diag.columns:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

        axes[0].hist(
            df_diag["motion_apex_alignment_deg"].dropna(),
            bins=35,
            color="tab:red",
            alpha=0.85,
        )
        axes[0].axvline(1.0, color="black", ls="--", lw=1, label="1 deg")
        axes[0].axvline(2.0, color="black", ls=":", lw=1, label="2 deg")
        axes[0].set_xlabel("Motion-apex alignment [deg]")
        axes[0].set_ylabel("Number of stars")
        axes[0].set_title("Proper-motion alignment with apex direction")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        sc1 = axes[1].scatter(
            df_diag["lambda_deg"],
            df_diag["motion_apex_alignment_deg"],
            c=df_diag["pole_residual_deg"],
            s=25,
            cmap="magma_r",
            alpha=0.85,
            edgecolor="none",
        )
        axes[1].set_xlabel(r"$\lambda$ [deg]")
        axes[1].set_ylabel("Motion-apex alignment [deg]")
        axes[1].set_title("Alignment vs angular distance to apex")
        axes[1].grid(alpha=0.3)

        cbar1 = plt.colorbar(sc1, ax=axes[1])
        cbar1.set_label("Pole residual [deg]")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Figura 6: residuos del polo en el cielo
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sc0 = axes[0].scatter(
        df_diag["ra"],
        df_diag["dec"],
        c=df_diag["pole_residual_deg"],
        s=25,
        cmap="magma_r",
        alpha=0.85,
        edgecolor="none",
    )
    
    axes[0].set_xlabel("RA [deg]")
    axes[0].set_ylabel("Dec [deg]")
    axes[0].set_title("Pole residual across the sky")
    axes[0].grid(alpha=0.3)

    cbar0 = plt.colorbar(sc0, ax=axes[0])
    cbar0.set_label("Pole residual [deg]")

    sc1 = axes[1].scatter(
        df_diag["pmra"],
        df_diag["pmdec"],
        c=df_diag["motion_apex_alignment_deg"],
        s=25,
        cmap="viridis_r",
        alpha=0.85,
        edgecolor="none",
    )
    axes[1].set_xlabel(r"$\mu_{\alpha^\ast}$ [mas yr$^{-1}$]")
    axes[1].set_ylabel(r"$\mu_{\delta}$ [mas yr$^{-1}$]")
    axes[1].set_title("Alignment residual in vector point diagram")
    axes[1].grid(alpha=0.3)

    cbar1 = plt.colorbar(sc1, ax=axes[1])
    cbar1.set_label("Motion-apex alignment [deg]")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Figura 7: bootstrap
    # ------------------------------------------------------------

    if boot_df is not None and len(boot_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(
            boot_df["apex_ra_deg"],
            boot_df["apex_dec_deg"],
            s=8,
            alpha=0.35,
            color="tab:blue",
        )

        axes[0].scatter(
            apex["apex_ra_deg"],
            apex["apex_dec_deg"],
            marker="*",
            s=250,
            color="tab:red",
            edgecolor="black",
            label="Reference apex",
        )

        axes[0].set_xlabel(r"$\alpha_{\rm apex}$ [deg]")
        axes[0].set_ylabel(r"$\delta_{\rm apex}$ [deg]")
        axes[0].set_title("Bootstrap apex cloud")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].hist(
            boot_df["delta_from_reference_deg"],
            bins=35,
            color="tab:orange",
            alpha=0.85,
        )
        axes[1].set_xlabel("Angular separation from reference [deg]")
        axes[1].set_ylabel("Number of samples")
        axes[1].set_title("Bootstrap apex uncertainty")
        axes[1].grid(alpha=0.3)

        axes[2].hist(
            boot_df["rms_pole_residual_deg"],
            bins=35,
            color="tab:purple",
            alpha=0.85,
        )
        axes[2].set_xlabel("RMS pole residual [deg]")
        axes[2].set_ylabel("Number of samples")
        axes[2].set_title("Bootstrap residual stability")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Figura 8: CMD si hay fotometría Gaia
    # ------------------------------------------------------------

    g_col = None

    if "phot_g_mean_mag" in df_diag.columns:
        g_col = "phot_g_mean_mag"
    elif "G" in df_diag.columns:
        g_col = "G"

    color = None

    if "bp_rp" in df_diag.columns:
        color = df_diag["bp_rp"]
    elif (
        "phot_bp_mean_mag" in df_diag.columns
        and "phot_rp_mean_mag" in df_diag.columns
    ):
        color = (
            df_diag["phot_bp_mean_mag"]
            - df_diag["phot_rp_mean_mag"]
        )

    if g_col is not None and color is not None:
        valid_cmd = (
            np.isfinite(color)
            & np.isfinite(df_diag[g_col])
            & np.isfinite(df_diag["parallax"])
            & (df_diag["parallax"] > 0)
        )

        abs_g = (
            df_diag[g_col]
            - 5.0 * np.log10(1000.0 / df_diag["parallax"])
            + 5.0
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        sc0 = axes[0].scatter(
            color[valid_cmd],
            df_diag.loc[valid_cmd, g_col],
            c=df_diag.loc[valid_cmd, "pole_residual_deg"],
            s=25,
            cmap="magma_r",
            alpha=0.85,
            edgecolor="none",
        )

        axes[0].invert_yaxis()
        axes[0].set_xlabel(r"$G_{\rm BP}-G_{\rm RP}$")
        axes[0].set_ylabel(r"$G$")
        axes[0].set_title("Observed CMD")
        axes[0].grid(alpha=0.3)

        sc1 = axes[1].scatter(
            color[valid_cmd],
            abs_g[valid_cmd],
            c=df_diag.loc[valid_cmd, "pole_residual_deg"],
            s=25,
            cmap="magma_r",
            alpha=0.85,
            edgecolor="none",
        )

        axes[1].invert_yaxis()
        axes[1].set_xlabel(r"$G_{\rm BP}-G_{\rm RP}$")
        axes[1].set_ylabel(r"$M_G$")
        axes[1].set_title("Absolute CMD")
        axes[1].grid(alpha=0.3)

        cbar = plt.colorbar(sc1, ax=axes)
        cbar.set_label("Pole residual [deg]")

        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 72)
    print("Returned objects")
    print("=" * 72)
    print("summary_df : global validation metrics")
    print("df_diag    : star-by-star diagnostic table")
    print("boot_df    : bootstrap apex table")

    return summary_df, df_diag, boot_df

def plot_pm_vectors_on_sky(
    df,
    ra_col="ra",
    dec_col="dec",
    pmra_col="pmra",
    pmdec_col="pmdec",
    pmra_corr_col="pmra_corrected_corr",
    pmdec_corr_col="pmdec_corrected_corr",
    years=50000,
    max_stars=1000,
    random_state=42,
    figsize=(15, 6),
    title="Proper-motion vectors on the sky",
    save_path=None,
):
    """
    Grafica los vectores de movimiento propio observados y corregidos
    sobre el plano del cielo RA-Dec.

    Panel izquierdo:
        - Flechas grises: movimiento propio observado.
        - Flechas rojas: movimiento propio corregido.

    Panel derecho:
        - Flechas azules: diferencia entre movimiento corregido y observado.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con columnas Gaia originales y corregidas.

    ra_col, dec_col : str
        Columnas de ascensión recta y declinación en grados.

    pmra_col, pmdec_col : str
        Columnas de movimiento propio observado.
        En Gaia:
            pmra = mu_alpha* = mu_alpha cos(dec)
            pmdec = mu_delta

    pmra_corr_col, pmdec_corr_col : str
        Columnas de movimiento propio corregido.

    years : float
        Factor visual para convertir movimiento propio en desplazamiento angular.
        No representa una integración dinámica real; solo amplifica las flechas.

    max_stars : int
        Número máximo de estrellas a graficar.
        Si el DataFrame tiene más estrellas, se toma una submuestra aleatoria.

    save_path : str or None
        Si se entrega una ruta, guarda la figura.
    """

    required = [
        ra_col,
        dec_col,
        pmra_col,
        pmdec_col,
        pmra_corr_col,
        pmdec_corr_col,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {missing}")

    data = df.copy()

    # ------------------------------------------------------------
    # Selección de datos válidos
    # ------------------------------------------------------------
    mask = np.isfinite(data[ra_col])
    mask &= np.isfinite(data[dec_col])
    mask &= np.isfinite(data[pmra_col])
    mask &= np.isfinite(data[pmdec_col])
    mask &= np.isfinite(data[pmra_corr_col])
    mask &= np.isfinite(data[pmdec_corr_col])

    data = data.loc[mask].copy()

    if len(data) == 0:
        raise ValueError("No hay datos válidos para graficar.")

    # Submuestra para evitar figuras demasiado saturadas
    if len(data) > max_stars:
        data = data.sample(max_stars, random_state=random_state)

    # ------------------------------------------------------------
    # Conversión de movimiento propio a desplazamiento angular
    # ------------------------------------------------------------
    #
    # Gaia entrega:
    #     pmra = mu_alpha* = mu_alpha cos(dec)
    #
    # Para dibujar en el eje RA necesitamos:
    #     Delta_RA = mu_alpha * tiempo
    #              = pmra / cos(dec) * tiempo
    #
    # El factor 3.6e6 convierte mas a grados:
    #     1 degree = 3600 arcsec = 3.6e6 mas
    #
    # Estas flechas son solo una representación visual del vector tangencial.
    # ------------------------------------------------------------

    dec_rad = np.deg2rad(data[dec_col].to_numpy())
    cos_dec = np.cos(dec_rad)

    # Evitar divisiones problemáticas cerca de los polos
    cos_dec = np.where(np.abs(cos_dec) < 1e-6, np.nan, cos_dec)

    ra = data[ra_col].to_numpy()
    dec = data[dec_col].to_numpy()

    pmra = data[pmra_col].to_numpy()
    pmdec = data[pmdec_col].to_numpy()

    pmra_corr = data[pmra_corr_col].to_numpy()
    pmdec_corr = data[pmdec_corr_col].to_numpy()

    # Desplazamientos observados en grados
    dra_obs = (pmra / cos_dec) * years / 3.6e6
    ddec_obs = pmdec * years / 3.6e6

    # Desplazamientos corregidos en grados
    dra_corr = (pmra_corr / cos_dec) * years / 3.6e6
    ddec_corr = pmdec_corr * years / 3.6e6

    # Diferencia entre corregido y observado
    d_pmra = -1 * (pmra_corr - pmra)
    d_pmdec = -1 * (pmdec_corr - pmdec)

    dra_diff = (d_pmra / cos_dec) * years / 3.6e6
    ddec_diff = d_pmdec * years / 3.6e6

    # ------------------------------------------------------------
    # Figura
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1, ax2 = axes

    # ------------------------------------------------------------
    # Panel 1: vectores observados y corregidos
    # ------------------------------------------------------------
    ax1.scatter(
        ra,
        dec,
        s=8,
        color="black",
        alpha=0.35,
        label="Stars"
    )

    ax1.quiver(
        ra,
        dec,
        dra_obs,
        ddec_obs,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0025,
        alpha=0.55,
        color="gray",
        label="Observed PM"
    )

    ax1.quiver(
        ra,
        dec,
        dra_corr,
        ddec_corr,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0025,
        alpha=0.75,
        color="tab:red",
        label="Corrected PM"
    )

    ax1.set_xlabel("RA [deg]")
    ax1.set_ylabel("Dec [deg]")
    ax1.set_title("Observed vs corrected proper-motion vectors")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ------------------------------------------------------------
    # Panel 2: vector de diferencia
    # ------------------------------------------------------------
    ax2.scatter(
        ra,
        dec,
        s=8,
        color="black",
        alpha=0.35,
        label="Stars"
    )

    ax2.quiver(
        ra,
        dec,
        dra_diff,
        ddec_diff,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        alpha=0.8,
        color="tab:blue",
        label=r"$\Delta$PM = observed - corrected"
    )

    ax2.set_xlabel("RA [deg]")
    ax2.set_ylabel("Dec [deg]")
    ax2.set_title("Correction vectors on the sky")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(f"{title}  |  visual baseline = {years:g} yr", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_pm_correction_diagnostics(
    df,
    pmra_col="pmra",
    pmdec_col="pmdec",
    pmra_corr_col="pmra_corrected_corr",
    pmdec_corr_col="pmdec_corrected_corr",
    figsize=(14, 10),
    bins=40,
    title_prefix="Proper-motion correction diagnostics"
):
    """
    Grafica diagnósticos de la corrección de movimientos propios.

    Paneles:
        1. Histograma de las diferencias:
              Delta pmra  = pmra_corr  - pmra
              Delta pmdec = pmdec_corr - pmdec

        2. Plano de movimientos propios:
              original vs corregido

        3. Histograma de pmra:
              pmra original vs pmra corregido

        4. Histograma de pmdec:
              pmdec original vs pmdec corregido
    """

    required = [
        pmra_col,
        pmdec_col,
        pmra_corr_col,
        pmdec_corr_col,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {missing}")

    data = df.copy()

    # Diferencias entre movimiento propio corregido y observado.
    data["d_pmra"] = data[pmra_corr_col] - data[pmra_col]
    data["d_pmdec"] = data[pmdec_corr_col] - data[pmdec_col]

    # Selección de datos válidos.
    mask = np.isfinite(data[pmra_col])
    mask &= np.isfinite(data[pmdec_col])
    mask &= np.isfinite(data[pmra_corr_col])
    mask &= np.isfinite(data[pmdec_corr_col])
    mask &= np.isfinite(data["d_pmra"])
    mask &= np.isfinite(data["d_pmdec"])

    data = data.loc[mask].copy()

    if len(data) == 0:
        raise ValueError("No hay datos válidos para graficar.")

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax1, ax2, ax3, ax4 = axes.ravel()

    # ------------------------------------------------------------
    # 1. Histograma de las diferencias
    # ------------------------------------------------------------
    ax1.hist(
        data["d_pmra"],
        bins=bins,
        alpha=0.65,
        label=r"$\Delta \mu_{\alpha *}$",
        color="tab:blue"
    )

    ax1.hist(
        data["d_pmdec"],
        bins=bins,
        alpha=0.65,
        label=r"$\Delta \mu_{\delta}$",
        color="tab:orange"
    )

    ax1.axvline(0, color="black", linestyle="--", linewidth=1)

    ax1.set_xlabel(r"$\Delta \mu$ [mas yr$^{-1}$]")
    ax1.set_ylabel("Number of stars")
    ax1.set_title("Correction distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ------------------------------------------------------------
    # 2. Plano de movimientos propios: original vs corregido
    # ------------------------------------------------------------
    ax2.scatter(
        data[pmra_col],
        data[pmdec_col],
        s=10,
        alpha=0.35,
        label="Observed",
        color="gray"
    )

    ax2.scatter(
        data[pmra_corr_col],
        data[pmdec_corr_col],
        s=10,
        alpha=0.35,
        label="Corrected",
        color="tab:red"
    )

    ax2.set_xlabel(r"$\mu_{\alpha *}$ [mas yr$^{-1}$]")
    ax2.set_ylabel(r"$\mu_{\delta}$ [mas yr$^{-1}$]")
    ax2.set_title("Proper-motion plane")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ------------------------------------------------------------
    # 3. Histograma solo para movimiento propio en ascensión recta
    # ------------------------------------------------------------
    ax3.hist(
        data[pmra_col],
        bins=bins,
        alpha=0.6,
        label="Observed",
        color="gray"
    )

    ax3.hist(
        data[pmra_corr_col],
        bins=bins,
        alpha=0.6,
        label="Corrected",
        color="tab:blue"
    )

    ax3.axvline(
        np.nanmedian(data[pmra_col]),
        color="black",
        linestyle="--",
        linewidth=1,
        label="Observed median"
    )

    ax3.axvline(
        np.nanmedian(data[pmra_corr_col]),
        color="tab:blue",
        linestyle="--",
        linewidth=1,
        label="Corrected median"
    )

    ax3.set_xlabel(r"$\mu_{\alpha *}$ [mas yr$^{-1}$]")
    ax3.set_ylabel("Number of stars")
    ax3.set_title(r"Right ascension proper motion")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # ------------------------------------------------------------
    # 4. Histograma solo para movimiento propio en declinación
    # ------------------------------------------------------------
    ax4.hist(
        data[pmdec_col],
        bins=bins,
        alpha=0.6,
        label="Observed",
        color="gray"
    )

    ax4.hist(
        data[pmdec_corr_col],
        bins=bins,
        alpha=0.6,
        label="Corrected",
        color="tab:orange"
    )

    ax4.axvline(
        np.nanmedian(data[pmdec_col]),
        color="black",
        linestyle="--",
        linewidth=1,
        label="Observed median"
    )

    ax4.axvline(
        np.nanmedian(data[pmdec_corr_col]),
        color="tab:orange",
        linestyle="--",
        linewidth=1,
        label="Corrected median"
    )

    ax4.set_xlabel(r"$\mu_{\delta}$ [mas yr$^{-1}$]")
    ax4.set_ylabel("Number of stars")
    ax4.set_title(r"Declination proper motion")
    ax4.legend()
    ax4.grid(alpha=0.3)

    fig.suptitle(title_prefix, fontsize=15)
    fig.tight_layout()

    plt.show()

    # ------------------------------------------------------------
    # Resumen numérico
    # ------------------------------------------------------------
    print("Number of valid stars:", len(data))
    print()
    print("Median pmra observed   [mas/yr]:", np.nanmedian(data[pmra_col]))
    print("Median pmra corrected  [mas/yr]:", np.nanmedian(data[pmra_corr_col]))
    print("Median d_pmra          [mas/yr]:", np.nanmedian(data["d_pmra"]))
    print()
    print("Median pmdec observed  [mas/yr]:", np.nanmedian(data[pmdec_col]))
    print("Median pmdec corrected [mas/yr]:", np.nanmedian(data[pmdec_corr_col]))
    print("Median d_pmdec         [mas/yr]:", np.nanmedian(data["d_pmdec"]))
    print()
    print("Mean d_pmra            [mas/yr]:", np.nanmean(data["d_pmra"]))
    print("Mean d_pmdec           [mas/yr]:", np.nanmean(data["d_pmdec"]))

def correct_gaia_proper_motions(
    df: pd.DataFrame,
    ra_col: str = "ra",
    dec_col: str = "dec",
    parallax_col: str = "parallax",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    solar_motion_uvw_kms: tuple[float, float, float] = (11.1, 12.24, 7.25),
    correct_solar_reflex: bool = True,
    correct_oort_rotation: bool = False,
    suffix: str = "_corr",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Corrige movimientos propios de Gaia.

    Esta función toma las columnas astrométricas básicas de Gaia:

        ra        [grados]
        dec       [grados]
        parallax  [mas]
        pmra      [mas/yr] = mu_alpha* = mu_alpha cos(dec)
        pmdec     [mas/yr]

    y calcula movimientos propios corregidos por:

        1. Movimiento solar reflejo respecto al Local Standard of Rest, LSR.

    La corrección se hace estrella por estrella, porque la proyección del
    movimiento solar y de la rotación galáctica depende de la posición en el cielo
    y, en el caso del movimiento solar reflejo, también de la distancia.

    Parámetros
    ----------
    df : pandas.DataFrame
        Tabla con los datos Gaia.

    ra_col, dec_col : str
        Nombres de las columnas de ascensión recta y declinación en grados.

    parallax_col : str
        Nombre de la columna de paralaje en milisegundos de arco, mas.

    pmra_col, pmdec_col : str
        Nombres de las columnas de movimiento propio en ICRS.
        En Gaia, pmra ya incluye el factor cos(dec):
            pmra = mu_alpha* = mu_alpha cos(dec)

    solar_motion_uvw_kms : tuple
        Movimiento solar respecto al LSR en km/s.

        Convención galáctica estándar:
            U > 0 hacia el centro galáctico
            V > 0 en la dirección de rotación galáctica
            W > 0 hacia el polo norte galáctico

        Valor por defecto:
            (U, V, W) = (11.1, 12.24, 7.25) km/s

    correct_solar_reflex : bool
        Si True, resta el movimiento solar reflejo de los movimientos propios.

    suffix : str
        Sufijo para las columnas corregidas.
        Por defecto se crearán:
            pmra_corr
            pmdec_corr

    copy : bool
        Si True, no modifica el DataFrame original.
        Si False, añade las columnas directamente sobre df.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con columnas adicionales:

            parallax_corr
            distance_pc

            l_deg
            b_deg

            pm_l_cosb
            pm_b

            mu_l_solar_reflex
            mu_b_solar_reflex

            pm_l_cosb_corr
            pm_b_corr

            pmra_corr
            pmdec_corr

            d_pmra_corr
            d_pmdec_corr

            pm_corr_valid

    Uso típico
    ----------
    df_corr = correct_gaia_proper_motions(df)

    Luego usas:

        pmra_col  = "pmra_corr"
        pmdec_col = "pmdec_corr"

    """

    # ------------------------------------------------------------------
    # 1. Verificar que el DataFrame tiene las columnas necesarias
    # ------------------------------------------------------------------

    required = [ra_col, dec_col, parallax_col, pmra_col, pmdec_col]
    missing = set(required).difference(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    # Si copy=True, trabajamos sobre una copia para no alterar el DataFrame original.
    # Si copy=False, añadimos columnas directamente sobre df.
    result = df.copy() if copy else df

    # Convertimos columnas a arreglos numpy de tipo float.
    # Esto acelera las operaciones y evita problemas con tipos mixtos.
    ra = result[ra_col].to_numpy(dtype=float)
    dec = result[dec_col].to_numpy(dtype=float)
    parallax_obs = result[parallax_col].to_numpy(dtype=float)
    pmra = result[pmra_col].to_numpy(dtype=float)
    pmdec = result[pmdec_col].to_numpy(dtype=float)

    parallax_corr = parallax_obs

    # ------------------------------------------------------------------
    # 3. Máscara de estrellas válidas
    # ------------------------------------------------------------------
    #
    # Solo podemos corregir estrellas con:
    #   - ra, dec finitos
    #   - parallax_corr finita y positiva
    #   - pmra, pmdec finitos
    #
    # Si parallax <= 0, la distancia 1000/parallax no tiene sentido físico
    # en este esquema simple.
    # ------------------------------------------------------------------

    valid = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(parallax_corr)
        & np.isfinite(pmra)
        & np.isfinite(pmdec)
        & (parallax_corr > 0.0)
    )

    if valid.sum() == 0:
        raise ValueError("No hay estrellas válidas con parallax > 0 y PM finitos.")

    n = len(result)

    # ------------------------------------------------------------------
    # 4. Crear arreglos de salida llenos inicialmente con NaN
    # ------------------------------------------------------------------
    #
    # Para las estrellas no válidas se dejarán NaN.
    # Para las válidas se rellenarán los valores corregidos.
    # ------------------------------------------------------------------

    distance_pc = np.full(n, np.nan)

    l_deg = np.full(n, np.nan)
    b_deg = np.full(n, np.nan)

    pm_l_cosb = np.full(n, np.nan)
    pm_b = np.full(n, np.nan)

    pm_l_cosb_corr = np.full(n, np.nan)
    pm_b_corr = np.full(n, np.nan)

    pmra_corr = np.full(n, np.nan)
    pmdec_corr = np.full(n, np.nan)

    mu_l_solar = np.full(n, np.nan)
    mu_b_solar = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # 5. Calcular distancia en pc
    # ------------------------------------------------------------------

    distance_pc_valid = 1000.0 / parallax_corr[valid]
    distance_pc[valid] = distance_pc_valid

    # ------------------------------------------------------------------
    # 6. Crear objeto SkyCoord en coordenadas ICRS
    # ------------------------------------------------------------------
    #
    # Gaia entrega:
    #     ra, dec, pmra, pmdec
    #
    # en el sistema ecuatorial ICRS.

    c_icrs = SkyCoord(
        ra=ra[valid] * u.deg,
        dec=dec[valid] * u.deg,
        distance=distance_pc_valid * u.pc,
        pm_ra_cosdec=pmra[valid] * u.mas / u.yr,
        pm_dec=pmdec[valid] * u.mas / u.yr,
        radial_velocity=np.zeros(valid.sum()) * u.km / u.s,
        frame="icrs",
    )

    # ------------------------------------------------------------------
    # 7. Transformar a coordenadas galácticas
    # ------------------------------------------------------------------
    #
    # En coordenadas galácticas es más natural aplicar:
    #   - corrección por movimiento solar reflejo
    #   - corrección por rotación galáctica diferencial
    #
    # Astropy devuelve:
    #     pm_l_cosb = mu_l* = mu_l cos(b)
    #     pm_b      = mu_b
    #
    # en mas/yr.
    # ------------------------------------------------------------------

    c_gal = c_icrs.galactic

    l = c_gal.l.to_value(u.rad)
    b = c_gal.b.to_value(u.rad)

    l_deg[valid] = c_gal.l.to_value(u.deg)
    b_deg[valid] = c_gal.b.to_value(u.deg)

    pm_l_cosb_valid = c_gal.pm_l_cosb.to_value(u.mas / u.yr)
    pm_b_valid = c_gal.pm_b.to_value(u.mas / u.yr)

    pm_l_cosb[valid] = pm_l_cosb_valid
    pm_b[valid] = pm_b_valid

    # Estas son las variables que iremos corrigiendo.
    # Empezamos desde los movimientos propios observados en coordenadas galácticas.
    pm_l_new = pm_l_cosb_valid.copy()
    pm_b_new = pm_b_valid.copy()

    # ------------------------------------------------------------------
    # 8. Corrección por movimiento solar reflejo
    # ------------------------------------------------------------------
    #
    # El Sol se mueve respecto al LSR con velocidad (U, V, W).
    # Esa velocidad produce un movimiento aparente reflejo en las estrellas.
    #
    # Para una estrella que estuviera en reposo respecto al LSR, nosotros
    # observaríamos un movimiento propio aparente causado por el movimiento
    # del Sol.
    #
    # En coordenadas galácticas, la contribución reflejo es:
    #
    #   mu_l_solar =
    #       parallax / KAPPA * ( U sin(l) - V cos(l) )
    #
    #   mu_b_solar =
    #       parallax / KAPPA *
    #       ( U cos(l) sin(b) + V sin(l) sin(b) - W cos(b) )
    #
    # donde:
    #     parallax está en mas
    #     U,V,W están en km/s
    #     mu queda en mas/yr
    #
    # Para obtener el movimiento propio corregido al LSR, restamos esta
    # contribución:
    #
    #     mu_corr = mu_obs - mu_solar_reflex
    #
    # ------------------------------------------------------------------

    if correct_solar_reflex:
        U, V, W = solar_motion_uvw_kms

        mu_l_solar_valid = (
            parallax_corr[valid]
            / KM_S_PER_AU_YR
            * (U * np.sin(l) - V * np.cos(l))
        )

        mu_b_solar_valid = (
            parallax_corr[valid]
            / KM_S_PER_AU_YR
            * (
                U * np.cos(l) * np.sin(b)
                + V * np.sin(l) * np.sin(b)
                - W * np.cos(b)
            )
        )

        # Restamos el movimiento solar reflejo.
        pm_l_new = pm_l_new - mu_l_solar_valid
        pm_b_new = pm_b_new - mu_b_solar_valid

        # Guardamos las correcciones aplicadas para inspección posterior.
        mu_l_solar[valid] = mu_l_solar_valid
        mu_b_solar[valid] = mu_b_solar_valid

    # Guardamos movimientos propios corregidos en coordenadas galácticas.
    pm_l_cosb_corr[valid] = pm_l_new
    pm_b_corr[valid] = pm_b_new

    # ------------------------------------------------------------------
    # 10. Transformar los movimientos propios corregidos de vuelta a ICRS
    # ------------------------------------------------------------------
    #
    # El resultado final será:
    #     pmra_corr
    #     pmdec_corr
    #
    # con las mismas unidades y convención que Gaia:
    #     pmra_corr = mu_alpha* = mu_alpha cos(dec)
    # ------------------------------------------------------------------

    c_gal_corr = SkyCoord(
        l=c_gal.l,
        b=c_gal.b,
        distance=distance_pc_valid * u.pc,
        pm_l_cosb=pm_l_new * u.mas / u.yr,
        pm_b=pm_b_new * u.mas / u.yr,
        radial_velocity=np.zeros(valid.sum()) * u.km / u.s,
        frame="galactic",
    )

    c_icrs_corr = c_gal_corr.icrs

    pmra_corr[valid] = c_icrs_corr.pm_ra_cosdec.to_value(u.mas / u.yr)
    pmdec_corr[valid] = c_icrs_corr.pm_dec.to_value(u.mas / u.yr)

    # ------------------------------------------------------------------
    # 11. Añadir columnas al DataFrame
    # ------------------------------------------------------------------
    #
    # Además de las columnas finales corregidas, guardamos columnas
    # intermedias útiles para depurar, graficar y entender qué tan grande
    # fue cada corrección.
    # ------------------------------------------------------------------

    result["parallax_corr"] = parallax_corr
    result["distance_pc"] = distance_pc

    result["l_deg"] = l_deg
    result["b_deg"] = b_deg

    # Movimientos propios originales, pero expresados en coordenadas galácticas.
    result["pm_l_cosb"] = pm_l_cosb
    result["pm_b"] = pm_b

    # Contribución del movimiento solar reflejo.
    result["mu_l_solar_reflex"] = mu_l_solar
    result["mu_b_solar_reflex"] = mu_b_solar
    
    # Movimientos propios corregidos en coordenadas galácticas.
    result["pm_l_cosb_corr"] = pm_l_cosb_corr
    result["pm_b_corr"] = pm_b_corr

    # Movimientos propios corregidos en coordenadas ecuatoriales ICRS.
    # Estas son las columnas principales para tu código de apex/antapex.
    result[f"{pmra_col}{suffix}"] = pmra_corr
    result[f"{pmdec_col}{suffix}"] = pmdec_corr

    # Diferencia entre el movimiento propio corregido y el observado.
    # Sirve para ver cuánto cambió cada estrella.
    result[f"d_{pmra_col}{suffix}"] = result[f"{pmra_col}{suffix}"] - result[pmra_col]
    result[f"d_{pmdec_col}{suffix}"] = result[f"{pmdec_col}{suffix}"] - result[pmdec_col]

    # Bandera booleana:
    # True  -> la estrella tenía datos válidos y fue corregida.
    # False -> la estrella tenía datos inválidos y queda con NaN en las columnas corregidas.
    result["pm_corr_valid"] = valid

    return result

def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Input vector has zero or invalid norm.")

    return vector / norm


def radec_to_unit_vector(
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


def unit_vector_to_radec(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte uno o varios vectores unitarios a RA, Dec en grados.
    """
    vectors = np.asarray(vectors, dtype=float)

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, 3)

    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & (norms > 0)

    ra_deg = np.full(len(vectors), np.nan)
    dec_deg = np.full(len(vectors), np.nan)

    v = vectors[valid] / norms[valid, None]

    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]

    ra_rad = np.arctan2(y, x) % (2.0 * np.pi)
    dec_rad = np.arcsin(np.clip(z, -1.0, 1.0))

    ra_deg[valid] = np.rad2deg(ra_rad)
    dec_deg[valid] = np.rad2deg(dec_rad)

    return ra_deg, dec_deg

def build_apex_parallel_radec(
    apex_ra_deg: float,
    apex_dec_deg: float,
    angular_distance_deg: float,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye un paralelo del sistema asociado al ápex.

    El paralelo es el conjunto de puntos separados del ápex por una
    distancia angular constante.

    angular_distance_deg:
        0 deg   -> ápex
        90 deg  -> ecuador del ápex
        180 deg -> antápex
    """
    apex_vector, e1, e2 = build_apex_reference_frame(
        apex_ra_deg=apex_ra_deg,
        apex_dec_deg=apex_dec_deg,
    )

    theta = np.deg2rad(angular_distance_deg)
    phi = np.linspace(0.0, 2.0 * np.pi, n_points)

    circle_xyz = (
        np.cos(theta) * apex_vector[None, :]
        + np.sin(theta)
        * (
            np.cos(phi)[:, None] * e1[None, :]
            + np.sin(phi)[:, None] * e2[None, :]
        )
    )

    parallel_ra_deg, parallel_dec_deg = unit_vector_to_radec(circle_xyz)

    return parallel_ra_deg, parallel_dec_deg

def build_apex_equator_radec(
    apex_ra_deg: float,
    apex_dec_deg: float,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye el ecuador asociado al ápex.

    Este ecuador es el círculo máximo perpendicular al vector del ápex.
    No es el ecuador celeste físico, sino el círculo donde deberían caer
    los polos ideales de estrellas que se mueven hacia ese ápex.
    """
    apex_vector = radec_to_unit_vector(apex_ra_deg, apex_dec_deg)
    apex_vector = _normalize_vector(apex_vector)

    # Vector auxiliar que no sea casi paralelo al ápex
    aux = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(aux, apex_vector)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])

    e1 = _normalize_vector(np.cross(apex_vector, aux))
    e2 = _normalize_vector(np.cross(apex_vector, e1))

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)

    circle_xyz = (
        np.cos(theta)[:, None] * e1[None, :]
        + np.sin(theta)[:, None] * e2[None, :]
    )

    equator_ra_deg, equator_dec_deg = unit_vector_to_radec(circle_xyz)

    return equator_ra_deg, equator_dec_deg


def plot_radec_curve_no_wrap(
    ax,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    label: Optional[str] = None,
    **plot_kwargs,
):
    """
    Grafica una curva RA-Dec evitando líneas artificiales cuando la curva
    cruza RA = 0/360 deg.
    """
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)

    valid = np.isfinite(ra_deg) & np.isfinite(dec_deg)
    ra_deg = ra_deg[valid]
    dec_deg = dec_deg[valid]

    if len(ra_deg) < 2:
        return

    jumps = np.where(np.abs(np.diff(ra_deg)) > 180.0)[0] + 1

    ra_segments = np.split(ra_deg, jumps)
    dec_segments = np.split(dec_deg, jumps)

    for i, (ra_seg, dec_seg) in enumerate(zip(ra_segments, dec_segments)):
        if len(ra_seg) < 2:
            continue

        segment_label = label if i == 0 else "_nolegend_"

        ax.plot(
            ra_seg,
            dec_seg,
            label=segment_label,
            **plot_kwargs,
        )

def build_apex_reference_frame(
    apex_ra_deg: float,
    apex_dec_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye una base ortonormal asociada al ápex.

    apex_vector: eje polar del sistema.
    e1, e2: base ortonormal en el plano perpendicular al ápex.
    """
    apex_vector = radec_to_unit_vector(apex_ra_deg, apex_dec_deg)
    apex_vector = _normalize_vector(apex_vector)

    aux = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(aux, apex_vector)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])

    e1 = _normalize_vector(np.cross(apex_vector, aux))
    e2 = _normalize_vector(np.cross(apex_vector, e1))

    return apex_vector, e1, e2


def build_apex_equator_radec(
    apex_ra_deg: float,
    apex_dec_deg: float,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye el ecuador asociado al ápex.

    Este ecuador es el círculo máximo perpendicular al vector del ápex.
    """
    apex_vector, e1, e2 = build_apex_reference_frame(
        apex_ra_deg=apex_ra_deg,
        apex_dec_deg=apex_dec_deg,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)

    circle_xyz = (
        np.cos(theta)[:, None] * e1[None, :]
        + np.sin(theta)[:, None] * e2[None, :]
    )

    equator_ra_deg, equator_dec_deg = unit_vector_to_radec(circle_xyz)

    return equator_ra_deg, equator_dec_deg


def build_apex_meridian_radec(
    apex_ra_deg: float,
    apex_dec_deg: float,
    meridian_phase_deg: float,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye un meridiano del sistema asociado al ápex.

    Cada meridiano es un círculo máximo que pasa por:
        - ápex,
        - antápex,
        - dos puntos del ecuador del ápex.
    """
    apex_vector, e1, e2 = build_apex_reference_frame(
        apex_ra_deg=apex_ra_deg,
        apex_dec_deg=apex_dec_deg,
    )

    phi = np.deg2rad(meridian_phase_deg)

    equator_direction = (
        np.cos(phi) * e1
        + np.sin(phi) * e2
    )
    equator_direction = _normalize_vector(equator_direction)

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)

    meridian_xyz = (
        np.cos(theta)[:, None] * apex_vector[None, :]
        + np.sin(theta)[:, None] * equator_direction[None, :]
    )

    meridian_ra_deg, meridian_dec_deg = unit_vector_to_radec(meridian_xyz)

    return meridian_ra_deg, meridian_dec_deg


def add_apex_equator_to_axes(
    ax,
    *,
    apex_ra_deg: float,
    apex_dec_deg: float,
    show_apex: bool = True,
    show_antapex: bool = False,

    show_meridians: bool = True,
    n_meridians: int = 4,
    meridian_phases_deg: Optional[list[float]] = None,

    show_parallels: bool = True,
    parallel_distances_deg: Optional[list[float]] = None,

    n_points: int = 1000,

    equator_color: str = "white",
    equator_lw: float = 2.2,
    equator_ls: str = "--",
    equator_alpha: float = 0.95,

    meridian_color: str = "white",
    meridian_lw: float = 1.3,
    meridian_ls: str = ":",
    meridian_alpha: float = 0.65,

    parallel_color: str = "white",
    parallel_lw: float = 1.1,
    parallel_ls: str = "-.",
    parallel_alpha: float = 0.50,

    apex_color: str = "crimson",
    antapex_color: str = "purple",
):
    """
    Agrega una grilla angular asociada al ápex:
        - ecuador del ápex,
        - meridianos,
        - paralelos,
        - ápex y antápex opcionales.
    """

    # Ecuador del ápex
    equator_ra, equator_dec = build_apex_equator_radec(
        apex_ra_deg=apex_ra_deg,
        apex_dec_deg=apex_dec_deg,
        n_points=n_points,
    )

    plot_radec_curve_no_wrap(
        ax,
        equator_ra,
        equator_dec,
        label="Apex equator",
        color=equator_color,
        linewidth=equator_lw,
        linestyle=equator_ls,
        alpha=equator_alpha,
        zorder=8,
    )

    # Meridianos: círculos máximos que pasan por ápex y antápex
    if show_meridians:
        if meridian_phases_deg is None:
            meridian_phases_deg = np.linspace(
                0.0,
                180.0,
                n_meridians,
                endpoint=False,
            )

        for i, phase_deg in enumerate(meridian_phases_deg):
            meridian_ra, meridian_dec = build_apex_meridian_radec(
                apex_ra_deg=apex_ra_deg,
                apex_dec_deg=apex_dec_deg,
                meridian_phase_deg=phase_deg,
                n_points=n_points,
            )

            plot_radec_curve_no_wrap(
                ax,
                meridian_ra,
                meridian_dec,
                label="Apex meridians" if i == 0 else "_nolegend_",
                color=meridian_color,
                linewidth=meridian_lw,
                linestyle=meridian_ls,
                alpha=meridian_alpha,
                zorder=7,
            )

    # Paralelos: círculos de distancia angular constante al ápex
    if show_parallels:
        if parallel_distances_deg is None:
            parallel_distances_deg = [30.0, 60.0, 120.0, 150.0]

        for i, distance_deg in enumerate(parallel_distances_deg):
            if np.isclose(distance_deg, 0.0) or np.isclose(distance_deg, 180.0):
                continue

            # Si se incluye 90 deg, sería el ecuador, que ya se dibujó.
            if np.isclose(distance_deg, 90.0):
                continue

            parallel_ra, parallel_dec = build_apex_parallel_radec(
                apex_ra_deg=apex_ra_deg,
                apex_dec_deg=apex_dec_deg,
                angular_distance_deg=distance_deg,
                n_points=n_points,
            )

            plot_radec_curve_no_wrap(
                ax,
                parallel_ra,
                parallel_dec,
                label="Apex parallels" if i == 0 else "_nolegend_",
                color=parallel_color,
                linewidth=parallel_lw,
                linestyle=parallel_ls,
                alpha=parallel_alpha,
                zorder=6,
            )

    if show_apex:
        ax.scatter(
            [apex_ra_deg],
            [apex_dec_deg],
            s=170,
            marker="*",
            color=apex_color,
            edgecolor="black",
            linewidth=0.8,
            label="Apex",
            zorder=10,
        )

    if show_antapex:
        antapex_ra_deg = (apex_ra_deg + 180.0) % 360.0
        antapex_dec_deg = -apex_dec_deg

        ax.scatter(
            [antapex_ra_deg],
            [antapex_dec_deg],
            s=120,
            marker="X",
            color=antapex_color,
            edgecolor="black",
            linewidth=0.8,
            label="Antapex",
            zorder=10,
        )

def plot_apex_error_grid(
    grid_summary: pd.DataFrame,
    value_col: str = "metric_mean",
    title: str = "Apex residual RMS across sky positions",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (13, 5),
    apex_ra_deg: Optional[float] = None,
    apex_dec_deg: Optional[float] = None,
    show_apex_equator: bool = True,
    show_apex: bool = True,
    show_antapex: bool = False,
    show_meridians: bool = True,
    n_meridians: int = 4,
    show_parallels: bool = True,
    parallel_distances_deg: Optional[list[float]] = None,
):
    """
    Grafica una malla RA-Dec de la métrica de error.

    El eje X es RA y el eje Y es Dec.

    Si se pasan apex_ra_deg y apex_dec_deg, también se grafica el ecuador
    correspondiente al ápex.
    """

    pivot = grid_summary.pivot(
        index="center_dec_deg",
        columns="center_ra_deg",
        values=value_col,
    )

    ra_values = pivot.columns.to_numpy(dtype=float)
    dec_values = pivot.index.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            ra_values.min(),
            ra_values.max(),
            dec_values.min(),
            dec_values.max(),
        ],
        cmap=cmap,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    if (
        show_apex_equator
        and apex_ra_deg is not None
        and apex_dec_deg is not None
    ):
        add_apex_equator_to_axes(
            ax,
            apex_ra_deg=apex_ra_deg,
            apex_dec_deg=apex_dec_deg,
            show_apex=show_apex,
            show_antapex=show_antapex,

            show_meridians=show_meridians,
            n_meridians=n_meridians,

            show_parallels=show_parallels,
            parallel_distances_deg=parallel_distances_deg,

            equator_color="white",
            equator_lw=2.4,
            equator_ls="--",
            equator_alpha=0.95,

            meridian_color="white",
            meridian_lw=1.2,
            meridian_ls=":",
            meridian_alpha=0.55,

            parallel_color="white",
            parallel_lw=1.1,
            parallel_ls="-.",
            parallel_alpha=0.45,
        )

    ax.set_xlabel("Cluster centre RA [deg]")
    ax.set_ylabel("Cluster centre Dec [deg]")
    ax.set_title(title)

    ax.set_xlim(ra_values.min(), ra_values.max())
    ax.set_ylim(dec_values.min(), dec_values.max())

    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(dec_values.min(), dec_values.max() + 1, 20))

    if apex_ra_deg is not None and apex_dec_deg is not None:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_apex_error_scatter(
    grid_summary: pd.DataFrame,
    value_col: str = "metric_mean",
    title: str = "Apex residual RMS across sky positions",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (12, 5),
    apex_ra_deg: Optional[float] = None,
    apex_dec_deg: Optional[float] = None,
    show_apex_equator: bool = True,
    show_apex: bool = True,
    show_antapex: bool = False,
):
    """
    Alternativa en scatter, útil si hay posiciones fallidas o mallas irregulares.

    Si se pasan apex_ra_deg y apex_dec_deg, también se grafica el ecuador
    correspondiente al ápex.
    """

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        grid_summary["center_ra_deg"],
        grid_summary["center_dec_deg"],
        c=grid_summary[value_col],
        s=80,
        cmap=cmap,
        edgecolor="k",
        linewidth=0.2,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(value_col)

    if (
        show_apex_equator
        and apex_ra_deg is not None
        and apex_dec_deg is not None
    ):
        add_apex_equator_to_axes(
            ax,
            apex_ra_deg=apex_ra_deg,
            apex_dec_deg=apex_dec_deg,
            show_apex=show_apex,
            show_antapex=show_antapex,
            equator_color="black",
            equator_lw=2.2,
            equator_ls="--",
            equator_alpha=0.9,
        )

    ax.set_xlabel("Cluster centre RA [deg]")
    ax.set_ylabel("Cluster centre Dec [deg]")
    ax.set_title(title)

    ax.set_xlim(0, 360)
    ax.set_ylim(
        grid_summary["center_dec_deg"].min() - 5,
        grid_summary["center_dec_deg"].max() + 5,
    )

    ax.set_xticks(np.arange(0, 361, 30))

    if apex_ra_deg is not None and apex_dec_deg is not None:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    return fig, ax

@dataclass
class ApexResult:
    """Resultados cinemáticos del cálculo del ápex."""

    n_stars: int

    # Ápex y antápex en ICRS
    apex_ra_deg: float
    apex_dec_deg: float
    antapex_ra_deg: float
    antapex_dec_deg: float

    # Ápex en coordenadas galácticas
    apex_l_deg: float
    apex_b_deg: float

    # Velocidad media heliocéntrica cartesiana ICRS
    vx_mean_kms: float
    vy_mean_kms: float
    vz_mean_kms: float
    speed_mean_kms: float

    # Dispersiones cinemáticas
    sigma_vx_rms_kms: float
    sigma_vy_rms_kms: float
    sigma_vz_rms_kms: float
    sigma_3d_rms_kms: float
    sigma_1d_rms_kms: float

    # Dispersión en dirección del movimiento
    sigma_apex_rms_deg: float

    # Incertidumbres bootstrap del ápex
    apex_ra_bootstrap_std_deg: float
    apex_dec_bootstrap_std_deg: float
    apex_angular_bootstrap_rms_deg: float

    # Tabla con resultados individuales
    stars: pd.DataFrame


def calcular_apex_3d_gaia(
    data: pd.DataFrame,
    ra_col: str = "ra",
    dec_col: str = "dec",
    parallax_col: str = "parallax",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    rv_col: str = "radial_velocity",
    weight_col: Optional[str] = None,
    source_id_col: Optional[str] = "source_id",
    min_parallax: float = 0.0,
    sigma_clip: Optional[float] = 3.0,
    max_clip_iterations: int = 5,
    bootstrap_samples: int = 1000,
    random_state: Optional[int] = 42,
) -> ApexResult:
    """
    Calcula el ápex de un cúmulo usando astrometría y velocidad radial de Gaia.

    Parámetros
    ----------
    data
        DataFrame que contiene ra, dec, parallax, pmra, pmdec y
        radial_velocity.

    ra_col, dec_col
        Ascensión recta y declinación en grados.

    parallax_col
        Paralaje en milisegundos de arco.

    pmra_col, pmdec_col
        Movimientos propios en mas/año. Se asume que ``pmra`` corresponde
        a mu_alpha* = mu_alpha cos(dec), como en Gaia.

    rv_col
        Velocidad radial heliocéntrica en km/s.

    weight_col
        Columna opcional de pesos. Puede contener probabilidades de
        pertenencia o pesos derivados de incertidumbres.

    source_id_col
        Columna identificadora que será conservada en la salida.

    min_parallax
        Paralaje mínima permitida en mas.

    sigma_clip
        Umbral opcional para eliminar estrellas cinemáticamente discrepantes.
        El clipping se aplica sobre la norma del residual 3D. Use None para
        desactivarlo.

    max_clip_iterations
        Máximo número de iteraciones de sigma clipping.

    bootstrap_samples
        Número de remuestreos bootstrap usados para estimar la incertidumbre
        del ápex. Use 0 para desactivarlo.

    random_state
        Semilla del generador aleatorio.

    Retorna
    -------
    ApexResult
        Resultados globales y tabla de resultados por estrella.

    Notas
    -----
    La dispersión se calcula sobre los residuales

        delta_v_i = v_i - <v>,

    imponiendo media residual igual a cero. Por tanto,

        sigma_RMS = sqrt(sum(w_i * delta_i**2) / sum(w_i)).

    No se vuelve a estimar o sustraer la media de los residuales dentro de la
    expresión de la dispersión.
    """

    required = [
        ra_col,
        dec_col,
        parallax_col,
        pmra_col,
        pmdec_col,
        rv_col,
    ]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(
            "Faltan las siguientes columnas obligatorias: "
            + ", ".join(missing)
        )

    if weight_col is not None and weight_col not in data.columns:
        raise KeyError(f"No existe la columna de pesos {weight_col!r}.")

    columns = required.copy()

    if weight_col is not None:
        columns.append(weight_col)

    if source_id_col is not None and source_id_col in data.columns:
        columns.append(source_id_col)

    stars = data.loc[:, list(dict.fromkeys(columns))].copy()

    # Conversión segura a valores numéricos.
    for column in required:
        stars[column] = pd.to_numeric(stars[column], errors="coerce")

    if weight_col is not None:
        stars[weight_col] = pd.to_numeric(
            stars[weight_col], errors="coerce"
        )

    valid = np.ones(len(stars), dtype=bool)

    for column in required:
        valid &= np.isfinite(stars[column].to_numpy(dtype=float))

    valid &= stars[parallax_col].to_numpy(dtype=float) > min_parallax

    if weight_col is not None:
        weights_input = stars[weight_col].to_numpy(dtype=float)
        valid &= np.isfinite(weights_input)
        valid &= weights_input > 0.0

    stars = stars.loc[valid].reset_index(drop=True)

    if len(stars) < 3:
        raise ValueError(
            "Se necesitan al menos tres estrellas con astrometría, "
            "paralaje y velocidad radial válidas."
        )

    parallax = stars[parallax_col].to_numpy(dtype=float)

    # Para paralajes de buena señal/ruido:
    # distancia [pc] = 1000 / parallax [mas].
    distance_pc = 1000.0 / parallax

    coord = SkyCoord(
        ra=stars[ra_col].to_numpy(dtype=float) * u.deg,
        dec=stars[dec_col].to_numpy(dtype=float) * u.deg,
        distance=distance_pc * u.pc,
        pm_ra_cosdec=stars[pmra_col].to_numpy(dtype=float)
        * u.mas
        / u.yr,
        pm_dec=stars[pmdec_col].to_numpy(dtype=float)
        * u.mas
        / u.yr,
        radial_velocity=stars[rv_col].to_numpy(dtype=float)
        * u.km
        / u.s,
        frame=ICRS(),
    )

    velocity = coord.velocity.d_xyz.to_value(u.km / u.s).T

    if weight_col is None:
        weights = np.ones(len(stars), dtype=float)
    else:
        weights = stars[weight_col].to_numpy(dtype=float)

    def weighted_mean_vector(
        vectors: np.ndarray,
        vector_weights: np.ndarray,
    ) -> np.ndarray:
        return np.average(vectors, axis=0, weights=vector_weights)

    def rms_about_zero(
        values: np.ndarray,
        value_weights: np.ndarray,
    ) -> float:
        """
        RMS ponderado respecto a cero.

        No sustrae la media de ``values``.
        """
        return float(
            np.sqrt(
                np.average(
                    np.asarray(values, dtype=float) ** 2,
                    weights=value_weights,
                )
            )
        )

    def vector_to_icrs(vector: np.ndarray) -> SkyCoord:
        norm = np.linalg.norm(vector)

        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                "El vector de velocidad media tiene módulo nulo o inválido."
            )

        unit_vector = vector / norm

        return SkyCoord(
            CartesianRepresentation(
                unit_vector[0] * u.one,
                unit_vector[1] * u.one,
                unit_vector[2] * u.one,
            ),
            frame=ICRS(),
        )

    # ------------------------------------------------------------------
    # Clipping cinemático iterativo
    # ------------------------------------------------------------------
    selected = np.ones(len(stars), dtype=bool)

    if sigma_clip is not None:
        if sigma_clip <= 0:
            raise ValueError("sigma_clip debe ser positivo o None.")

        for _ in range(max_clip_iterations):
            mean_velocity = weighted_mean_vector(
                velocity[selected],
                weights[selected],
            )

            residual_velocity = velocity - mean_velocity
            residual_norm = np.linalg.norm(residual_velocity, axis=1)

            selected_residuals = residual_norm[selected]
            selected_weights = weights[selected]

            # RMS alrededor de cero, tal como se solicita.
            scale = rms_about_zero(
                selected_residuals,
                selected_weights,
            )

            if not np.isfinite(scale) or scale <= 0.0:
                break

            new_selected = selected & (
                residual_norm <= sigma_clip * scale
            )

            if new_selected.sum() < 3:
                break

            if np.array_equal(new_selected, selected):
                break

            selected = new_selected

    stars = stars.loc[selected].reset_index(drop=True)
    velocity = velocity[selected]
    weights = weights[selected]
    distance_pc = distance_pc[selected]

    if len(stars) < 3:
        raise ValueError(
            "Después del filtrado quedaron menos de tres estrellas."
        )

    # ------------------------------------------------------------------
    # Velocidad media y dirección del ápex
    # ------------------------------------------------------------------
    mean_velocity = weighted_mean_vector(velocity, weights)
    mean_speed = float(np.linalg.norm(mean_velocity))

    apex_icrs = vector_to_icrs(mean_velocity)
    antapex_icrs = vector_to_icrs(-mean_velocity)
    apex_galactic = apex_icrs.transform_to(Galactic())

    # ------------------------------------------------------------------
    # Residuales respecto al movimiento común
    # ------------------------------------------------------------------
    residual_velocity = velocity - mean_velocity

    residual_vx = residual_velocity[:, 0]
    residual_vy = residual_velocity[:, 1]
    residual_vz = residual_velocity[:, 2]
    residual_speed = np.linalg.norm(residual_velocity, axis=1)

    sigma_vx = rms_about_zero(residual_vx, weights)
    sigma_vy = rms_about_zero(residual_vy, weights)
    sigma_vz = rms_about_zero(residual_vz, weights)

    # RMS del módulo del residual tridimensional:
    #
    # sqrt(<dv_x² + dv_y² + dv_z²>)
    sigma_3d = rms_about_zero(residual_speed, weights)

    # Dispersión equivalente por dimensión:
    #
    # sigma_1D = sigma_3D / sqrt(3)
    sigma_1d = sigma_3d / np.sqrt(3.0)

    # ------------------------------------------------------------------
    # Dispersión angular de las velocidades individuales
    # ------------------------------------------------------------------

    # Módulo de la velocidad tridimensional de cada estrella.
    individual_speeds = np.linalg.norm(velocity, axis=1)

    # Solo las velocidades con módulo positivo tienen una dirección válida.
    valid_direction = (
        np.isfinite(individual_speeds)
        & (individual_speeds > 0.0)
    )

    # Vector unitario de velocidad de cada estrella.
    individual_unit_vectors = np.full(
        velocity.shape,
        np.nan,
        dtype=float,
    )

    individual_unit_vectors[valid_direction] = (
        velocity[valid_direction]
        / individual_speeds[valid_direction, None]
    )

    # Vector unitario que apunta hacia el ápex medio del cúmulo.
    if not np.isfinite(mean_speed) or mean_speed <= 0.0:
        raise ValueError(
            "La velocidad media del cúmulo tiene módulo nulo o inválido."
        )

    apex_unit_vector = mean_velocity / mean_speed

    # Producto punto entre cada dirección de velocidad y el ápex.
    dot_products = np.full(len(stars), np.nan, dtype=float)

    dot_products[valid_direction] = (
        individual_unit_vectors[valid_direction]
        @ apex_unit_vector
    )

    # Evita errores numéricos ligeramente fuera de [-1, 1].
    dot_products[valid_direction] = np.clip(
        dot_products[valid_direction],
        -1.0,
        1.0,
    )

    # Residual angular individual:
    #
    # theta_i = arccos(vhat_i · ahat)
    #
    # Representa la separación angular entre la dirección de movimiento
    # de cada estrella y la dirección media del ápex.
    angular_residual_deg = np.full(
        len(stars),
        np.nan,
        dtype=float,
    )

    angular_residual_deg[valid_direction] = np.degrees(
        np.arccos(dot_products[valid_direction])
    )

    # Cuadrado del residual angular individual.
    #
    # Esta es la cantidad que aporta cada estrella al cálculo del RMS:
    #
    # RMS = sqrt(sum(w_i * theta_i²) / sum(w_i))
    angular_residual_squared_deg2 = angular_residual_deg**2

    # RMS angular global ponderado del cúmulo.
    sigma_apex_rms_deg = rms_about_zero(
        angular_residual_deg[valid_direction],
        weights[valid_direction],
    )

    # Aporte ponderado no normalizado de cada estrella:
    #
    # w_i * theta_i²
    angular_weighted_squared_contribution = np.full(
        len(stars),
        np.nan,
        dtype=float,
    )

    angular_weighted_squared_contribution[valid_direction] = (
        weights[valid_direction]
        * angular_residual_squared_deg2[valid_direction]
    )

    # Contribución fraccional de cada estrella a la suma cuadrática total.
    #
    # La suma de esta columna para estrellas válidas debe ser aproximadamente 1.
    total_angular_squared_contribution = np.nansum(
        angular_weighted_squared_contribution
    )

    angular_fractional_contribution = np.full(
        len(stars),
        np.nan,
        dtype=float,
    )

    if (
        np.isfinite(total_angular_squared_contribution)
        and total_angular_squared_contribution > 0.0
    ):
        angular_fractional_contribution[valid_direction] = (
            angular_weighted_squared_contribution[valid_direction]
            / total_angular_squared_contribution
        )

    # Residual angular expresado en unidades del RMS global.
    #
    # Ejemplo:
    #   valor = 1.0 -> la estrella está a una distancia angular igual al RMS
    #   valor = 2.0 -> la estrella está a dos veces el RMS
    angular_residual_over_rms = np.full(
        len(stars),
        np.nan,
        dtype=float,
    )

    if (
        np.isfinite(sigma_apex_rms_deg)
        and sigma_apex_rms_deg > 0.0
    ):
        angular_residual_over_rms[valid_direction] = (
            angular_residual_deg[valid_direction]
            / sigma_apex_rms_deg
        )

    # Indicadores prácticos de posibles estrellas discrepantes.
    angular_outlier_2rms = np.zeros(len(stars), dtype=bool)
    angular_outlier_3rms = np.zeros(len(stars), dtype=bool)

    angular_outlier_2rms[valid_direction] = (
        angular_residual_over_rms[valid_direction] > 2.0
    )

    angular_outlier_3rms[valid_direction] = (
        angular_residual_over_rms[valid_direction] > 3.0
    )

    # ------------------------------------------------------------------
    # Bootstrap del ápex
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_state)

    bootstrap_ra = []
    bootstrap_dec = []
    bootstrap_separation = []

    if bootstrap_samples > 0:
        probabilities = weights / np.sum(weights)
        n_stars = len(stars)

        for _ in range(bootstrap_samples):
            indices = rng.choice(
                n_stars,
                size=n_stars,
                replace=True,
                p=probabilities,
            )

            # Media ponderada de la muestra bootstrap.
            #
            # Los índices se extraen usando probabilidades derivadas de
            # los pesos originales. Dentro de cada muestra se conservan
            # también los pesos de las estrellas seleccionadas.
            bootstrap_vector = weighted_mean_vector(
                velocity[indices],
                weights[indices],
            )

            bootstrap_norm = np.linalg.norm(bootstrap_vector)

            if (
                not np.isfinite(bootstrap_norm)
                or bootstrap_norm <= 0.0
            ):
                continue

            bootstrap_apex = vector_to_icrs(bootstrap_vector)

            bootstrap_ra.append(
                bootstrap_apex.ra.wrap_at(360.0 * u.deg).deg
            )

            bootstrap_dec.append(
                bootstrap_apex.dec.deg
            )

            bootstrap_separation.append(
                bootstrap_apex.separation(apex_icrs).deg
            )

    if bootstrap_ra:
        bootstrap_ra = np.asarray(
            bootstrap_ra,
            dtype=float,
        )

        bootstrap_dec = np.asarray(
            bootstrap_dec,
            dtype=float,
        )

        bootstrap_separation = np.asarray(
            bootstrap_separation,
            dtype=float,
        )

        # La RA es una variable circular.
        delta_ra = (
            (
                bootstrap_ra
                - apex_icrs.ra.deg
                + 180.0
            )
            % 360.0
            - 180.0
        )

        # Proyección de la diferencia de RA en el plano tangente.
        delta_ra_projected = (
            delta_ra
            * np.cos(apex_icrs.dec.to_value(u.rad))
        )

        # RMS de las fluctuaciones bootstrap en RA.
        ra_bootstrap_std = float(
            np.sqrt(
                np.mean(delta_ra_projected**2)
            )
        )

        # RMS de las fluctuaciones bootstrap en Dec.
        dec_bootstrap_std = float(
            np.sqrt(
                np.mean(
                    (
                        bootstrap_dec
                        - apex_icrs.dec.deg
                    )
                    ** 2
                )
            )
        )

        # RMS de la separación angular entre cada ápex bootstrap
        # y el ápex calculado con la muestra completa.
        angular_bootstrap_rms = float(
            np.sqrt(
                np.mean(bootstrap_separation**2)
            )
        )

    else:
        ra_bootstrap_std = np.nan
        dec_bootstrap_std = np.nan
        angular_bootstrap_rms = np.nan

    # ------------------------------------------------------------------
    # Tabla individual de diagnóstico
    # ------------------------------------------------------------------

    # Distancia estimada.
    stars["distance_pc"] = distance_pc

    # Velocidades cartesianas ICRS.
    stars["velocity_x_icrs_kms"] = velocity[:, 0]
    stars["velocity_y_icrs_kms"] = velocity[:, 1]
    stars["velocity_z_icrs_kms"] = velocity[:, 2]
    stars["speed_3d_kms"] = individual_speeds

    # Residuales de velocidad respecto al movimiento medio.
    stars["residual_vx_kms"] = residual_vx
    stars["residual_vy_kms"] = residual_vy
    stars["residual_vz_kms"] = residual_vz
    stars["residual_speed_3d_kms"] = residual_speed

    # ------------------------------------------------------------------
    # Información angular por estrella
    # ------------------------------------------------------------------

    # Ángulo individual entre la velocidad de la estrella y el ápex.
    #
    # Esta es la columna principal que debes consultar.
    stars["apex_angular_residual_deg"] = (
        angular_residual_deg
    )

    # Se conserva también el nombre anterior por compatibilidad.
    stars["velocity_apex_angular_residual_deg"] = (
        angular_residual_deg
    )

    # theta_i²: contribución individual antes de aplicar pesos.
    stars["apex_angular_residual_squared_deg2"] = (
        angular_residual_squared_deg2
    )

    # w_i * theta_i²: contribución ponderada no normalizada.
    stars["apex_angular_weighted_contribution_deg2"] = (
        angular_weighted_squared_contribution
    )

    # Fracción del total cuadrático aportada por cada estrella.
    #
    # La suma de esta columna es aproximadamente 1.
    stars["apex_angular_fractional_contribution"] = (
        angular_fractional_contribution
    )

    # Separación angular individual dividida entre el RMS global.
    stars["apex_angular_residual_over_rms"] = (
        angular_residual_over_rms
    )

    # Indicadores simples de estrellas alejadas más de 2 o 3 RMS.
    stars["apex_angular_outlier_2rms"] = (
        angular_outlier_2rms
    )

    stars["apex_angular_outlier_3rms"] = (
        angular_outlier_3rms
    )

    # RMS angular global repetido en cada fila.
    #
    # Este valor será el mismo para todas las estrellas porque es una
    # propiedad global del conjunto, no una propiedad individual.
    stars["cluster_apex_angular_rms_deg"] = (
        sigma_apex_rms_deg
    )

    # Pesos cinemáticos utilizados.
    stars["kinematic_weight"] = weights

    # RA de la dirección de velocidad de cada estrella
    velocity_direction_ra_deg = (
        np.degrees(
            np.arctan2(
                individual_unit_vectors[:, 1],
                individual_unit_vectors[:, 0],
            )
        )
        % 360.0
    )

    # Diferencia de RA envuelta en el intervalo [-180, 180)
    delta_ra_deg = (
        (
            velocity_direction_ra_deg
            - apex_icrs.ra.deg
            + 180.0
        )
        % 360.0
        - 180.0
    )

    # Asignar signo a la separación angular:
    # positivo: dirección al este del ápex
    # negativo: dirección al oeste del ápex
    signo_angular = np.sign(delta_ra_deg)

    # Evita que una diferencia exactamente igual a cero produzca signo cero
    signo_angular[signo_angular == 0.0] = 1.0

    signed_angular_residual_deg = (
        angular_residual_deg
        * signo_angular
    )

    stars["apex_angular_residual_signed_deg"] = (
        signed_angular_residual_deg
    )

    return ApexResult(
        n_stars=len(stars),

        apex_ra_deg=float(apex_icrs.ra.deg),
        apex_dec_deg=float(apex_icrs.dec.deg),
        antapex_ra_deg=float(antapex_icrs.ra.deg),
        antapex_dec_deg=float(antapex_icrs.dec.deg),

        apex_l_deg=float(apex_galactic.l.deg),
        apex_b_deg=float(apex_galactic.b.deg),

        vx_mean_kms=float(mean_velocity[0]),
        vy_mean_kms=float(mean_velocity[1]),
        vz_mean_kms=float(mean_velocity[2]),
        speed_mean_kms=mean_speed,

        sigma_vx_rms_kms=sigma_vx,
        sigma_vy_rms_kms=sigma_vy,
        sigma_vz_rms_kms=sigma_vz,
        sigma_3d_rms_kms=sigma_3d,
        sigma_1d_rms_kms=sigma_1d,

        sigma_apex_rms_deg=sigma_apex_rms_deg,

        apex_ra_bootstrap_std_deg=ra_bootstrap_std,
        apex_dec_bootstrap_std_deg=dec_bootstrap_std,
        apex_angular_bootstrap_rms_deg=angular_bootstrap_rms,

        stars=stars,
    )