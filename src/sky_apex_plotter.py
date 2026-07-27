"""
sky_apex_plotter.py

Visualización reutilizable para cúmulos simulados o reales:
    - estrellas del cúmulo
    - polos del movimiento propio
    - apex y antapex

Soporta tres modos:
    projection="radec"      -> plano RA vs Dec
    projection="mollweide"  -> proyección oval tipo mapa celeste
    projection="sphere"     -> esfera 3D interactiva con Plotly

Pensado para usarse con salidas de módulos tipo vector_director_icrs.py,
donde los polos están en columnas:
    pole_x_unit, pole_y_unit, pole_z_unit

y el apex puede venir como:
    result["cluster_result"]["apex_refined"]["apex_vector"]
o como diccionario con:
    apex_ra_deg, apex_dec_deg
o:
    apex_l_deg, apex_b_deg
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord


ProjectionType = Literal["radec", "mollweide", "sphere"]


# ============================================================
# Utilidades geométricas
# ============================================================

def radec_to_xyz(
    ra_deg: np.ndarray | float,
    dec_deg: np.ndarray | float,
) -> np.ndarray:
    """Convierte RA, Dec en grados a vectores cartesianos unitarios ICRS."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    return np.column_stack([x, y, z]) if np.ndim(ra_deg) else np.array([x, y, z])


def xyz_to_radec(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte vectores cartesianos unitarios ICRS a RA, Dec en grados."""
    xyz = np.asarray(xyz, dtype=float)

    if xyz.ndim == 1:
        xyz = xyz.reshape(1, 3)

    norm = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (norm > 0)

    ra = np.full(len(xyz), np.nan)
    dec = np.full(len(xyz), np.nan)

    v = xyz[valid] / norm[valid, None]
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    ra[valid] = np.degrees(np.arctan2(y, x)) % 360.0
    dec[valid] = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))

    return ra, dec


def wrap_ra_centered(
    ra_deg: np.ndarray | float,
    center_ra_deg: float = 180.0,
) -> np.ndarray:
    """
    Envuelve RA alrededor de un centro dado para usar en mapas 2D.

    Devuelve valores en [-180, 180].
    """
    return ((np.asarray(ra_deg) - center_ra_deg + 180.0) % 360.0) - 180.0


def antipode_radec(ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    """Devuelve el punto opuesto en la esfera celeste."""
    return (ra_deg + 180.0) % 360.0, -dec_deg


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normaliza un vector 3D."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("El vector no se puede normalizar.")

    return vector / norm


# ============================================================
# Lectura flexible de apex y polos
# ============================================================

def extract_apex_radec(
    apex: Optional[Dict[str, Any]] = None,
    analysis_result: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """
    Extrae RA, Dec del apex desde varias estructuras posibles.

    Acepta:
        - apex={"apex_ra_deg": ..., "apex_dec_deg": ...}
        - apex={"apex_vector": np.array([x, y, z])}
        - apex={"apex_l_deg": ..., "apex_b_deg": ...}
        - analysis_result["cluster_result"]["apex_refined"]
    """

    if apex is None and analysis_result is not None:
        if "cluster_result" in analysis_result:
            apex = analysis_result["cluster_result"].get("apex_refined")
        elif "apex_refined" in analysis_result:
            apex = analysis_result["apex_refined"]
        else:
            apex = analysis_result

    if apex is None:
        raise ValueError(
            "Debes pasar un diccionario apex=... o analysis_result=..."
        )

    if "apex_ra_deg" in apex and "apex_dec_deg" in apex:
        return float(apex["apex_ra_deg"]), float(apex["apex_dec_deg"])

    if "apex_vector" in apex:
        ra, dec = xyz_to_radec(np.asarray(apex["apex_vector"], dtype=float))
        return float(ra[0]), float(dec[0])

    if "apex_l_deg" in apex and "apex_b_deg" in apex:
        coord = SkyCoord(
            l=float(apex["apex_l_deg"]) * u.deg,
            b=float(apex["apex_b_deg"]) * u.deg,
            frame="galactic",
        ).icrs
        return float(coord.ra.deg), float(coord.dec.deg)

    raise ValueError(
        "No pude extraer el apex. Usa apex_ra_deg/apex_dec_deg, "
        "apex_vector o apex_l_deg/apex_b_deg."
    )


def extract_poles_radec(
    df: pd.DataFrame,
    pole_ra_col: str = "pole_ra_deg",
    pole_dec_col: str = "pole_dec_deg",
    pole_x_col: str = "pole_x_unit",
    pole_y_col: str = "pole_y_unit",
    pole_z_col: str = "pole_z_unit",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae los polos en RA, Dec.

    Primero intenta usar columnas de ángulos:
        pole_ra_deg, pole_dec_deg

    Si no existen, usa columnas cartesianas:
        pole_x_unit, pole_y_unit, pole_z_unit
    """

    if pole_ra_col in df.columns and pole_dec_col in df.columns:
        return (
            df[pole_ra_col].to_numpy(dtype=float),
            df[pole_dec_col].to_numpy(dtype=float),
        )

    required = [pole_x_col, pole_y_col, pole_z_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "No encontré columnas de polos. Faltan: "
            + ", ".join(missing)
            + ". Esperaba pole_ra_deg/pole_dec_deg o "
            + "pole_x_unit/pole_y_unit/pole_z_unit."
        )

    xyz = df[required].to_numpy(dtype=float)
    return xyz_to_radec(xyz)


# ============================================================
# Círculo máximo esperado de los polos
# ============================================================

def great_circle_perpendicular_to_vector(
    normal_vector: np.ndarray,
    n_points: int = 361,
) -> np.ndarray:
    """
    Genera un círculo máximo cuyo plano es perpendicular al vector dado.

    Para el caso del CPM, si el apex es el vector normal, los polos deberían
    caer aproximadamente sobre este círculo máximo.
    """

    n = normalize_vector(normal_vector)

    # Escoge un vector auxiliar que no sea paralelo a n
    aux = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(aux, n)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])

    e1 = normalize_vector(np.cross(n, aux))
    e2 = normalize_vector(np.cross(n, e1))

    t = np.linspace(0.0, 2.0 * np.pi, n_points)

    circle = (
        np.cos(t)[:, None] * e1[None, :]
        + np.sin(t)[:, None] * e2[None, :]
    )

    return circle


# ============================================================
# Graficadores
# ============================================================

def plot_cluster_motion_geometry(
    df: pd.DataFrame,
    *,
    projection: ProjectionType = "radec",
    analysis_result: Optional[Dict[str, Any]] = None,
    apex: Optional[Dict[str, Any]] = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
    pole_ra_col: str = "pole_ra_deg",
    pole_dec_col: str = "pole_dec_deg",
    pole_x_col: str = "pole_x_unit",
    pole_y_col: str = "pole_y_unit",
    pole_z_col: str = "pole_z_unit",
    center_ra_deg: float = 180.0,
    reverse_ra: bool = True,
    show_poles: bool = True,
    show_antapex: bool = True,
    show_pole_great_circle: bool = True,
    star_size: float = 18,
    pole_size: float = 18,
    title: Optional[str] = None,
    save_html: Optional[str] = None,
    save_fig: Optional[str] = None,
):
    """
    Grafica estrellas del cúmulo, polos, apex y antapex.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabla con las estrellas del cúmulo. Debe tener columnas RA/Dec y,
        para graficar polos, columnas de polos en formato angular o cartesiano.

    projection : {"radec", "mollweide", "sphere"}
        Tipo de visualización:
            - "radec": plano RA-Dec.
            - "mollweide": proyección oval celeste.
            - "sphere": esfera 3D interactiva con Plotly.

    analysis_result : dict, optional
        Salida completa de tu análisis, por ejemplo result de run_cluster_analysis.

    apex : dict, optional
        Diccionario del apex. Si se pasa analysis_result, no es necesario.

    ra_col, dec_col : str
        Nombres de columnas de coordenadas ecuatoriales de las estrellas.

    pole_* : str
        Nombres de columnas para los polos.

    center_ra_deg : float
        Centro de RA para las proyecciones 2D.

    reverse_ra : bool
        Si True, invierte el eje de RA para seguir la convención astronómica.

    show_poles : bool
        Si True, grafica los polos.

    show_antapex : bool
        Si True, grafica el antapex.

    show_pole_great_circle : bool
        Si True, grafica el círculo máximo donde deberían alinearse los polos.

    save_html : str, optional
        Ruta para guardar la figura interactiva HTML cuando projection="sphere".

    save_fig : str, optional
        Ruta para guardar figura estática en "radec" o "mollweide".

    Returns
    -------
    fig
        Figura de Matplotlib o Plotly, según projection.
    """

    if ra_col not in df.columns or dec_col not in df.columns:
        raise ValueError(f"El DataFrame debe tener columnas {ra_col}, {dec_col}.")

    stars_ra = df[ra_col].to_numpy(dtype=float)
    stars_dec = df[dec_col].to_numpy(dtype=float)

    apex_ra, apex_dec = extract_apex_radec(
        apex=apex,
        analysis_result=analysis_result,
    )
    antapex_ra, antapex_dec = antipode_radec(apex_ra, apex_dec)

    pole_ra = pole_dec = None
    if show_poles:
        pole_ra, pole_dec = extract_poles_radec(
            df,
            pole_ra_col=pole_ra_col,
            pole_dec_col=pole_dec_col,
            pole_x_col=pole_x_col,
            pole_y_col=pole_y_col,
            pole_z_col=pole_z_col,
        )

    if projection == "radec":
        fig = _plot_radec(
            stars_ra=stars_ra,
            stars_dec=stars_dec,
            pole_ra=pole_ra,
            pole_dec=pole_dec,
            apex_ra=apex_ra,
            apex_dec=apex_dec,
            antapex_ra=antapex_ra,
            antapex_dec=antapex_dec,
            center_ra_deg=center_ra_deg,
            reverse_ra=reverse_ra,
            show_poles=show_poles,
            show_antapex=show_antapex,
            star_size=star_size,
            pole_size=pole_size,
            title=title,
        )

    elif projection == "mollweide":
        fig = _plot_mollweide(
            stars_ra=stars_ra,
            stars_dec=stars_dec,
            pole_ra=pole_ra,
            pole_dec=pole_dec,
            apex_ra=apex_ra,
            apex_dec=apex_dec,
            antapex_ra=antapex_ra,
            antapex_dec=antapex_dec,
            center_ra_deg=center_ra_deg,
            reverse_ra=reverse_ra,
            show_poles=show_poles,
            show_antapex=show_antapex,
            show_pole_great_circle=show_pole_great_circle,
            star_size=star_size,
            pole_size=pole_size,
            title=title,
        )

    elif projection == "sphere":
        fig = _plot_sphere_interactive(
            stars_ra=stars_ra,
            stars_dec=stars_dec,
            pole_ra=pole_ra,
            pole_dec=pole_dec,
            apex_ra=apex_ra,
            apex_dec=apex_dec,
            antapex_ra=antapex_ra,
            antapex_dec=antapex_dec,
            show_poles=show_poles,
            show_antapex=show_antapex,
            show_pole_great_circle=show_pole_great_circle,
            star_size=star_size,
            pole_size=pole_size,
            title=title,
        )

    else:
        raise ValueError(
            "projection debe ser 'radec', 'mollweide' o 'sphere'."
        )

    if save_fig is not None and projection in {"radec", "mollweide"}:
        fig.savefig(save_fig, dpi=250, bbox_inches="tight")

    if save_html is not None and projection == "sphere":
        fig.write_html(save_html)

    return fig


def _project_ra_for_2d(
    ra_deg: np.ndarray | float,
    center_ra_deg: float,
    reverse_ra: bool,
) -> np.ndarray:
    x = wrap_ra_centered(ra_deg, center_ra_deg=center_ra_deg)
    if reverse_ra:
        x = -x
    return x


def _plot_radec(
    *,
    stars_ra,
    stars_dec,
    pole_ra,
    pole_dec,
    apex_ra,
    apex_dec,
    antapex_ra,
    antapex_dec,
    center_ra_deg,
    reverse_ra,
    show_poles,
    show_antapex,
    star_size,
    pole_size,
    title,
):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x_stars = _project_ra_for_2d(stars_ra, center_ra_deg, reverse_ra)
    x_apex = _project_ra_for_2d(apex_ra, center_ra_deg, reverse_ra)
    x_antapex = _project_ra_for_2d(antapex_ra, center_ra_deg, reverse_ra)

    ax.scatter(
        x_stars,
        stars_dec,
        s=star_size,
        alpha=0.75,
        color="tab:blue",
        label="Estrellas del cúmulo",
    )

    if show_poles and pole_ra is not None:
        x_poles = _project_ra_for_2d(pole_ra, center_ra_deg, reverse_ra)
        ax.scatter(
            x_poles,
            pole_dec,
            s=pole_size,
            alpha=0.65,
            color="tab:orange",
            label="Polos",
        )

    ax.scatter(
        x_apex,
        apex_dec,
        s=180,
        marker="*",
        color="crimson",
        edgecolor="black",
        linewidth=0.8,
        label="Apex",
        zorder=5,
    )

    if show_antapex:
        ax.scatter(
            x_antapex,
            antapex_dec,
            s=130,
            marker="X",
            color="purple",
            edgecolor="black",
            linewidth=0.8,
            label="Antapex",
            zorder=5,
        )

    ax.set_xlabel(r"$\Delta$RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_ylim(-90, 90)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    if title is None:
        title = "Geometría del movimiento: cúmulo, polos y apex"

    ax.set_title(title)

    return fig


def _plot_mollweide(
    *,
    stars_ra,
    stars_dec,
    pole_ra,
    pole_dec,
    apex_ra,
    apex_dec,
    antapex_ra,
    antapex_dec,
    center_ra_deg,
    reverse_ra,
    show_poles,
    show_antapex,
    show_pole_great_circle,
    star_size,
    pole_size,
    title,
):
    fig = plt.figure(figsize=(10, 5.8))
    ax = fig.add_subplot(111, projection="mollweide")

    x_stars = np.deg2rad(_project_ra_for_2d(stars_ra, center_ra_deg, reverse_ra))
    y_stars = np.deg2rad(stars_dec)

    ax.scatter(
        x_stars,
        y_stars,
        s=star_size,
        alpha=0.75,
        color="tab:blue",
        label="Estrellas del cúmulo",
    )

    if show_poles and pole_ra is not None:
        x_poles = np.deg2rad(_project_ra_for_2d(pole_ra, center_ra_deg, reverse_ra))
        y_poles = np.deg2rad(pole_dec)

        ax.scatter(
            x_poles,
            y_poles,
            s=pole_size,
            alpha=0.65,
            color="tab:orange",
            label="Polos",
        )

    x_apex = np.deg2rad(_project_ra_for_2d(apex_ra, center_ra_deg, reverse_ra))
    y_apex = np.deg2rad(apex_dec)

    ax.scatter(
        x_apex,
        y_apex,
        s=180,
        marker="*",
        color="crimson",
        edgecolor="black",
        linewidth=0.8,
        label="Apex",
        zorder=5,
    )

    if show_antapex:
        x_antapex = np.deg2rad(
            _project_ra_for_2d(antapex_ra, center_ra_deg, reverse_ra)
        )
        y_antapex = np.deg2rad(antapex_dec)

        ax.scatter(
            x_antapex,
            y_antapex,
            s=130,
            marker="X",
            color="purple",
            edgecolor="black",
            linewidth=0.8,
            label="Antapex",
            zorder=5,
        )

    if show_pole_great_circle:
        apex_vector = radec_to_xyz(apex_ra, apex_dec)
        circle_xyz = great_circle_perpendicular_to_vector(apex_vector)
        circle_ra, circle_dec = xyz_to_radec(circle_xyz)

        x_circle = np.deg2rad(
            _project_ra_for_2d(circle_ra, center_ra_deg, reverse_ra)
        )
        y_circle = np.deg2rad(circle_dec)

        ax.plot(
            x_circle,
            y_circle,
            color="black",
            alpha=0.35,
            linewidth=1.2,
            label="Círculo esperado de polos",
        )

    ax.grid(alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    if title is None:
        title = "Proyección Mollweide: cúmulo, polos y apex"

    ax.set_title(title, pad=18)

    return fig


def _plot_sphere_interactive(
    *,
    stars_ra,
    stars_dec,
    pole_ra,
    pole_dec,
    apex_ra,
    apex_dec,
    antapex_ra,
    antapex_dec,
    show_poles,
    show_antapex,
    show_pole_great_circle,
    star_size,
    pole_size,
    title,
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Para projection='sphere' instala Plotly con: pip install plotly"
        ) from exc

    stars_xyz = radec_to_xyz(stars_ra, stars_dec)
    apex_xyz = radec_to_xyz(apex_ra, apex_dec)
    antapex_xyz = radec_to_xyz(antapex_ra, antapex_dec)

    traces = []

    # Esfera semitransparente
    u_grid = np.linspace(0, 2 * np.pi, 80)
    v_grid = np.linspace(-np.pi / 2, np.pi / 2, 40)

    xs = np.outer(np.cos(v_grid), np.cos(u_grid))
    ys = np.outer(np.cos(v_grid), np.sin(u_grid))
    zs = np.outer(np.sin(v_grid), np.ones_like(u_grid))

    traces.append(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            opacity=0.12,
            showscale=False,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            name="Esfera celeste",
            hoverinfo="skip",
        )
    )

    traces.append(
        go.Scatter3d(
            x=stars_xyz[:, 0],
            y=stars_xyz[:, 1],
            z=stars_xyz[:, 2],
            mode="markers",
            marker=dict(size=star_size / 4, color="royalblue", opacity=0.8),
            name="Estrellas del cúmulo",
            text=[
                f"RA={ra:.4f} deg<br>Dec={dec:.4f} deg"
                for ra, dec in zip(stars_ra, stars_dec)
            ],
            hoverinfo="text",
        )
    )

    if show_poles and pole_ra is not None:
        poles_xyz = radec_to_xyz(pole_ra, pole_dec)

        traces.append(
            go.Scatter3d(
                x=poles_xyz[:, 0],
                y=poles_xyz[:, 1],
                z=poles_xyz[:, 2],
                mode="markers",
                marker=dict(size=pole_size / 4, color="orange", opacity=0.75),
                name="Polos",
                text=[
                    f"Polo RA={ra:.4f} deg<br>Polo Dec={dec:.4f} deg"
                    for ra, dec in zip(pole_ra, pole_dec)
                ],
                hoverinfo="text",
            )
        )

    traces.append(
        go.Scatter3d(
            x=[apex_xyz[0]],
            y=[apex_xyz[1]],
            z=[apex_xyz[2]],
            mode="markers+text",
            marker=dict(size=10, color="crimson", symbol="diamond"),
            text=["Apex"],
            textposition="top center",
            name="Apex",
            customdata=[[apex_ra, apex_dec]],
            hovertemplate=(
                "<b>Apex</b><br>"
                "RA = %{customdata[0]:.6f} deg<br>"
                "Dec = %{customdata[1]:.6f} deg<br>"
                "X = %{x:.6f}<br>"
                "Y = %{y:.6f}<br>"
                "Z = %{z:.6f}"
                "<extra></extra>"
            ),
        )
    )

    if show_antapex:
        traces.append(
            go.Scatter3d(
                x=[antapex_xyz[0]],
                y=[antapex_xyz[1]],
                z=[antapex_xyz[2]],
                mode="markers+text",
                marker=dict(size=8, color="purple", symbol="x"),
                text=["Antapex"],
                textposition="top center",
                name="Antapex",
                customdata=[[antapex_ra, antapex_dec]],
                hovertemplate=(
                    "<b>Antapex</b><br>"
                    "RA = %{customdata[0]:.6f} deg<br>"
                    "Dec = %{customdata[1]:.6f} deg<br>"
                    "X = %{x:.6f}<br>"
                    "Y = %{y:.6f}<br>"
                    "Z = %{z:.6f}"
                    "<extra></extra>"
                ),
            )
        )

    if show_pole_great_circle:
        circle_xyz = great_circle_perpendicular_to_vector(apex_xyz)

        traces.append(
            go.Scatter3d(
                x=circle_xyz[:, 0],
                y=circle_xyz[:, 1],
                z=circle_xyz[:, 2],
                mode="lines",
                line=dict(color="black", width=4),
                opacity=0.45,
                name="Círculo esperado de polos",
                hoverinfo="skip",
            )
        )

    if title is None:
        title = "Esfera celeste interactiva: cúmulo, polos y apex"

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=title,
        width=900,
        height=750,
        scene=dict(
            xaxis=dict(title="X ICRS", showbackground=False),
            yaxis=dict(title="Y ICRS", showbackground=False),
            zaxis=dict(title="Z ICRS", showbackground=False),
            aspectmode="data",
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )

    return fig

from scipy.ndimage import gaussian_filter


def plot_pole_density_from_dataframe(
    df,
    n_lon=720,
    n_lat=360,
    smoothing=1.5,
    background_smoothing=15,
    enhance_ridges=True,
    percentile=99.7,
    cmap="inferno",
):
    """
    Grafica la densidad de polos utilizando pole_x_unit,
    pole_y_unit y pole_z_unit.

    Está diseñada para DataFrames grandes: no genera un scatter
    con todos los puntos, sino una imagen rasterizada.
    """

    required_columns = [
        "pole_x_unit",
        "pole_y_unit",
        "pole_z_unit",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias: {missing}"
        )

    # copy=False evita una copia cuando pandas lo permite
    x = df["pole_x_unit"].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    y = df["pole_y_unit"].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    z = df["pole_z_unit"].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]

    if x.size == 0:
        raise ValueError(
            "No hay polos válidos para graficar."
        )

    # Protección frente a pequeños errores numéricos
    z = np.clip(z, -1.0, 1.0)

    # Coordenadas esféricas de los polos
    pole_lon = np.degrees(np.arctan2(y, x))
    pole_lat = np.degrees(np.arcsin(z))

    lon_edges = np.linspace(
        -180.0,
        180.0,
        n_lon + 1,
        dtype=np.float32,
    )

    lat_edges = np.linspace(
        -90.0,
        90.0,
        n_lat + 1,
        dtype=np.float32,
    )

    # histogram2d recibe primero y, luego x
    density, _, _ = np.histogram2d(
        pole_lat,
        pole_lon,
        bins=[lat_edges, lon_edges],
    )

    density = density.astype(
        np.float32,
        copy=False,
    )

    # Suavizado a pequeña escala
    density_small = gaussian_filter(
        density,
        sigma=smoothing,
        mode=("nearest", "wrap"),
    )

    if enhance_ridges:
        # Fondo de gran escala
        density_background = gaussian_filter(
            density,
            sigma=background_smoothing,
            mode=("nearest", "wrap"),
        )

        # Resalta bandas o cicatrices
        image = density_small - density_background
        image[image < 0] = 0
    else:
        image = density_small

    image_log = np.log1p(image)

    positive = image_log[image_log > 0]

    if positive.size > 0:
        vmax = np.percentile(
            positive,
            percentile,
        )
    else:
        vmax = 1.0

    lon_centers = 0.5 * (
        lon_edges[:-1] + lon_edges[1:]
    )
    lat_centers = 0.5 * (
        lat_edges[:-1] + lat_edges[1:]
    )

    lon_grid, lat_grid = np.meshgrid(
        np.radians(lon_centers),
        np.radians(lat_centers),
    )

    fig = plt.figure(figsize=(14, 7))

    ax = fig.add_subplot(
        111,
        projection="mollweide",
    )

    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        image_log,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )

    ax.grid(
        color="white",
        alpha=0.20,
        linewidth=0.5,
    )

    if enhance_ridges:
        title = (
            "Crestas de densidad en el espacio de polos"
        )
    else:
        title = "Densidad de polos"

    ax.set_title(title, pad=18)

    colorbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        fraction=0.05,
    )

    colorbar.set_label(
        r"$\log(1 + \mathrm{densidad})$"
    )

    plt.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "density": density,
        "image": image,
        "pole_lon_deg": pole_lon,
        "pole_lat_deg": pole_lat,
        "valid_mask": valid,
    }

def fit_great_circle_and_straighten(
    df,
    n_lon=720,
    n_lat=240,
    lat_limit=30.0,
    smoothing=1.5,
    percentile=99.7,
):
    """
    Ajusta un círculo máximo a los polos y rota el sistema para que
    dicho círculo aparezca como una línea horizontal en latitud = 0.

    Requiere:
        pole_x_unit, pole_y_unit, pole_z_unit
    """

    cols = ["pole_x_unit", "pole_y_unit", "pole_z_unit"]

    xyz = df[cols].to_numpy(dtype=np.float64, copy=False)

    valid = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[valid]

    if len(xyz) < 3:
        raise ValueError("No hay suficientes polos válidos.")

    # Renormalizar por seguridad
    norms = np.linalg.norm(xyz, axis=1)
    good_norm = norms > 0

    xyz = xyz[good_norm]
    xyz = xyz / norms[good_norm, None]

    # ------------------------------------------------------------
    # 1. Ajuste del plano mediante SVD
    #
    # El vector singular asociado al menor valor singular es
    # la normal del plano que mejor contiene los polos.
    # ------------------------------------------------------------
    _, singular_values, vh = np.linalg.svd(
        xyz,
        full_matrices=False,
    )

    plane_normal = vh[-1]
    plane_normal /= np.linalg.norm(plane_normal)

    # ------------------------------------------------------------
    # 2. Construir una base ortonormal rotada
    #
    # z_new = normal del plano
    # x_new, y_new = ejes dentro del plano
    # ------------------------------------------------------------
    z_new = plane_normal

    reference = np.array([0.0, 0.0, 1.0])

    # Evitar que el vector de referencia sea paralelo a z_new
    if np.abs(np.dot(reference, z_new)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])

    x_new = np.cross(reference, z_new)
    x_new /= np.linalg.norm(x_new)

    y_new = np.cross(z_new, x_new)
    y_new /= np.linalg.norm(y_new)

    # Proyecciones en el sistema rotado
    x_rot = xyz @ x_new
    y_rot = xyz @ y_new
    z_rot = xyz @ z_new

    # Coordenadas angulares rotadas
    lon_rot = np.degrees(
        np.arctan2(y_rot, x_rot)
    )

    lat_rot = np.degrees(
        np.arcsin(np.clip(z_rot, -1.0, 1.0))
    )

    # ------------------------------------------------------------
    # 3. Histograma 2D
    # ------------------------------------------------------------
    lon_edges = np.linspace(
        -180.0,
        180.0,
        n_lon + 1,
    )

    lat_edges = np.linspace(
        -lat_limit,
        lat_limit,
        n_lat + 1,
    )

    density, _, _ = np.histogram2d(
        lat_rot,
        lon_rot,
        bins=[lat_edges, lon_edges],
    )

    density = density.astype(np.float32)

    if smoothing > 0:
        density = gaussian_filter(
            density,
            sigma=(smoothing, smoothing),
            mode=("nearest", "wrap"),
        )

    image = np.log1p(density)

    positive = image[image > 0]

    vmax = (
        np.percentile(positive, percentile)
        if positive.size > 0
        else 1.0
    )

    # ------------------------------------------------------------
    # 4. Gráfica rectangular
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(14, 6),
    )

    extent = [
        -180.0,
        180.0,
        -lat_limit,
        lat_limit,
    ]

    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )

    # Línea del círculo máximo ajustado
    ax.axhline(
        0.0,
        color="cyan",
        linewidth=1.5,
        linestyle="--",
        label="Círculo máximo ajustado",
    )

    ax.set_xlabel(
        "Longitud en el sistema rotado [deg]"
    )

    ax.set_ylabel(
        "Distancia angular al círculo máximo [deg]"
    )

    ax.set_title(
        "Espacio de polos rectificado"
    )

    ax.legend(loc="upper right")

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        r"$\log(1 + N)$"
    )

    plt.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "plane_normal": plane_normal,
        "singular_values": singular_values,
        "lon_rot_deg": lon_rot,
        "lat_rot_deg": lat_rot,
        "density": density,
        "valid_mask": valid,
    }

def plot_apex_lambda_experiment_sphere(
    experiment: Dict[str, Any],
    title: Optional[str] = None,
    show: bool = True,
):
    """
    Grafica en una esfera interactiva:
        - ápex
        - estrella
        - círculo de posibles estrellas a ángulo lambda
        - ecuador del ápex
        - trayectoria ideal y desviadas
        - vectores directores / polos
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Instala plotly: pip install plotly") from exc

    apex = experiment["apex"]["vector"]
    star = experiment["star"]["vector"]

    lambda_circle = experiment["lambda_circle_xyz"]
    apex_equator = experiment["apex_equator_xyz"]

    m_ideal = experiment["motions"]["ideal"]
    m_plus = experiment["motions"]["plus"]
    m_minus = experiment["motions"]["minus"]

    pole_ideal = m_ideal["pole_vector"]
    pole_plus = m_plus["pole_vector"]
    pole_minus = m_minus["pole_vector"]

    star_final_ideal = m_ideal["final_star_vector"]
    star_final_plus = m_plus["final_star_vector"]
    star_final_minus = m_minus["final_star_vector"]

    apex_ra = experiment["apex"]["ra_deg"]
    apex_dec = experiment["apex"]["dec_deg"]

    star_ra = experiment["star"]["ra_deg"]
    star_dec = experiment["star"]["dec_deg"]

    pole_ideal_ra = m_ideal["pole_ra_deg"]
    pole_ideal_dec = m_ideal["pole_dec_deg"]

    pole_plus_ra = m_plus["pole_ra_deg"]
    pole_plus_dec = m_plus["pole_dec_deg"]

    pole_minus_ra = m_minus["pole_ra_deg"]
    pole_minus_dec = m_minus["pole_dec_deg"]

    traces = []

    # Esfera semitransparente
    u = np.linspace(0, 2 * np.pi, 70)
    v = np.linspace(-np.pi / 2, np.pi / 2, 35)

    xs = np.outer(np.cos(v), np.cos(u))
    ys = np.outer(np.cos(v), np.sin(u))
    zs = np.outer(np.sin(v), np.ones_like(u))

    traces.append(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            opacity=0.12,
            showscale=False,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            hoverinfo="skip",
            name="Esfera",
        )
    )

    # Ecuador del ápex
    traces.append(
        go.Scatter3d(
            x=apex_equator[:, 0],
            y=apex_equator[:, 1],
            z=apex_equator[:, 2],
            mode="lines",
            line=dict(color="black", width=5, dash="dash"),
            name="Ecuador del ápex",
            hoverinfo="skip",
        )
    )

    # Círculo lambda: posibles posiciones de la estrella
    traces.append(
        go.Scatter3d(
            x=lambda_circle[:, 0],
            y=lambda_circle[:, 1],
            z=lambda_circle[:, 2],
            mode="lines",
            line=dict(color="royalblue", width=5),
            name="Círculo de posibles estrellas (λ fijo)",
            hoverinfo="skip",
        )
    )

    # Ápex
    traces.append(
        go.Scatter3d(
            x=[apex[0]],
            y=[apex[1]],
            z=[apex[2]],
            mode="markers+text",
            marker=dict(size=9, color="crimson", symbol="diamond"),
            text=["Apex"],
            textposition="top center",
            name="Apex",
            customdata=[[apex_ra, apex_dec]],
            hovertemplate=(
                "<b>Apex</b><br>"
                "RA = %{customdata[0]:.6f} deg<br>"
                "Dec = %{customdata[1]:.6f} deg<br>"
                "X = %{x:.6f}<br>"
                "Y = %{y:.6f}<br>"
                "Z = %{z:.6f}"
                "<extra></extra>"
            ),
        )
    )

    # Estrella
    traces.append(
        go.Scatter3d(
            x=[star[0]],
            y=[star[1]],
            z=[star[2]],
            mode="markers+text",
            marker=dict(size=9, color="navy", symbol="circle"),
            text=["Estrella"],
            textposition="top center",
            name="Estrella sintética",
            customdata=[[star_ra, star_dec]],
            hovertemplate=(
                "<b>Estrella sintética</b><br>"
                "RA = %{customdata[0]:.6f} deg<br>"
                "Dec = %{customdata[1]:.6f} deg<br>"
                "X = %{x:.6f}<br>"
                "Y = %{y:.6f}<br>"
                "Z = %{z:.6f}"
                "<extra></extra>"
            ),
        )
    )

    # Trayectorias de movimiento propio
    def motion_trace(p0, p1, color, name):
        return go.Scatter3d(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            z=[p0[2], p1[2]],
            mode="lines+markers",
            line=dict(color=color, width=8),
            marker=dict(size=4, color=color),
            name=name,
            hoverinfo="skip",
        )

    traces.append(motion_trace(star, star_final_ideal, "limegreen", "Movimiento ideal"))
    traces.append(motion_trace(star, star_final_plus, "darkorange", "Movimiento +10°"))
    traces.append(motion_trace(star, star_final_minus, "mediumpurple", "Movimiento -10°"))

    # Función auxiliar para dibujar un vector director desde el origen
    def director_trace(pole, pole_ra, pole_dec, color, name):
        return [
            go.Scatter3d(
                x=[0.0, pole[0]],
                y=[0.0, pole[1]],
                z=[0.0, pole[2]],
                mode="lines",
                line=dict(color=color, width=6),
                name=name + " (vector)",
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter3d(
                x=[pole[0]],
                y=[pole[1]],
                z=[pole[2]],
                mode="markers+text",
                marker=dict(size=8, color=color),
                text=[name],
                textposition="top center",
                name=name,
                customdata=[[pole_ra, pole_dec]],
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "RA = %{customdata[0]:.6f} deg<br>"
                    "Dec = %{customdata[1]:.6f} deg<br>"
                    "X = %{x:.6f}<br>"
                    "Y = %{y:.6f}<br>"
                    "Z = %{z:.6f}"
                    "<extra></extra>"
                ),
            ),
        ]

    traces.extend(director_trace(pole_ideal, pole_ideal_ra, pole_ideal_dec, "limegreen", "Director ideal"))
    traces.extend(director_trace(pole_plus, pole_plus_ra, pole_plus_dec, "darkorange", "Director +10°"))
    traces.extend(director_trace(pole_minus, pole_minus_ra, pole_minus_dec, "mediumpurple", "Director -10°"))

    if title is None:
        title = "Experimento geométrico: ápex, λ y vectores directores"

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=title,
        width=1000,
        height=800,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        scene=dict(
            xaxis=dict(title="X", showbackground=False),
            yaxis=dict(title="Y", showbackground=False),
            zaxis=dict(title="Z", showbackground=False),
            aspectmode="data",
        ),
    )

    if show:
        fig.show()

    return fig

def plot_pole_scar_belt(
    df,
    apex_ra_deg,
    apex_dec_deg,
    belt_deg=1.0,
    point_size=0.35,
    alpha=0.45,
    figsize=(15, 4),
    return_selected=False,
):
    """
    Grafica los polos estelares dentro de una franja de ±belt_deg
    alrededor del círculo máximo definido por el ápex.

    El círculo máximo se rectifica para aparecer como una línea
    horizontal en residual angular = 0 deg.

    Parameters
    ----------
    df : pandas.DataFrame
        Debe contener:
        pole_x_unit, pole_y_unit, pole_z_unit.

    apex_ra_deg : float
        Ascensión recta del ápex en grados.

    apex_dec_deg : float
        Declinación del ápex en grados.

    belt_deg : float, default=1.0
        Semiancho del cinturón alrededor del círculo máximo.

    point_size : float
        Tamaño de los puntos.

    alpha : float
        Transparencia de los puntos.

    figsize : tuple
        Tamaño de la figura.

    return_selected : bool
        Si es True, devuelve también las filas que caen dentro
        del cinturón.

    Returns
    -------
    result : dict
        Diccionario con figura, ejes, coordenadas rectificadas,
        máscaras y métricas básicas.
    """

    required = [
        "pole_x_unit",
        "pole_y_unit",
        "pole_z_unit",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan las columnas necesarias: {missing}"
        )

    # ----------------------------------------------------------
    # 1. Extraer los polos unitarios
    # ----------------------------------------------------------
    poles_all = df[required].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    valid_mask = np.all(
        np.isfinite(poles_all),
        axis=1,
    )

    poles = poles_all[valid_mask]

    if poles.shape[0] == 0:
        raise ValueError(
            "No hay polos finitos en el DataFrame."
        )

    # Renormalización de seguridad
    norms = np.linalg.norm(poles, axis=1)

    norm_valid = (
        np.isfinite(norms)
        & (norms > 0)
    )

    poles = poles[norm_valid]
    poles = poles / norms[norm_valid, None]

    # Índices originales correspondientes
    original_positions = np.flatnonzero(valid_mask)
    original_positions = original_positions[norm_valid]

    # ----------------------------------------------------------
    # 2. Vector unitario del ápex
    # ----------------------------------------------------------
    ra_apex = np.radians(apex_ra_deg)
    dec_apex = np.radians(apex_dec_deg)

    apex_vector = np.array([
        np.cos(dec_apex) * np.cos(ra_apex),
        np.cos(dec_apex) * np.sin(ra_apex),
        np.sin(dec_apex),
    ])

    apex_vector /= np.linalg.norm(apex_vector)

    # ----------------------------------------------------------
    # 3. Base ortonormal dentro del plano del círculo máximo
    # ----------------------------------------------------------
    # Intentamos usar el polo celeste como referencia.
    reference = np.array([0.0, 0.0, 1.0])

    # Si el ápex está casi alineado con el polo celeste,
    # cambiamos el vector de referencia.
    if abs(np.dot(reference, apex_vector)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])

    axis_x = np.cross(reference, apex_vector)
    axis_x /= np.linalg.norm(axis_x)

    axis_y = np.cross(apex_vector, axis_x)
    axis_y /= np.linalg.norm(axis_y)

    # ----------------------------------------------------------
    # 4. Coordenadas rectificadas
    # ----------------------------------------------------------
    projection_x = poles @ axis_x
    projection_y = poles @ axis_y
    projection_normal = poles @ apex_vector

    # Coordenada a lo largo del círculo máximo
    longitude_along_belt_deg = np.degrees(
        np.arctan2(
            projection_y,
            projection_x,
        )
    )

    # Distancia angular firmada al círculo máximo
    pole_residual_deg = np.degrees(
        np.arcsin(
            np.clip(
                projection_normal,
                -1.0,
                1.0,
            )
        )
    )

    # ----------------------------------------------------------
    # 5. Selección del cinturón ±belt_deg
    # ----------------------------------------------------------
    belt_mask = np.abs(pole_residual_deg) <= belt_deg

    lon_belt = longitude_along_belt_deg[belt_mask]
    residual_belt = pole_residual_deg[belt_mask]

    # ----------------------------------------------------------
    # 6. Gráfica en blanco y negro
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        lon_belt,
        residual_belt,
        s=point_size,
        c="black",
        alpha=alpha,
        marker=".",
        linewidths=0,
        rasterized=True,
    )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-belt_deg, belt_deg)

    ax.set_xlabel(
        "Posición a lo largo del círculo máximo [deg]"
    )

    ax.set_ylabel(
        "Distancia angular al círculo máximo [deg]"
    )

    ax.set_title(
        f"Cinturón de polos: ±{belt_deg:.2f}° alrededor del círculo máximo\n"
        f"N total válido = {len(poles):,}   |   "
        f"N en cinturón = {belt_mask.sum():,}   |   "
        f"fracción = {belt_mask.mean():.4f}"
    )

    ax.grid(
        color="0.85",
        linewidth=0.5,
        alpha=0.7,
    )

    plt.tight_layout()

    # ----------------------------------------------------------
    # 7. Máscara en el tamaño original del DataFrame
    # ----------------------------------------------------------
    original_belt_mask = np.zeros(
        len(df),
        dtype=bool,
    )

    original_belt_positions = original_positions[belt_mask]
    original_belt_mask[original_belt_positions] = True

    result = {
        "fig": fig,
        "ax": ax,
        "apex_vector": apex_vector,
        "longitude_along_belt_deg": longitude_along_belt_deg,
        "pole_residual_deg": pole_residual_deg,
        "belt_mask_valid": belt_mask,
        "belt_mask_dataframe": original_belt_mask,
        "n_valid": len(poles),
        "n_in_belt": int(belt_mask.sum()),
        "fraction_in_belt": float(belt_mask.mean()),
    }

    if return_selected:
        result["df_belt"] = df.loc[
            original_belt_mask
        ].copy()

    return result