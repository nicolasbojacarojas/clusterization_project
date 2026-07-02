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