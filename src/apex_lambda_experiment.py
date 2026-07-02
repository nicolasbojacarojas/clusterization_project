from __future__ import annotations

from typing import Dict, Any, Optional, Literal, Tuple

import numpy as np

ProjectionType = Literal["sphere", "mollweide", "radec"]


# ============================================================
# Utilidades base
# ============================================================

def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("No se puede normalizar el vector.")
    return v / norm


def radec_to_xyz(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def xyz_to_radec(v: np.ndarray) -> Tuple[float, float]:
    v = normalize_vector(v)
    x, y, z = v

    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))

    return float(ra), float(dec)


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def build_orthonormal_basis_from_pole(pole: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye dos vectores ortonormales perpendiculares al vector `pole`.
    """
    pole = normalize_vector(pole)

    aux = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(aux, pole)) > 0.9:
        aux = np.array([0.0, 1.0, 0.0])

    e1 = normalize_vector(np.cross(pole, aux))
    e2 = normalize_vector(np.cross(pole, e1))

    return e1, e2


# ============================================================
# Geometría del experimento
# ============================================================

def generate_lambda_circle(
    apex_vector: np.ndarray,
    lambda_deg: float,
    n_points: int = 361,
) -> np.ndarray:
    """
    Genera el círculo de posibles posiciones de estrellas que están
    a un ángulo lambda del ápex.
    """
    apex_vector = normalize_vector(apex_vector)
    lam = np.deg2rad(lambda_deg)

    e1, e2 = build_orthonormal_basis_from_pole(apex_vector)

    phi = np.linspace(0.0, 2.0 * np.pi, n_points)

    circle = (
        np.cos(lam) * apex_vector[None, :]
        + np.sin(lam) * (
            np.cos(phi)[:, None] * e1[None, :]
            + np.sin(phi)[:, None] * e2[None, :]
        )
    )

    return circle


def generate_star_at_lambda(
    apex_vector: np.ndarray,
    lambda_deg: float,
    phase_deg: float,
) -> np.ndarray:
    """
    Genera una estrella sintética sobre el círculo definido por lambda
    alrededor del ápex. `phase_deg` controla en qué punto del círculo cae.
    """
    apex_vector = normalize_vector(apex_vector)
    lam = np.deg2rad(lambda_deg)
    phi = np.deg2rad(phase_deg)

    e1, e2 = build_orthonormal_basis_from_pole(apex_vector)

    star = (
        np.cos(lam) * apex_vector
        + np.sin(lam) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    )

    return normalize_vector(star)


def tangent_towards_apex(
    star_vector: np.ndarray,
    apex_vector: np.ndarray,
) -> np.ndarray:
    """
    Dirección tangente ideal en la estrella, apuntando hacia el ápex.
    """
    x = normalize_vector(star_vector)
    a = normalize_vector(apex_vector)

    t = a - np.dot(a, x) * x
    return normalize_vector(t)


def rotate_tangent_in_tangent_plane(
    star_vector: np.ndarray,
    tangent_vector: np.ndarray,
    delta_deg: float,
) -> np.ndarray:
    """
    Rota una dirección tangente dentro del plano tangente local de la estrella.
    """
    x = normalize_vector(star_vector)
    t = normalize_vector(tangent_vector)
    delta = np.deg2rad(delta_deg)

    # Dirección tangente perpendicular dentro del plano tangente
    u = normalize_vector(np.cross(x, t))

    t_rot = np.cos(delta) * t + np.sin(delta) * u
    return normalize_vector(t_rot)


def move_on_sphere(
    start_vector: np.ndarray,
    tangent_vector: np.ndarray,
    step_deg: float,
) -> np.ndarray:
    """
    Mueve un punto sobre la esfera siguiendo una dirección tangente,
    a una distancia angular `step_deg`.
    """
    x = normalize_vector(start_vector)
    t = normalize_vector(tangent_vector)
    step = np.deg2rad(step_deg)

    xf = np.cos(step) * x + np.sin(step) * t
    return normalize_vector(xf)


def director_vector_from_motion(
    star_vector: np.ndarray,
    tangent_vector: np.ndarray,
) -> np.ndarray:
    """
    Vector director / polo del círculo máximo definido por la estrella
    y su dirección de movimiento.
    """
    x = normalize_vector(star_vector)
    t = normalize_vector(tangent_vector)

    p = np.cross(x, t)
    return normalize_vector(p)


def great_circle_perpendicular_to_vector(
    normal_vector: np.ndarray,
    n_points: int = 361,
) -> np.ndarray:
    """
    Genera el ecuador asociado al ápex: el círculo máximo perpendicular
    al vector del ápex.
    """
    n = normalize_vector(normal_vector)
    e1, e2 = build_orthonormal_basis_from_pole(n)

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = (
        np.cos(theta)[:, None] * e1[None, :]
        + np.sin(theta)[:, None] * e2[None, :]
    )

    return circle


# ============================================================
# Armado completo del experimento
# ============================================================

def build_apex_lambda_experiment(
    *,
    apex_ra_deg: float,
    apex_dec_deg: float,
    lambda_deg: float,
    phase_deg: float = 0.0,
    deviation_deg: float = 10.0,
    motion_step_deg: float = 12.0,
    n_circle_points: int = 361,
) -> Dict[str, Any]:
    """
    Construye todo el experimento geométrico:
        - ápex
        - estrella sintética a ángulo lambda
        - dirección ideal hacia el ápex
        - desviaciones +delta y -delta
        - vectores directores / polos
        - ecuador del ápex
        - círculo de posibles posiciones de la estrella
    """

    if not (0.0 < lambda_deg < 180.0):
        raise ValueError("lambda_deg debe estar entre 0 y 180 grados.")
    if motion_step_deg <= 0:
        raise ValueError("motion_step_deg debe ser positivo.")

    apex = radec_to_xyz(apex_ra_deg, apex_dec_deg)

    star = generate_star_at_lambda(
        apex_vector=apex,
        lambda_deg=lambda_deg,
        phase_deg=phase_deg,
    )

    lambda_circle = generate_lambda_circle(
        apex_vector=apex,
        lambda_deg=lambda_deg,
        n_points=n_circle_points,
    )

    apex_equator = great_circle_perpendicular_to_vector(
        normal_vector=apex,
        n_points=n_circle_points,
    )

    # Dirección ideal
    t_ideal = tangent_towards_apex(star, apex)

    # Direcciones desviadas
    t_plus = rotate_tangent_in_tangent_plane(
        star_vector=star,
        tangent_vector=t_ideal,
        delta_deg=deviation_deg,
    )
    t_minus = rotate_tangent_in_tangent_plane(
        star_vector=star,
        tangent_vector=t_ideal,
        delta_deg=-deviation_deg,
    )

    # Puntos finales al moverse un pequeño paso sobre la esfera
    star_final_ideal = move_on_sphere(star, t_ideal, motion_step_deg)
    star_final_plus = move_on_sphere(star, t_plus, motion_step_deg)
    star_final_minus = move_on_sphere(star, t_minus, motion_step_deg)

    # Vectores directores / polos
    pole_ideal = director_vector_from_motion(star, t_ideal)
    pole_plus = director_vector_from_motion(star, t_plus)
    pole_minus = director_vector_from_motion(star, t_minus)

    # Conversión a RA/Dec
    star_ra_deg, star_dec_deg = xyz_to_radec(star)
    pole_ideal_ra_deg, pole_ideal_dec_deg = xyz_to_radec(pole_ideal)
    pole_plus_ra_deg, pole_plus_dec_deg = xyz_to_radec(pole_plus)
    pole_minus_ra_deg, pole_minus_dec_deg = xyz_to_radec(pole_minus)

    return {
        "inputs": {
            "apex_ra_deg": apex_ra_deg,
            "apex_dec_deg": apex_dec_deg,
            "lambda_deg": lambda_deg,
            "phase_deg": phase_deg,
            "deviation_deg": deviation_deg,
            "motion_step_deg": motion_step_deg,
        },
        "apex": {
            "vector": apex,
            "ra_deg": apex_ra_deg,
            "dec_deg": apex_dec_deg,
        },
        "star": {
            "vector": star,
            "ra_deg": star_ra_deg,
            "dec_deg": star_dec_deg,
        },
        "lambda_circle_xyz": lambda_circle,
        "apex_equator_xyz": apex_equator,
        "motions": {
            "ideal": {
                "tangent_vector": t_ideal,
                "final_star_vector": star_final_ideal,
                "final_star_radec": xyz_to_radec(star_final_ideal),
                "pole_vector": pole_ideal,
                "pole_ra_deg": pole_ideal_ra_deg,
                "pole_dec_deg": pole_ideal_dec_deg,
            },
            "plus": {
                "tangent_vector": t_plus,
                "final_star_vector": star_final_plus,
                "final_star_radec": xyz_to_radec(star_final_plus),
                "pole_vector": pole_plus,
                "pole_ra_deg": pole_plus_ra_deg,
                "pole_dec_deg": pole_plus_dec_deg,
            },
            "minus": {
                "tangent_vector": t_minus,
                "final_star_vector": star_final_minus,
                "final_star_radec": xyz_to_radec(star_final_minus),
                "pole_vector": pole_minus,
                "pole_ra_deg": pole_plus_ra_deg if False else pole_minus_ra_deg,
                "pole_dec_deg": pole_minus_dec_deg,
            },
        },
        "checks": {
            "angle_apex_star_deg": angle_between_vectors_deg(apex, star),
            "ideal_pole_dot_apex": float(np.dot(pole_ideal, apex)),
            "plus_pole_dot_apex": float(np.dot(pole_plus, apex)),
            "minus_pole_dot_apex": float(np.dot(pole_minus, apex)),
        },
    }