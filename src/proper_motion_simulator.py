from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


KM_S_PER_AU_YR = 4.74047


# Matriz estándar aproximada ICRS -> Galactic.
# Para velocidades se usa igual que para vectores de posición.
ICRS_TO_GALACTIC_MATRIX = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=float,
)

GALACTIC_TO_ICRS_MATRIX = ICRS_TO_GALACTIC_MATRIX.T


def radec_to_unit_vector(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> np.ndarray:
    """
    Convierte RA, Dec en grados a vectores unitarios ICRS.
    """

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    return np.column_stack(
        (
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        )
    )


def unit_vector_to_radec(vector: np.ndarray) -> tuple[float, float]:
    """
    Convierte un vector cartesiano ICRS a RA, Dec en grados.
    """

    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm <= 0.0 or not np.isfinite(norm):
        return np.nan, np.nan

    x, y, z = vector / norm

    ra_deg = np.degrees(np.arctan2(y, x)) % 360.0
    dec_deg = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))

    return float(ra_deg), float(dec_deg)


def uvw_to_icrs_velocity(
    u_kms: float,
    v_kms: float,
    w_kms: float,
) -> np.ndarray:
    """
    Convierte velocidades heliocéntricas galácticas U,V,W a vector ICRS.

    Convención usada:
        U positivo hacia el centro galáctico.
        V positivo en la dirección de rotación galáctica.
        W positivo hacia el polo norte galáctico.

    Si tu convención usa U positivo alejándose del centro galáctico,
    debes pasar -U.
    """

    velocity_galactic = np.array([u_kms, v_kms, w_kms], dtype=float)

    return GALACTIC_TO_ICRS_MATRIX @ velocity_galactic


def apex_to_icrs_velocity(
    apex_ra_deg: float,
    apex_dec_deg: float,
    speed_kms: float,
) -> np.ndarray:
    """
    Construye un vector de velocidad ICRS a partir de ápex RA/Dec
    y velocidad total.
    """

    apex_vector = radec_to_unit_vector(
        np.array([apex_ra_deg], dtype=float),
        np.array([apex_dec_deg], dtype=float),
    )[0]

    apex_vector = apex_vector / np.linalg.norm(apex_vector)

    return speed_kms * apex_vector


def _validate_triangular_log_error(
    name: str,
    log_error_range: Tuple[float, float],
    log_error_center: float,
) -> None:
    """
    Valida parámetros de una distribución triangular en log10(error).
    """

    log_min, log_max = log_error_range

    if log_min > log_max:
        raise ValueError(
            f"{name}_log_error_range debe satisfacer min <= max."
        )

    if not log_min <= log_error_center <= log_max:
        raise ValueError(
            f"{name}_log_error_center debe estar entre "
            f"{name}_log_error_range[0] y {name}_log_error_range[1]."
        )


def _draw_log10_triangular_error(
    rng: np.random.Generator,
    n_stars: int,
    log_error_range: Tuple[float, float],
    log_error_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dibuja errores desde:

        x ~ Triangular(left, mode, right)
        sigma = 10**x

    Returns
    -------
    log_error : np.ndarray
        Valores de log10(sigma).

    error : np.ndarray
        Valores de sigma en escala lineal.
    """

    log_min, log_max = log_error_range

    log_error = rng.triangular(
        left=log_min,
        mode=log_error_center,
        right=log_max,
        size=n_stars,
    )

    error = 10.0 ** log_error

    return log_error, error


def simulate_proper_motions_from_3d_velocity(
    df: pd.DataFrame,

    # Modo 1: velocidad definida por ápex + rapidez
    apex_ra_deg: Optional[float] = None,
    apex_dec_deg: Optional[float] = None,
    speed_kms: Optional[float] = None,

    # Modo 2: velocidad definida por U,V,W galácticos
    u_kms: Optional[float] = None,
    v_kms: Optional[float] = None,
    w_kms: Optional[float] = None,

    parallax_mas: Optional[float] = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
    parallax_col: str = "parallax",

    # ------------------------------------------------------------
    # Modelo de error para pmra
    # Por defecto:
    # log10(pmra_error) ~ Triangular(-3, -1.7, 0)
    # ------------------------------------------------------------
    pmra_log_error_range: Tuple[float, float] = (-3.0, 0.0),
    pmra_log_error_center: float = -1.7,

    # ------------------------------------------------------------
    # Modelo de error para pmdec
    #
    # "scaled_pmra":
    #     pmdec_error = pm_error_slope * pmra_error
    #
    # "independent_log10_triangular":
    #     log10(pmdec_error) se genera con su propia triangular.
    # ------------------------------------------------------------
    pmdec_error_model: Literal[
        "scaled_pmra",
        "independent_log10_triangular",
    ] = "scaled_pmra",

    pm_error_slope: float = 0.7,
    pm_error_slope_logscatter: float = 0.0,

    pmdec_log_error_range: Tuple[float, float] = (-3.0, 0.0),
    pmdec_log_error_center: float = -1.7,

    # Correlación entre las perturbaciones aleatorias de pmra y pmdec.
    # No es la pendiente entre pmra_error y pmdec_error.
    pmra_pmdec_corr: float = 0.2,

    # ------------------------------------------------------------
    # Modelo de error de paralaje
    # Solo se genera si apply_parallax_error=True.
    # ------------------------------------------------------------
    parallax_log_error_range: Tuple[float, float] = (-3.0, 0.0),
    parallax_log_error_center: float = -1.7,

    apply_pmra_error: bool = True,
    apply_pmdec_error: bool = True,
    apply_parallax_error: bool = False,

    seed: Optional[int] = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Simula movimientos propios Gaia para estrellas con posiciones existentes.

    Permite dos modos cinemáticos:

    1. Modo ápex:
        apex_ra_deg, apex_dec_deg, speed_kms

    2. Modo U,V,W:
        u_kms, v_kms, w_kms

    En modo U,V,W no se usa la coordenada del ápex como entrada.
    El ápex se calcula solo como diagnóstico a partir del vector de velocidad.

    Gaia usa:
        pmra  = mu_alpha* = mu_alpha cos(dec)
        pmdec = mu_delta

    Modelo Gaia-like de errores en movimiento propio:

        log10(pmra_error) ~ Triangular(-3, -1.7, 0)

    y, por defecto:

        pmdec_error = 0.7 * pmra_error

    Las perturbaciones de pmra y pmdec se generan con una normal bivariada
    usando pmra_pmdec_corr.

    Los errores solo se generan y aplican si apply_pmra_error=True o
    apply_pmdec_error=True. Si ambos son False, el cúmulo queda perfecto
    en movimientos propios.
    """

    required = [ra_col, dec_col]

    if parallax_mas is None:
        required.append(parallax_col)

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------
    # Validación de modo cinemático
    # ------------------------------------------------------------

    apex_values = [apex_ra_deg, apex_dec_deg, speed_kms]
    uvw_values = [u_kms, v_kms, w_kms]

    apex_any = any(value is not None for value in apex_values)
    apex_complete = all(value is not None for value in apex_values)

    uvw_any = any(value is not None for value in uvw_values)
    uvw_complete = all(value is not None for value in uvw_values)

    if apex_any and not apex_complete:
        raise ValueError(
            "Modo ápex incompleto. Debes pasar apex_ra_deg, "
            "apex_dec_deg y speed_kms."
        )

    if uvw_any and not uvw_complete:
        raise ValueError(
            "Modo U,V,W incompleto. Debes pasar u_kms, v_kms y w_kms."
        )

    if apex_complete and uvw_complete:
        raise ValueError(
            "Entrada cinemática ambigua. Usa ápex+speed o U,V,W, "
            "pero no ambos a la vez."
        )

    if not apex_complete and not uvw_complete:
        raise ValueError(
            "Debes definir la velocidad con apex_ra_deg, apex_dec_deg, "
            "speed_kms o con u_kms, v_kms, w_kms."
        )

    # ------------------------------------------------------------
    # Validación de parámetros de errores
    # ------------------------------------------------------------

    if apply_pmra_error or apply_pmdec_error:
        _validate_triangular_log_error(
            name="pmra",
            log_error_range=pmra_log_error_range,
            log_error_center=pmra_log_error_center,
        )

        if pmdec_error_model not in {
            "scaled_pmra",
            "independent_log10_triangular",
        }:
            raise ValueError(
                "pmdec_error_model debe ser 'scaled_pmra' o "
                "'independent_log10_triangular'."
            )

        if pmdec_error_model == "independent_log10_triangular":
            _validate_triangular_log_error(
                name="pmdec",
                log_error_range=pmdec_log_error_range,
                log_error_center=pmdec_log_error_center,
            )

        if pm_error_slope <= 0.0:
            raise ValueError("pm_error_slope debe ser positivo.")

        if pm_error_slope_logscatter < 0.0:
            raise ValueError("pm_error_slope_logscatter debe ser >= 0.")

        if not -1.0 <= pmra_pmdec_corr <= 1.0:
            raise ValueError("pmra_pmdec_corr debe estar entre -1 y 1.")

    if apply_parallax_error:
        _validate_triangular_log_error(
            name="parallax",
            log_error_range=parallax_log_error_range,
            log_error_center=parallax_log_error_center,
        )

    # ------------------------------------------------------------
    # Construcción del vector de velocidad ICRS
    # ------------------------------------------------------------

    if apex_complete:
        velocity_mode = "apex"
        velocity_icrs_kms = apex_to_icrs_velocity(
            apex_ra_deg=apex_ra_deg,
            apex_dec_deg=apex_dec_deg,
            speed_kms=speed_kms,
        )
    else:
        velocity_mode = "uvw"
        velocity_icrs_kms = uvw_to_icrs_velocity(
            u_kms=u_kms,
            v_kms=v_kms,
            w_kms=w_kms,
        )

    speed_norm_kms = float(np.linalg.norm(velocity_icrs_kms))

    if speed_norm_kms > 0.0:
        velocity_unit = velocity_icrs_kms / speed_norm_kms
        derived_apex_ra_deg, derived_apex_dec_deg = unit_vector_to_radec(
            velocity_unit
        )
    else:
        velocity_unit = np.full(3, np.nan)
        derived_apex_ra_deg = np.nan
        derived_apex_dec_deg = np.nan

    velocity_galactic_kms = ICRS_TO_GALACTIC_MATRIX @ velocity_icrs_kms

    rng = np.random.default_rng(seed)

    result = df.copy() if copy else df

    ra_deg = result[ra_col].to_numpy(dtype=float)
    dec_deg = result[dec_col].to_numpy(dtype=float)

    n_stars = len(result)

    # ------------------------------------------------------------
    # Paralaje verdadera
    # ------------------------------------------------------------

    if parallax_mas is None:
        parallax_true = result[parallax_col].to_numpy(dtype=float)
    else:
        parallax_true = np.full(n_stars, float(parallax_mas), dtype=float)

    valid_parallax_true = np.isfinite(parallax_true) & (parallax_true > 0.0)

    distance_pc_true = np.full(n_stars, np.nan, dtype=float)
    distance_pc_true[valid_parallax_true] = (
        1000.0 / parallax_true[valid_parallax_true]
    )

    result["parallax_true"] = parallax_true
    result["distance_pc_true"] = distance_pc_true

    # ------------------------------------------------------------
    # Errores de paralaje
    # Solo se generan si apply_parallax_error=True
    # ------------------------------------------------------------

    if apply_parallax_error:
        parallax_log_error, parallax_error_mas = _draw_log10_triangular_error(
            rng=rng,
            n_stars=n_stars,
            log_error_range=parallax_log_error_range,
            log_error_center=parallax_log_error_center,
        )

        parallax_noise_mas = rng.normal(
            loc=0.0,
            scale=parallax_error_mas,
            size=n_stars,
        )
    else:
        parallax_log_error = np.full(n_stars, np.nan, dtype=float)
        parallax_error_mas = np.zeros(n_stars, dtype=float)
        parallax_noise_mas = np.zeros(n_stars, dtype=float)

    parallax_observed = parallax_true + parallax_noise_mas

    result["parallax_log_error"] = parallax_log_error
    result["parallax_error_mas"] = parallax_error_mas
    result["parallax_noise_mas"] = parallax_noise_mas
    result[parallax_col] = parallax_observed

    # ------------------------------------------------------------
    # Vectores unitarios de posición
    # ------------------------------------------------------------

    star_vec = radec_to_unit_vector(ra_deg, dec_deg)

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

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

    # ------------------------------------------------------------
    # Proyección de la velocidad 3D sobre la línea de visión
    # y sobre el plano tangente
    # ------------------------------------------------------------

    radial_velocity_kms_true = star_vec @ velocity_icrs_kms

    velocity_ra_kms_true = e_ra @ velocity_icrs_kms
    velocity_dec_kms_true = e_dec @ velocity_icrs_kms

    vt_kms_true = np.sqrt(
        velocity_ra_kms_true**2 + velocity_dec_kms_true**2
    )

    pmra_true = np.full(n_stars, np.nan, dtype=float)
    pmdec_true = np.full(n_stars, np.nan, dtype=float)

    pmra_true[valid_parallax_true] = (
        1000.0
        * velocity_ra_kms_true[valid_parallax_true]
        / (
            KM_S_PER_AU_YR
            * distance_pc_true[valid_parallax_true]
        )
    )

    pmdec_true[valid_parallax_true] = (
        1000.0
        * velocity_dec_kms_true[valid_parallax_true]
        / (
            KM_S_PER_AU_YR
            * distance_pc_true[valid_parallax_true]
        )
    )

    mu_total_masyr_true = np.sqrt(pmra_true**2 + pmdec_true**2)

    # ------------------------------------------------------------
    # Lambda respecto al ápex efectivo de la velocidad
    # ------------------------------------------------------------

    if speed_norm_kms > 0.0:
        cos_lambda = np.clip(star_vec @ velocity_unit, -1.0, 1.0)
        sin_lambda = np.sqrt(np.clip(1.0 - cos_lambda**2, 0.0, 1.0))
        lambda_rad = np.arctan2(sin_lambda, cos_lambda)
        lambda_deg = np.rad2deg(lambda_rad)
    else:
        cos_lambda = np.full(n_stars, np.nan, dtype=float)
        sin_lambda = np.full(n_stars, np.nan, dtype=float)
        lambda_deg = np.full(n_stars, np.nan, dtype=float)

    # ------------------------------------------------------------
    # Guardar diagnósticos cinemáticos
    # ------------------------------------------------------------

    result["velocity_mode"] = velocity_mode

    result["velocity_icrs_x_kms"] = velocity_icrs_kms[0]
    result["velocity_icrs_y_kms"] = velocity_icrs_kms[1]
    result["velocity_icrs_z_kms"] = velocity_icrs_kms[2]

    result["u_kms"] = velocity_galactic_kms[0]
    result["v_kms"] = velocity_galactic_kms[1]
    result["w_kms"] = velocity_galactic_kms[2]

    result["speed_kms"] = speed_norm_kms
    result["apex_ra_deg"] = derived_apex_ra_deg
    result["apex_dec_deg"] = derived_apex_dec_deg

    result["cos_lambda"] = cos_lambda
    result["sin_lambda"] = sin_lambda
    result["lambda_deg"] = lambda_deg

    result["radial_velocity_kms_true"] = radial_velocity_kms_true
    result["velocity_ra_kms_true"] = velocity_ra_kms_true
    result["velocity_dec_kms_true"] = velocity_dec_kms_true
    result["vt_kms_true"] = vt_kms_true
    result["mu_total_masyr_true"] = mu_total_masyr_true

    result["pmra_true"] = pmra_true
    result["pmdec_true"] = pmdec_true

    # ------------------------------------------------------------
    # Errores de movimientos propios Gaia-like
    #
    # Solo se generan si apply_pmra_error=True o apply_pmdec_error=True.
    #
    # pmra:
    #   log10(pmra_error) ~ Triangular(-3, -1.7, 0)
    #
    # pmdec por defecto:
    #   pmdec_error = 0.7 * pmra_error
    #
    # Ruido:
    #   normal bivariada con correlación pmra_pmdec_corr
    # ------------------------------------------------------------

    if apply_pmra_error or apply_pmdec_error:
        pmra_log_error, pmra_error_masyr = _draw_log10_triangular_error(
            rng=rng,
            n_stars=n_stars,
            log_error_range=pmra_log_error_range,
            log_error_center=pmra_log_error_center,
        )

        if pmdec_error_model == "scaled_pmra":
            pmdec_error_masyr = pm_error_slope * pmra_error_masyr

            if pm_error_slope_logscatter > 0.0:
                slope_factor = np.exp(
                    rng.normal(
                        loc=0.0,
                        scale=pm_error_slope_logscatter,
                        size=n_stars,
                    )
                )

                pmdec_error_masyr *= slope_factor

            pmdec_log_error = np.log10(pmdec_error_masyr)

        elif pmdec_error_model == "independent_log10_triangular":
            pmdec_log_error, pmdec_error_masyr = (
                _draw_log10_triangular_error(
                    rng=rng,
                    n_stars=n_stars,
                    log_error_range=pmdec_log_error_range,
                    log_error_center=pmdec_log_error_center,
                )
            )

        else:
            raise ValueError(
                "pmdec_error_model debe ser 'scaled_pmra' o "
                "'independent_log10_triangular'."
            )

        rho = np.clip(pmra_pmdec_corr, -0.999, 0.999)

        z1 = rng.normal(size=n_stars)
        z2 = rng.normal(size=n_stars)

        pmra_noise_candidate = pmra_error_masyr * z1

        pmdec_noise_candidate = pmdec_error_masyr * (
            rho * z1 + np.sqrt(1.0 - rho**2) * z2
        )

        if apply_pmra_error:
            pmra_noise_masyr = pmra_noise_candidate
        else:
            pmra_noise_masyr = np.zeros(n_stars, dtype=float)

        if apply_pmdec_error:
            pmdec_noise_masyr = pmdec_noise_candidate
        else:
            pmdec_noise_masyr = np.zeros(n_stars, dtype=float)

        pm_corr_array = np.full(n_stars, rho, dtype=float)

    else:
        pmra_log_error = np.full(n_stars, np.nan, dtype=float)
        pmdec_log_error = np.full(n_stars, np.nan, dtype=float)

        pmra_error_masyr = np.zeros(n_stars, dtype=float)
        pmdec_error_masyr = np.zeros(n_stars, dtype=float)

        pmra_noise_masyr = np.zeros(n_stars, dtype=float)
        pmdec_noise_masyr = np.zeros(n_stars, dtype=float)

        pm_corr_array = np.zeros(n_stars, dtype=float)

    result["pmra_log_error"] = pmra_log_error
    result["pmdec_log_error"] = pmdec_log_error

    result["pmra_error_masyr"] = pmra_error_masyr
    result["pmdec_error_masyr"] = pmdec_error_masyr
    result["pmra_pmdec_corr"] = pm_corr_array

    result["pmra_noise_masyr"] = pmra_noise_masyr
    result["pmdec_noise_masyr"] = pmdec_noise_masyr

    result["pmra"] = pmra_true + pmra_noise_masyr
    result["pmdec"] = pmdec_true + pmdec_noise_masyr

    return result