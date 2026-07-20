"""
open_cluster_pipeline.py

Pipeline unificado para:

1. Simular un cúmulo abierto en posiciones y velocidades 3D.
2. Proyectar el espacio de fases a observables tipo Gaia.
3. Añadir errores observacionales.
4. Reconstruir velocidades cartesianas ICRS.
5. Estimar el ápex cinemático.
6. Evaluar la incertidumbre del ápex mediante Monte Carlo.

Convención cartesiana ICRS
--------------------------
x: dirección RA = 0 deg, Dec = 0 deg
y: dirección RA = 90 deg, Dec = 0 deg
z: dirección Dec = 90 deg

Unidades
--------
Posición: pc
Velocidad: km/s
RA, Dec: grados
Paralaje: mas
Movimiento propio: mas/año
Velocidad radial: km/s
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord


# ============================================================
# CONSTANTES
# ============================================================

KM_S_PER_ARCSEC_YR_PC = 4.74047

__all__ = [
    "ClusterSimulationConfig",
    "MonteCarloConfig",
    "OpenClusterPipeline",
    "simulate_cluster",
    "reconstruct_icrs_velocities",
    "estimate_apex_from_velocities",
    "angular_separation_deg",
    "analyze_catalog_apex",
    "run_apex_monte_carlo",
    "summarize_monte_carlo",
    "save_catalog",
    "save_monte_carlo_results",
]


# ============================================================
# CONFIGURACIONES
# ============================================================

@dataclass(frozen=True)
class ClusterSimulationConfig:
    """
    Configuración física y observacional de la simulación.

    Debe proporcionarse exactamente un sistema de coordenadas para
    el centro del cúmulo y exactamente uno para el ápex.
    """

    # --------------------------------------------------------
    # Propiedades generales
    # --------------------------------------------------------

    n_members: int
    distance_pc: float
    radius_pc: float
    speed_kms: float

    # --------------------------------------------------------
    # Centro del cúmulo: ICRS
    # --------------------------------------------------------

    center_ra_deg: Optional[float] = None
    center_dec_deg: Optional[float] = None

    # --------------------------------------------------------
    # Centro del cúmulo: galácticas
    # --------------------------------------------------------

    center_l_deg: Optional[float] = None
    center_b_deg: Optional[float] = None

    # --------------------------------------------------------
    # Ápex: ICRS
    # --------------------------------------------------------

    apex_ra_deg: Optional[float] = None
    apex_dec_deg: Optional[float] = None

    # --------------------------------------------------------
    # Ápex: galácticas
    # --------------------------------------------------------

    apex_l_deg: Optional[float] = None
    apex_b_deg: Optional[float] = None

    # --------------------------------------------------------
    # Modelo cinemático
    # --------------------------------------------------------

    # Dispersión gaussiana isotrópica por componente:
    # delta_vx, delta_vy, delta_vz ~ N(0, sigma_v^2)
    velocity_dispersion_1d_kms: float = 0.0

    # Dispersión opcional de la rapidez sistémica entre realizaciones.
    # Normalmente debe dejarse en cero si se usa
    # velocity_dispersion_1d_kms.
    bulk_speed_sigma_kms: float = 0.0

    # Dispersión angular opcional del ápex entre realizaciones.
    apex_lon_sigma_deg: float = 0.0
    apex_lat_sigma_deg: float = 0.0

    # --------------------------------------------------------
    # Errores observacionales
    # --------------------------------------------------------

    parallax_error_mas: float = 0.0
    radial_velocity_error_kms: float = 0.0
    position_error_mas: float = 0.0

    # Compatibilidad: error común de PM.
    proper_motion_error_masyr: float = 0.0

    pmra_error_masyr: Optional[float] = None
    pmdec_error_masyr: Optional[float] = None

    pmra_error_distribution: Literal[
        "constant",
        "log10_triangular",
    ] = "constant"

    pmra_error_log10_left: float = -3.0
    pmra_error_log10_mode: float = -1.7
    pmra_error_log10_right: float = 0.0

    # sigma_pmdec = slope * sigma_pmra
    pm_error_slope: Optional[float] = None

    # Correlación de los errores observacionales de PM.
    pmra_pmdec_corr: float = 0.0

    # Dispersión estrella a estrella de los errores de PM.
    proper_motion_error_logscatter: float = 0.0

    # Dispersión alrededor de la relación sigma_pmdec/sigma_pmra.
    pm_error_slope_logscatter: float = 0.0

    # --------------------------------------------------------
    # Control
    # --------------------------------------------------------

    seed: Optional[int] = None
    include_true_values: bool = True


@dataclass(frozen=True)
class MonteCarloConfig:
    """
    Configuración para el análisis Monte Carlo del ápex.
    """

    n_iterations: int = 100
    seed: Optional[int] = None

    # True: se genera un nuevo cúmulo físico en cada iteración.
    resample_intrinsic_kinematics: bool = True

    # True: se generan nuevos errores observacionales.
    resample_observational_noise: bool = True

    # Método para combinar las velocidades estelares.
    apex_method: Literal["mean", "median"] = "mean"

    # Elimina estrellas que no permiten reconstrucción física.
    remove_invalid_sources: bool = True


# ============================================================
# FUNCIONES GEOMÉTRICAS
# ============================================================

def spherical_to_unit_vector(
    lon_deg: float | np.ndarray,
    lat_deg: float | np.ndarray,
) -> np.ndarray:
    """
    Convierte longitud y latitud esféricas en vectores unitarios.
    """

    lon_rad = np.radians(np.asarray(lon_deg, dtype=float))
    lat_rad = np.radians(np.asarray(lat_deg, dtype=float))

    cos_lat = np.cos(lat_rad)

    return np.stack(
        (
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        ),
        axis=-1,
    )


def unit_vector_to_spherical(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte vectores cartesianos en longitud y latitud.
    """

    vectors = np.asarray(vectors, dtype=float)

    norm = np.linalg.norm(vectors, axis=-1)

    if np.any(norm <= 0.0):
        raise ValueError("No se pueden convertir vectores de norma cero.")

    unit = vectors / np.expand_dims(norm, axis=-1)

    lon = np.degrees(
        np.arctan2(unit[..., 1], unit[..., 0])
    ) % 360.0

    lat = np.degrees(
        np.arcsin(np.clip(unit[..., 2], -1.0, 1.0))
    )

    return lon, lat


def angular_separation_deg(
    lon1_deg: float,
    lat1_deg: float,
    lon2_deg: float,
    lat2_deg: float,
) -> float:
    """
    Separación angular esférica entre dos direcciones.
    """

    vector1 = spherical_to_unit_vector(lon1_deg, lat1_deg)
    vector2 = spherical_to_unit_vector(lon2_deg, lat2_deg)

    dot_product = float(np.dot(vector1, vector2))
    dot_product = np.clip(dot_product, -1.0, 1.0)

    return float(np.degrees(np.arccos(dot_product)))


def icrs_to_galactic(
    ra_deg: float,
    dec_deg: float,
) -> tuple[float, float]:
    """
    Convierte coordenadas ICRS a galácticas.
    """

    coordinate = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        frame="icrs",
    )

    return (
        float(coordinate.galactic.l.deg),
        float(coordinate.galactic.b.deg),
    )


def galactic_to_icrs(
    l_deg: float,
    b_deg: float,
) -> tuple[float, float]:
    """
    Convierte coordenadas galácticas a ICRS.
    """

    coordinate = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        frame="galactic",
    ).icrs

    return float(coordinate.ra.deg), float(coordinate.dec.deg)


# ============================================================
# SIMULADOR
# ============================================================

class OpenClusterPipeline:
    """
    Simulador y analizador cinemático de cúmulos abiertos.
    """

    def __init__(
        self,
        config: ClusterSimulationConfig,
    ) -> None:

        self.config = config
        self._validate_config()

        self.rng = np.random.default_rng(config.seed)

    # ========================================================
    # INTERFAZ PÚBLICA
    # ========================================================

    def simulate(
        self,
        add_noise: bool = True,
    ) -> pd.DataFrame:
        """
        Ejecuta la simulación completa.

        Parameters
        ----------
        add_noise
            Si es True, añade errores observacionales.

        Returns
        -------
        pandas.DataFrame
            Catálogo simulado.
        """

        positions_pc = self.build_cluster_positions()
        velocities_kms = self.build_cluster_velocities()

        catalog = self.phase_space_to_gaia_observables(
            positions_pc=positions_pc,
            velocities_kms=velocities_kms,
        )

        catalog.insert(
            0,
            "source_id",
            np.arange(1, len(catalog) + 1, dtype=np.int64),
        )

        # Valores cartesianos verdaderos.
        catalog["x_pc_true"] = positions_pc[:, 0]
        catalog["y_pc_true"] = positions_pc[:, 1]
        catalog["z_pc_true"] = positions_pc[:, 2]

        catalog["vx_kms_true"] = velocities_kms[:, 0]
        catalog["vy_kms_true"] = velocities_kms[:, 1]
        catalog["vz_kms_true"] = velocities_kms[:, 2]

        catalog["speed_kms_true"] = np.linalg.norm(
            velocities_kms,
            axis=1,
        )

        # Conservamos una copia explícita de los observables verdaderos.
        true_observable_columns = [
            "ra",
            "dec",
            "parallax",
            "pmra",
            "pmdec",
        ]

        for column in true_observable_columns:
            catalog[f"{column}_true"] = catalog[column].to_numpy()

        if add_noise:
            catalog = self.add_observational_noise(catalog)
        else:
            catalog = self.add_zero_error_columns(catalog)

        if not self.config.include_true_values:
            true_columns = [
                column
                for column in catalog.columns
                if column.endswith("_true")
            ]

            catalog = catalog.drop(columns=true_columns)

        return catalog

    def build_cluster_positions(self) -> np.ndarray:
        """
        Genera posiciones distribuidas uniformemente en volumen
        dentro de una esfera de radio radius_pc.
        """

        n_members = self.config.n_members

        # Direcciones isotrópicas.
        directions = self.rng.normal(
            size=(n_members, 3),
        )

        directions /= np.linalg.norm(
            directions,
            axis=1,
            keepdims=True,
        )

        # Para uniformidad volumétrica: r ~ U(0,1)^(1/3).
        radii = (
            self.config.radius_pc
            * self.rng.random(n_members) ** (1.0 / 3.0)
        )

        offsets_pc = directions * radii[:, None]

        center = self.get_center_coord_icrs()

        center_vector_pc = np.array(
            [
                center.cartesian.x.to_value(u.pc),
                center.cartesian.y.to_value(u.pc),
                center.cartesian.z.to_value(u.pc),
            ],
            dtype=float,
        )

        return center_vector_pc[None, :] + offsets_pc

    def build_cluster_velocities(self) -> np.ndarray:
        """
        Genera velocidades como:

            v_i = V_bulk + delta_v_i

        donde delta_v_i es una dispersión gaussiana isotrópica.
        """

        n_members = self.config.n_members

        apex_ra_deg, apex_dec_deg = self.sample_apex_icrs()

        apex_unit = spherical_to_unit_vector(
            apex_ra_deg,
            apex_dec_deg,
        )

        bulk_speed = self.config.speed_kms

        if self.config.bulk_speed_sigma_kms > 0.0:
            bulk_speed = self.rng.normal(
                loc=self.config.speed_kms,
                scale=self.config.bulk_speed_sigma_kms,
            )

        if bulk_speed <= 0.0:
            raise ValueError(
                "La rapidez sistémica muestreada debe ser positiva."
            )

        bulk_velocity = bulk_speed * apex_unit

        internal_velocities = self.rng.normal(
            loc=0.0,
            scale=self.config.velocity_dispersion_1d_kms,
            size=(n_members, 3),
        )

        return bulk_velocity[None, :] + internal_velocities

    def phase_space_to_gaia_observables(
        self,
        positions_pc: np.ndarray,
        velocities_kms: np.ndarray,
    ) -> pd.DataFrame:
        """
        Proyecta posiciones y velocidades cartesianas ICRS
        a observables tipo Gaia.
        """

        positions_pc = np.asarray(positions_pc, dtype=float)
        velocities_kms = np.asarray(velocities_kms, dtype=float)

        if positions_pc.shape != velocities_kms.shape:
            raise ValueError(
                "positions_pc y velocities_kms deben tener "
                "la misma forma."
            )

        if positions_pc.ndim != 2 or positions_pc.shape[1] != 3:
            raise ValueError(
                "Los arreglos deben tener forma (n_sources, 3)."
            )

        x_coord = positions_pc[:, 0]
        y_coord = positions_pc[:, 1]
        z_coord = positions_pc[:, 2]

        distance_pc = np.linalg.norm(
            positions_pc,
            axis=1,
        )

        if np.any(distance_pc <= 0.0):
            raise ValueError(
                "Todas las fuentes deben tener distancia positiva."
            )

        unit_position = positions_pc / distance_pc[:, None]

        ra_rad = np.arctan2(
            y_coord,
            x_coord,
        ) % (2.0 * np.pi)

        dec_rad = np.arcsin(
            np.clip(
                z_coord / distance_pc,
                -1.0,
                1.0,
            )
        )

        basis_ra, basis_dec = _build_tangent_basis(
            ra_rad,
            dec_rad,
        )

        velocity_ra_kms = np.sum(
            velocities_kms * basis_ra,
            axis=1,
        )

        velocity_dec_kms = np.sum(
            velocities_kms * basis_dec,
            axis=1,
        )

        radial_velocity_kms = np.sum(
            velocities_kms * unit_position,
            axis=1,
        )

        pmra_masyr = (
            1000.0
            * velocity_ra_kms
            / (
                KM_S_PER_ARCSEC_YR_PC
                * distance_pc
            )
        )

        pmdec_masyr = (
            1000.0
            * velocity_dec_kms
            / (
                KM_S_PER_ARCSEC_YR_PC
                * distance_pc
            )
        )

        return pd.DataFrame(
            {
                "ra": np.degrees(ra_rad),
                "dec": np.degrees(dec_rad),
                "parallax": 1000.0 / distance_pc,
                "pmra": pmra_masyr,
                "pmdec": pmdec_masyr,
                "radial_velocity": radial_velocity_kms,
                "distance_pc_true": distance_pc,
                "radial_velocity_true": radial_velocity_kms,
            }
        )

    def add_observational_noise(
        self,
        catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Añade errores observacionales al catálogo.
        """

        result = catalog.copy()
        n_sources = len(result)

        # ----------------------------------------------------
        # Posición angular
        # ----------------------------------------------------

        position_sigma_deg = (
            self.config.position_error_mas
            / 3_600_000.0
        )

        result["ra_error"] = self.config.position_error_mas
        result["dec_error"] = self.config.position_error_mas

        if position_sigma_deg > 0.0:
            dec_rad = np.radians(result["dec"].to_numpy())

            cos_dec = np.cos(dec_rad)
            cos_dec = np.clip(
                np.abs(cos_dec),
                1.0e-8,
                None,
            )

            delta_ra = self.rng.normal(
                0.0,
                position_sigma_deg / cos_dec,
                n_sources,
            )

            delta_dec = self.rng.normal(
                0.0,
                position_sigma_deg,
                n_sources,
            )

            result["ra"] = (
                result["ra"].to_numpy() + delta_ra
            ) % 360.0

            result["dec"] = np.clip(
                result["dec"].to_numpy() + delta_dec,
                -90.0,
                90.0,
            )

        # ----------------------------------------------------
        # Paralaje
        # ----------------------------------------------------

        parallax_error = np.full(
            n_sources,
            self.config.parallax_error_mas,
            dtype=float,
        )

        result["parallax_error"] = parallax_error

        if self.config.parallax_error_mas > 0.0:
            result["parallax"] = (
                result["parallax"].to_numpy()
                + self.rng.normal(
                    0.0,
                    parallax_error,
                    n_sources,
                )
            )

        # ----------------------------------------------------
        # Movimientos propios
        # ----------------------------------------------------

        (
            pmra_error,
            pmdec_error,
            pm_corr,
        ) = self.build_proper_motion_error_model(n_sources)

        result["pmra_error"] = pmra_error
        result["pmdec_error"] = pmdec_error
        result["pmra_pmdec_corr"] = pm_corr

        z1 = self.rng.normal(size=n_sources)
        z2 = self.rng.normal(size=n_sources)

        delta_pmra = pmra_error * z1

        delta_pmdec = pmdec_error * (
            pm_corr * z1
            + np.sqrt(
                np.clip(
                    1.0 - pm_corr**2,
                    0.0,
                    None,
                )
            )
            * z2
        )

        result["pmra"] = (
            result["pmra"].to_numpy()
            + delta_pmra
        )

        result["pmdec"] = (
            result["pmdec"].to_numpy()
            + delta_pmdec
        )

        # ----------------------------------------------------
        # Velocidad radial
        # ----------------------------------------------------

        radial_velocity_error = np.full(
            n_sources,
            self.config.radial_velocity_error_kms,
            dtype=float,
        )

        result["radial_velocity_error"] = radial_velocity_error

        if self.config.radial_velocity_error_kms > 0.0:
            result["radial_velocity"] = (
                result["radial_velocity"].to_numpy()
                + self.rng.normal(
                    0.0,
                    radial_velocity_error,
                    n_sources,
                )
            )

        return result

    def add_zero_error_columns(
        self,
        catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Añade columnas de incertidumbre iguales a cero.
        """

        result = catalog.copy()
        n_sources = len(result)

        result["ra_error"] = np.zeros(n_sources)
        result["dec_error"] = np.zeros(n_sources)
        result["parallax_error"] = np.zeros(n_sources)
        result["pmra_error"] = np.zeros(n_sources)
        result["pmdec_error"] = np.zeros(n_sources)
        result["pmra_pmdec_corr"] = np.zeros(n_sources)
        result["radial_velocity_error"] = np.zeros(n_sources)

        return result

    def build_proper_motion_error_model(
        self,
        n_sources: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye errores estrella a estrella para pmra y pmdec.
        """

        config = self.config

        if config.pmra_error_distribution == "constant":
            pmra_sigma = (
                config.pmra_error_masyr
                if config.pmra_error_masyr is not None
                else config.proper_motion_error_masyr
            )

            pmra_error = np.full(
                n_sources,
                pmra_sigma,
                dtype=float,
            )

        else:
            log10_pmra_error = self.rng.triangular(
                left=config.pmra_error_log10_left,
                mode=config.pmra_error_log10_mode,
                right=config.pmra_error_log10_right,
                size=n_sources,
            )

            pmra_error = 10.0 ** log10_pmra_error

        if config.proper_motion_error_logscatter > 0.0:
            common_factor = self.rng.lognormal(
                mean=0.0,
                sigma=config.proper_motion_error_logscatter,
                size=n_sources,
            )

            pmra_error = pmra_error * common_factor

        if config.pmdec_error_masyr is not None:
            pmdec_error = np.full(
                n_sources,
                config.pmdec_error_masyr,
                dtype=float,
            )

        elif config.pm_error_slope is not None:
            slope = np.full(
                n_sources,
                config.pm_error_slope,
                dtype=float,
            )

            if config.pm_error_slope_logscatter > 0.0:
                slope *= self.rng.lognormal(
                    mean=0.0,
                    sigma=config.pm_error_slope_logscatter,
                    size=n_sources,
                )

            pmdec_error = slope * pmra_error

        else:
            pmdec_sigma = (
                config.pmra_error_masyr
                if config.pmra_error_masyr is not None
                else config.proper_motion_error_masyr
            )

            pmdec_error = np.full(
                n_sources,
                pmdec_sigma,
                dtype=float,
            )

        correlation = np.full(
            n_sources,
            config.pmra_pmdec_corr,
            dtype=float,
        )

        return pmra_error, pmdec_error, correlation

    # ========================================================
    # COORDENADAS
    # ========================================================

    def get_center_coord_icrs(self) -> SkyCoord:
        """
        Devuelve el centro del cúmulo en ICRS.
        """

        config = self.config

        if config.center_ra_deg is not None:
            return SkyCoord(
                ra=config.center_ra_deg * u.deg,
                dec=config.center_dec_deg * u.deg,
                distance=config.distance_pc * u.pc,
                frame="icrs",
            )

        return SkyCoord(
            l=config.center_l_deg * u.deg,
            b=config.center_b_deg * u.deg,
            distance=config.distance_pc * u.pc,
            frame="galactic",
        ).icrs

    def get_apex_coord_icrs(self) -> SkyCoord:
        """
        Devuelve el ápex nominal en ICRS.
        """

        config = self.config

        if config.apex_ra_deg is not None:
            return SkyCoord(
                ra=config.apex_ra_deg * u.deg,
                dec=config.apex_dec_deg * u.deg,
                frame="icrs",
            )

        return SkyCoord(
            l=config.apex_l_deg * u.deg,
            b=config.apex_b_deg * u.deg,
            frame="galactic",
        ).icrs

    def sample_apex_icrs(self) -> tuple[float, float]:
        """
        Muestrea una dirección de ápex para una realización.
        """

        config = self.config

        if config.apex_ra_deg is not None:
            lon = config.apex_ra_deg
            lat = config.apex_dec_deg
            input_frame = "icrs"
        else:
            lon = config.apex_l_deg
            lat = config.apex_b_deg
            input_frame = "galactic"

        if config.apex_lon_sigma_deg > 0.0:
            lon = self.rng.normal(
                lon,
                config.apex_lon_sigma_deg,
            )

        if config.apex_lat_sigma_deg > 0.0:
            lat = self.rng.normal(
                lat,
                config.apex_lat_sigma_deg,
            )

        lon = lon % 360.0
        lat = np.clip(lat, -90.0, 90.0)

        if input_frame == "icrs":
            return float(lon), float(lat)

        return galactic_to_icrs(
            float(lon),
            float(lat),
        )

    # ========================================================
    # VALIDACIÓN
    # ========================================================

    def _validate_config(self) -> None:
        """
        Valida los parámetros de entrada.
        """

        config = self.config

        if config.n_members <= 0:
            raise ValueError("n_members debe ser mayor que cero.")

        if config.distance_pc <= 0.0:
            raise ValueError("distance_pc debe ser positivo.")

        if config.radius_pc < 0.0:
            raise ValueError("radius_pc no puede ser negativo.")

        if config.speed_kms <= 0.0:
            raise ValueError("speed_kms debe ser positivo.")

        self._validate_coordinate_choice(
            object_name="centro",
            equatorial_lon=config.center_ra_deg,
            equatorial_lat=config.center_dec_deg,
            galactic_lon=config.center_l_deg,
            galactic_lat=config.center_b_deg,
        )

        self._validate_coordinate_choice(
            object_name="ápex",
            equatorial_lon=config.apex_ra_deg,
            equatorial_lat=config.apex_dec_deg,
            galactic_lon=config.apex_l_deg,
            galactic_lat=config.apex_b_deg,
        )

        latitude_values = {
            "center_dec_deg": config.center_dec_deg,
            "center_b_deg": config.center_b_deg,
            "apex_dec_deg": config.apex_dec_deg,
            "apex_b_deg": config.apex_b_deg,
        }

        for name, value in latitude_values.items():
            if value is not None and not -90.0 <= value <= 90.0:
                raise ValueError(
                    f"{name} debe estar entre -90 y 90 grados."
                )

        nonnegative_values = {
            "velocity_dispersion_1d_kms":
                config.velocity_dispersion_1d_kms,
            "bulk_speed_sigma_kms":
                config.bulk_speed_sigma_kms,
            "apex_lon_sigma_deg":
                config.apex_lon_sigma_deg,
            "apex_lat_sigma_deg":
                config.apex_lat_sigma_deg,
            "parallax_error_mas":
                config.parallax_error_mas,
            "radial_velocity_error_kms":
                config.radial_velocity_error_kms,
            "position_error_mas":
                config.position_error_mas,
            "proper_motion_error_masyr":
                config.proper_motion_error_masyr,
            "proper_motion_error_logscatter":
                config.proper_motion_error_logscatter,
            "pm_error_slope_logscatter":
                config.pm_error_slope_logscatter,
        }

        for name, value in nonnegative_values.items():
            if value < 0.0:
                raise ValueError(
                    f"{name} no puede ser negativo."
                )

        optional_nonnegative = {
            "pmra_error_masyr": config.pmra_error_masyr,
            "pmdec_error_masyr": config.pmdec_error_masyr,
        }

        for name, value in optional_nonnegative.items():
            if value is not None and value < 0.0:
                raise ValueError(
                    f"{name} no puede ser negativo."
                )

        if config.pmra_error_distribution not in {
            "constant",
            "log10_triangular",
        }:
            raise ValueError(
                "pmra_error_distribution debe ser "
                "'constant' o 'log10_triangular'."
            )

        if config.pmra_error_distribution == "log10_triangular":
            left = config.pmra_error_log10_left
            mode = config.pmra_error_log10_mode
            right = config.pmra_error_log10_right

            if not left <= mode <= right:
                raise ValueError(
                    "Los parámetros triangulares deben cumplir "
                    "left <= mode <= right."
                )

        if (
            config.pm_error_slope is not None
            and config.pm_error_slope <= 0.0
        ):
            raise ValueError(
                "pm_error_slope debe ser positivo."
            )

        if not -1.0 <= config.pmra_pmdec_corr <= 1.0:
            raise ValueError(
                "pmra_pmdec_corr debe estar entre -1 y 1."
            )

    @staticmethod
    def _validate_coordinate_choice(
        object_name: str,
        equatorial_lon: Optional[float],
        equatorial_lat: Optional[float],
        galactic_lon: Optional[float],
        galactic_lat: Optional[float],
    ) -> None:

        equatorial_any = (
            equatorial_lon is not None
            or equatorial_lat is not None
        )

        equatorial_complete = (
            equatorial_lon is not None
            and equatorial_lat is not None
        )

        galactic_any = (
            galactic_lon is not None
            or galactic_lat is not None
        )

        galactic_complete = (
            galactic_lon is not None
            and galactic_lat is not None
        )

        if equatorial_any and not equatorial_complete:
            raise ValueError(
                f"Coordenadas ecuatoriales incompletas para {object_name}."
            )

        if galactic_any and not galactic_complete:
            raise ValueError(
                f"Coordenadas galácticas incompletas para {object_name}."
            )

        if equatorial_complete and galactic_complete:
            raise ValueError(
                f"Coordenadas ambiguas para {object_name}: "
                "usa ICRS o galácticas, no ambas."
            )

        if not equatorial_complete and not galactic_complete:
            raise ValueError(
                f"Faltan las coordenadas de {object_name}."
            )


# ============================================================
# RECONSTRUCCIÓN DE VELOCIDADES
# ============================================================

def _build_tangent_basis(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye las bases tangenciales e_RA y e_Dec.
    """

    sin_ra = np.sin(ra_rad)
    cos_ra = np.cos(ra_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)

    basis_ra = np.column_stack(
        (
            -sin_ra,
            cos_ra,
            np.zeros_like(ra_rad),
        )
    )

    basis_dec = np.column_stack(
        (
            -cos_ra * sin_dec,
            -sin_ra * sin_dec,
            cos_dec,
        )
    )

    return basis_ra, basis_dec


def reconstruct_icrs_velocities(
    catalog: pd.DataFrame,
    ra_column: str = "ra",
    dec_column: str = "dec",
    parallax_column: str = "parallax",
    pmra_column: str = "pmra",
    pmdec_column: str = "pmdec",
    radial_velocity_column: str = "radial_velocity",
    invalid_value: float = np.nan,
) -> pd.DataFrame:
    """
    Reconstruye las velocidades cartesianas ICRS a partir de
    observables astrométricos y velocidad radial.

    La reconstrucción usa:

        distance_pc = 1000 / parallax_mas

        v_RA = 4.74047 * distance_pc * pmra / 1000

        v_Dec = 4.74047 * distance_pc * pmdec / 1000

        v = RV e_r + v_RA e_RA + v_Dec e_Dec
    """

    required_columns = [
        ra_column,
        dec_column,
        parallax_column,
        pmra_column,
        pmdec_column,
        radial_velocity_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in catalog.columns
    ]

    if missing:
        raise KeyError(
            f"Faltan columnas necesarias: {missing}"
        )

    result = catalog.copy()

    ra_deg = result[ra_column].to_numpy(dtype=float)
    dec_deg = result[dec_column].to_numpy(dtype=float)
    parallax_mas = result[parallax_column].to_numpy(dtype=float)
    pmra_masyr = result[pmra_column].to_numpy(dtype=float)
    pmdec_masyr = result[pmdec_column].to_numpy(dtype=float)
    radial_velocity = result[
        radial_velocity_column
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & np.isfinite(parallax_mas)
        & np.isfinite(pmra_masyr)
        & np.isfinite(pmdec_masyr)
        & np.isfinite(radial_velocity)
        & (parallax_mas > 0.0)
    )

    distance_pc = np.full(
        len(result),
        invalid_value,
        dtype=float,
    )

    distance_pc[valid] = (
        1000.0 / parallax_mas[valid]
    )

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    cos_dec = np.cos(dec_rad)

    unit_position = np.column_stack(
        (
            cos_dec * np.cos(ra_rad),
            cos_dec * np.sin(ra_rad),
            np.sin(dec_rad),
        )
    )

    basis_ra, basis_dec = _build_tangent_basis(
        ra_rad,
        dec_rad,
    )

    velocity_ra = np.full(
        len(result),
        invalid_value,
        dtype=float,
    )

    velocity_dec = np.full(
        len(result),
        invalid_value,
        dtype=float,
    )

    velocity_ra[valid] = (
        KM_S_PER_ARCSEC_YR_PC
        * distance_pc[valid]
        * pmra_masyr[valid]
        / 1000.0
    )

    velocity_dec[valid] = (
        KM_S_PER_ARCSEC_YR_PC
        * distance_pc[valid]
        * pmdec_masyr[valid]
        / 1000.0
    )

    velocities = (
        radial_velocity[:, None] * unit_position
        + velocity_ra[:, None] * basis_ra
        + velocity_dec[:, None] * basis_dec
    )

    velocities[~valid] = invalid_value

    result["distance_pc_reconstructed"] = distance_pc
    result["velocity_ra_kms"] = velocity_ra
    result["velocity_dec_kms"] = velocity_dec

    result["vx_kms"] = velocities[:, 0]
    result["vy_kms"] = velocities[:, 1]
    result["vz_kms"] = velocities[:, 2]

    result["speed_kms"] = np.linalg.norm(
        velocities,
        axis=1,
    )

    result["valid_velocity_reconstruction"] = valid

    return result


# ============================================================
# ESTIMACIÓN DEL ÁPEX
# ============================================================

def estimate_apex_from_velocities(
    velocities_kms: np.ndarray,
    method: Literal["mean", "median"] = "mean",
) -> dict[str, float]:
    """
    Estima el ápex a partir de una matriz de velocidades.

    Parameters
    ----------
    velocities_kms
        Arreglo de forma (N, 3).
    method
        "mean" o "median".

    Returns
    -------
    dict
        Coordenadas del ápex en ICRS y galácticas, además de
        la velocidad representativa.
    """

    velocities = np.asarray(
        velocities_kms,
        dtype=float,
    )

    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError(
            "velocities_kms debe tener forma (N, 3)."
        )

    valid = np.all(
        np.isfinite(velocities),
        axis=1,
    )

    velocities = velocities[valid]

    if len(velocities) == 0:
        raise ValueError(
            "No existen velocidades válidas para estimar el ápex."
        )

    if method == "mean":
        representative_velocity = np.mean(
            velocities,
            axis=0,
        )
    elif method == "median":
        representative_velocity = np.median(
            velocities,
            axis=0,
        )
    else:
        raise ValueError(
            "method debe ser 'mean' o 'median'."
        )

    speed_kms = float(
        np.linalg.norm(representative_velocity)
    )

    if speed_kms <= 0.0:
        raise ValueError(
            "La velocidad representativa tiene norma cero."
        )

    apex_ra_deg, apex_dec_deg = unit_vector_to_spherical(
        representative_velocity
    )

    apex_ra_deg = float(apex_ra_deg)
    apex_dec_deg = float(apex_dec_deg)

    apex_l_deg, apex_b_deg = icrs_to_galactic(
        apex_ra_deg,
        apex_dec_deg,
    )

    stellar_speeds = np.linalg.norm(
        velocities,
        axis=1,
    )

    residuals = velocities - representative_velocity[None, :]

    return {
        "apex_ra_deg": apex_ra_deg,
        "apex_dec_deg": apex_dec_deg,
        "apex_l_deg": apex_l_deg,
        "apex_b_deg": apex_b_deg,
        "vx_mean_kms": float(representative_velocity[0]),
        "vy_mean_kms": float(representative_velocity[1]),
        "vz_mean_kms": float(representative_velocity[2]),
        "speed_kms": speed_kms,
        "stellar_speed_mean_kms": float(
            np.mean(stellar_speeds)
        ),
        "stellar_speed_std_kms": float(
            np.std(stellar_speeds, ddof=1)
            if len(stellar_speeds) > 1
            else 0.0
        ),
        "velocity_residual_rms_kms": float(
            np.sqrt(np.mean(residuals**2))
        ),
        "n_sources_used": int(len(velocities)),
    }


def analyze_catalog_apex(
    catalog: pd.DataFrame,
    true_apex_ra_deg: Optional[float] = None,
    true_apex_dec_deg: Optional[float] = None,
    method: Literal["mean", "median"] = "mean",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Reconstruye las velocidades de un catálogo y calcula su ápex.
    """

    reconstructed = reconstruct_icrs_velocities(catalog)

    velocity_matrix = reconstructed[
        ["vx_kms", "vy_kms", "vz_kms"]
    ].to_numpy()

    apex_result = estimate_apex_from_velocities(
        velocity_matrix,
        method=method,
    )

    if (
        true_apex_ra_deg is not None
        and true_apex_dec_deg is not None
    ):
        apex_result["apex_error_deg"] = angular_separation_deg(
            true_apex_ra_deg,
            true_apex_dec_deg,
            apex_result["apex_ra_deg"],
            apex_result["apex_dec_deg"],
        )

    return reconstructed, apex_result


# ============================================================
# MONTE CARLO
# ============================================================

def run_apex_monte_carlo(
    simulation_config: ClusterSimulationConfig,
    monte_carlo_config: MonteCarloConfig,
) -> pd.DataFrame:
    """
    Ejecuta realizaciones Monte Carlo del ápex.

    Modos
    -----
    Intrínseco + observacional:
        resample_intrinsic_kinematics=True
        resample_observational_noise=True

    Solo observacional:
        resample_intrinsic_kinematics=False
        resample_observational_noise=True

    Solo intrínseco:
        resample_intrinsic_kinematics=True
        resample_observational_noise=False
    """

    if monte_carlo_config.n_iterations <= 0:
        raise ValueError(
            "n_iterations debe ser mayor que cero."
        )

    master_rng = np.random.default_rng(
        monte_carlo_config.seed
    )

    true_apex = _get_nominal_apex_icrs(
        simulation_config
    )

    true_apex_ra_deg = float(true_apex.ra.deg)
    true_apex_dec_deg = float(true_apex.dec.deg)

    results: list[dict[str, float | int]] = []

    base_clean_catalog: Optional[pd.DataFrame] = None

    # Si no se remuestrea la estructura física, se genera una sola vez.
    if not monte_carlo_config.resample_intrinsic_kinematics:
        intrinsic_seed = int(
            master_rng.integers(0, 2**32 - 1)
        )

        intrinsic_config = replace(
            simulation_config,
            seed=intrinsic_seed,
            include_true_values=True,
        )

        base_pipeline = OpenClusterPipeline(
            intrinsic_config
        )

        base_clean_catalog = base_pipeline.simulate(
            add_noise=False
        )

    for iteration in range(
        monte_carlo_config.n_iterations
    ):
        iteration_seed = int(
            master_rng.integers(0, 2**32 - 1)
        )

        iteration_config = replace(
            simulation_config,
            seed=iteration_seed,
            include_true_values=True,
        )

        pipeline = OpenClusterPipeline(
            iteration_config
        )

        if monte_carlo_config.resample_intrinsic_kinematics:
            catalog = pipeline.simulate(
                add_noise=(
                    monte_carlo_config
                    .resample_observational_noise
                )
            )

        else:
            if base_clean_catalog is None:
                raise RuntimeError(
                    "No se generó el catálogo físico base."
                )

            catalog = base_clean_catalog.copy()

            if (
                monte_carlo_config
                .resample_observational_noise
            ):
                catalog = pipeline.add_observational_noise(
                    catalog
                )

        reconstructed = reconstruct_icrs_velocities(
            catalog
        )

        if monte_carlo_config.remove_invalid_sources:
            reconstructed = reconstructed.loc[
                reconstructed[
                    "valid_velocity_reconstruction"
                ]
            ].copy()

        velocities = reconstructed[
            ["vx_kms", "vy_kms", "vz_kms"]
        ].to_numpy()

        apex_result = estimate_apex_from_velocities(
            velocities,
            method=monte_carlo_config.apex_method,
        )

        apex_error = angular_separation_deg(
            true_apex_ra_deg,
            true_apex_dec_deg,
            apex_result["apex_ra_deg"],
            apex_result["apex_dec_deg"],
        )

        true_velocity_apex = _estimate_true_velocity_apex(
            catalog,
            method=monte_carlo_config.apex_method,
        )

        observational_apex_error = angular_separation_deg(
            true_velocity_apex["apex_ra_deg"],
            true_velocity_apex["apex_dec_deg"],
            apex_result["apex_ra_deg"],
            apex_result["apex_dec_deg"],
        )

        realization_intrinsic_error = angular_separation_deg(
            true_apex_ra_deg,
            true_apex_dec_deg,
            true_velocity_apex["apex_ra_deg"],
            true_velocity_apex["apex_dec_deg"],
        )

        results.append(
            {
                "iteration": iteration,
                "seed": iteration_seed,
                "n_sources_used":
                    apex_result["n_sources_used"],

                "input_apex_ra_deg":
                    true_apex_ra_deg,
                "input_apex_dec_deg":
                    true_apex_dec_deg,

                "true_velocity_apex_ra_deg":
                    true_velocity_apex["apex_ra_deg"],
                "true_velocity_apex_dec_deg":
                    true_velocity_apex["apex_dec_deg"],

                "recovered_apex_ra_deg":
                    apex_result["apex_ra_deg"],
                "recovered_apex_dec_deg":
                    apex_result["apex_dec_deg"],

                "recovered_apex_l_deg":
                    apex_result["apex_l_deg"],
                "recovered_apex_b_deg":
                    apex_result["apex_b_deg"],

                "recovered_speed_kms":
                    apex_result["speed_kms"],

                # Error total respecto al ápex de entrada.
                "apex_error_deg":
                    apex_error,

                # Error causado por observaciones y reconstrucción.
                "observational_apex_error_deg":
                    observational_apex_error,

                # Diferencia entre el ápex de entrada y el ápex
                # de la realización física concreta.
                "intrinsic_realization_error_deg":
                    realization_intrinsic_error,
            }
        )

    return pd.DataFrame(results)


def _estimate_true_velocity_apex(
    catalog: pd.DataFrame,
    method: Literal["mean", "median"],
) -> dict[str, float]:

    required = [
        "vx_kms_true",
        "vy_kms_true",
        "vz_kms_true",
    ]

    missing = [
        column
        for column in required
        if column not in catalog.columns
    ]

    if missing:
        raise KeyError(
            "El catálogo no contiene velocidades verdaderas. "
            "Usa include_true_values=True."
        )

    true_velocities = catalog[required].to_numpy()

    return estimate_apex_from_velocities(
        true_velocities,
        method=method,
    )


def summarize_monte_carlo(
    monte_carlo_results: pd.DataFrame,
) -> pd.Series:
    """
    Resume la distribución Monte Carlo del error del ápex.
    """

    required = [
        "apex_error_deg",
        "observational_apex_error_deg",
        "intrinsic_realization_error_deg",
        "recovered_speed_kms",
    ]

    missing = [
        column
        for column in required
        if column not in monte_carlo_results.columns
    ]

    if missing:
        raise KeyError(
            f"Faltan columnas Monte Carlo: {missing}"
        )

    total_error = monte_carlo_results[
        "apex_error_deg"
    ].dropna()

    observational_error = monte_carlo_results[
        "observational_apex_error_deg"
    ].dropna()

    intrinsic_error = monte_carlo_results[
        "intrinsic_realization_error_deg"
    ].dropna()

    speed = monte_carlo_results[
        "recovered_speed_kms"
    ].dropna()

    return pd.Series(
        {
            "n_iterations": len(monte_carlo_results),

            "apex_error_mean_deg":
                total_error.mean(),
            "apex_error_std_deg":
                total_error.std(ddof=1),
            "apex_error_median_deg":
                total_error.median(),
            "apex_error_p16_deg":
                total_error.quantile(0.16),
            "apex_error_p84_deg":
                total_error.quantile(0.84),
            "apex_error_p025_deg":
                total_error.quantile(0.025),
            "apex_error_p975_deg":
                total_error.quantile(0.975),

            "observational_error_median_deg":
                observational_error.median(),
            "observational_error_p84_deg":
                observational_error.quantile(0.84),

            "intrinsic_error_median_deg":
                intrinsic_error.median(),
            "intrinsic_error_p84_deg":
                intrinsic_error.quantile(0.84),

            "speed_mean_kms":
                speed.mean(),
            "speed_std_kms":
                speed.std(ddof=1),
            "speed_p16_kms":
                speed.quantile(0.16),
            "speed_p84_kms":
                speed.quantile(0.84),
        },
        dtype=float,
    )


# ============================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================

def simulate_cluster(
    config: ClusterSimulationConfig,
    add_noise: bool = True,
) -> pd.DataFrame:
    """
    Atajo para ejecutar una simulación.
    """

    return OpenClusterPipeline(config).simulate(
        add_noise=add_noise
    )


def save_catalog(
    catalog: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Guarda un catálogo como CSV.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_monte_carlo_results(
    results: pd.DataFrame,
    output_path: str | Path,
    summary_path: Optional[str | Path] = None,
) -> tuple[Path, Optional[Path]]:
    """
    Guarda las realizaciones Monte Carlo y, opcionalmente, su resumen.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    saved_summary_path: Optional[Path] = None

    if summary_path is not None:
        saved_summary_path = Path(summary_path)
        saved_summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = summarize_monte_carlo(results)

        summary.rename("value").to_csv(
            saved_summary_path,
            header=True,
        )

    return output_path, saved_summary_path


def _get_nominal_apex_icrs(
    config: ClusterSimulationConfig,
) -> SkyCoord:
    """
    Obtiene el ápex nominal de una configuración.
    """

    if config.apex_ra_deg is not None:
        return SkyCoord(
            ra=config.apex_ra_deg * u.deg,
            dec=config.apex_dec_deg * u.deg,
            frame="icrs",
        )

    return SkyCoord(
        l=config.apex_l_deg * u.deg,
        b=config.apex_b_deg * u.deg,
        frame="galactic",
    ).icrs

def calculate_pole_apex_metrics(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    velocities_kms: np.ndarray,
    apex_ra_deg: float | None = None,
    apex_dec_deg: float | None = None,
    initial_apex_ra_deg: float | None = None,
    initial_apex_dec_deg: float | None = None,
    weights: np.ndarray | None = None,
    min_cross_norm: float = 1.0e-12,
) -> dict[str, np.ndarray | float]:
    """
    Calcula los residuos geométricos de los polos de movimiento
    respecto a una dirección de ápex.

    Parameters
    ----------
    ra_deg, dec_deg
        Posición ICRS de cada estrella, en grados.

    velocities_kms
        Velocidades cartesianas ICRS de forma (N, 3), en km/s.

    apex_ra_deg, apex_dec_deg
        Ápex final respecto al cual se calcula rms_pole_residual_deg.

        Si no se proporcionan, el ápex se estima a partir de la
        velocidad media del conjunto.

    initial_apex_ra_deg, initial_apex_dec_deg
        Ápex inicial usado para asignar signo a los residuos.

        Si no se proporcionan, se usa el mismo ápex final.

    weights
        Pesos opcionales por estrella. Si es None, todas las estrellas
        reciben peso unitario.

    min_cross_norm
        Norma mínima permitida para r_hat x v_hat.

    Returns
    -------
    dict
        Contiene:

        - pole_vectors
        - pole_residual_deg
        - pole_apex_error_direction_signed_initial_deg
        - rms_pole_residual_deg
        - valid_pole
        - apex_unit_vector
        - initial_apex_unit_vector
    """

    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    velocities_kms = np.asarray(velocities_kms, dtype=float)

    n_sources = len(ra_deg)

    if dec_deg.shape != ra_deg.shape:
        raise ValueError(
            "ra_deg y dec_deg deben tener la misma forma."
        )

    if velocities_kms.shape != (n_sources, 3):
        raise ValueError(
            "velocities_kms debe tener forma (N, 3)."
        )

    if weights is None:
        weights = np.ones(n_sources, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

        if weights.shape != (n_sources,):
            raise ValueError(
                "weights debe tener forma (N,)."
            )

    # ========================================================
    # 1. Vectores unitarios de posición
    # ========================================================

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    position_unit_vectors = np.column_stack(
        (
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        )
    )

    # ========================================================
    # 2. Vectores unitarios de velocidad
    # ========================================================

    velocity_norms = np.linalg.norm(
        velocities_kms,
        axis=1,
    )

    valid_velocity = (
        np.all(np.isfinite(velocities_kms), axis=1)
        & np.isfinite(velocity_norms)
        & (velocity_norms > 0.0)
    )

    velocity_unit_vectors = np.full(
        velocities_kms.shape,
        np.nan,
        dtype=float,
    )

    velocity_unit_vectors[valid_velocity] = (
        velocities_kms[valid_velocity]
        / velocity_norms[valid_velocity, None]
    )

    # ========================================================
    # 3. Polos de los círculos máximos
    #
    # p_i = r_i x v_i
    # ========================================================

    raw_pole_vectors = np.cross(
        position_unit_vectors,
        velocity_unit_vectors,
    )

    pole_norms = np.linalg.norm(
        raw_pole_vectors,
        axis=1,
    )

    valid_pole = (
        valid_velocity
        & np.all(np.isfinite(position_unit_vectors), axis=1)
        & np.isfinite(pole_norms)
        & (pole_norms > min_cross_norm)
        & np.isfinite(weights)
        & (weights > 0.0)
    )

    pole_vectors = np.full(
        raw_pole_vectors.shape,
        np.nan,
        dtype=float,
    )

    pole_vectors[valid_pole] = (
        raw_pole_vectors[valid_pole]
        / pole_norms[valid_pole, None]
    )

    if valid_pole.sum() < 2:
        raise ValueError(
            "No existen suficientes polos válidos."
        )

    # ========================================================
    # 4. Ápex final
    # ========================================================

    if apex_ra_deg is None or apex_dec_deg is None:
        mean_velocity = np.average(
            velocities_kms[valid_velocity],
            axis=0,
            weights=weights[valid_velocity],
        )

        mean_velocity_norm = np.linalg.norm(mean_velocity)

        if (
            not np.isfinite(mean_velocity_norm)
            or mean_velocity_norm <= 0.0
        ):
            raise ValueError(
                "No fue posible estimar el ápex desde las velocidades."
            )

        apex_unit_vector = (
            mean_velocity / mean_velocity_norm
        )

    else:
        apex_unit_vector = spherical_to_unit_vector(
            apex_ra_deg,
            apex_dec_deg,
        )

    apex_unit_vector = np.asarray(
        apex_unit_vector,
        dtype=float,
    ).reshape(3)

    # ========================================================
    # 5. Ápex inicial para el signo
    # ========================================================

    if (
        initial_apex_ra_deg is None
        or initial_apex_dec_deg is None
    ):
        initial_apex_unit_vector = apex_unit_vector.copy()

    else:
        initial_apex_unit_vector = spherical_to_unit_vector(
            initial_apex_ra_deg,
            initial_apex_dec_deg,
        )

        initial_apex_unit_vector = np.asarray(
            initial_apex_unit_vector,
            dtype=float,
        ).reshape(3)

    # ========================================================
    # 6. Residual de polo respecto al ápex final
    #
    # r_i = asin(|p_i dot a|)
    # ========================================================

    pole_dot_apex = np.full(
        n_sources,
        np.nan,
        dtype=float,
    )

    pole_dot_apex[valid_pole] = (
        pole_vectors[valid_pole]
        @ apex_unit_vector
    )

    pole_dot_apex[valid_pole] = np.clip(
        pole_dot_apex[valid_pole],
        -1.0,
        1.0,
    )

    pole_residual_deg = np.full(
        n_sources,
        np.nan,
        dtype=float,
    )

    pole_residual_deg[valid_pole] = np.degrees(
        np.arcsin(
            np.abs(
                pole_dot_apex[valid_pole]
            )
        )
    )

    # ========================================================
    # 7. Residual firmado respecto al ápex inicial
    #
    # signed_i = asin(p_i dot a_initial)
    #
    # El signo indica en qué lado del plano del polo está
    # el ápex inicial.
    # ========================================================

    pole_dot_initial_apex = np.full(
        n_sources,
        np.nan,
        dtype=float,
    )

    pole_dot_initial_apex[valid_pole] = (
        pole_vectors[valid_pole]
        @ initial_apex_unit_vector
    )

    pole_dot_initial_apex[valid_pole] = np.clip(
        pole_dot_initial_apex[valid_pole],
        -1.0,
        1.0,
    )

    pole_apex_error_direction_signed_initial_deg = np.full(
        n_sources,
        np.nan,
        dtype=float,
    )

    pole_apex_error_direction_signed_initial_deg[
        valid_pole
    ] = np.degrees(
        np.arcsin(
            pole_dot_initial_apex[valid_pole]
        )
    )

    # ========================================================
    # 8. RMS ponderado de los residuos de polo
    #
    # RMS = sqrt(sum(w_i r_i²) / sum(w_i))
    # ========================================================

    valid_weights = weights[valid_pole]
    valid_residuals = pole_residual_deg[valid_pole]

    rms_pole_residual_deg = float(
        np.sqrt(
            np.average(
                valid_residuals**2,
                weights=valid_weights,
            )
        )
    )

    return {
        "pole_vectors": pole_vectors,
        "pole_residual_deg": pole_residual_deg,
        "pole_apex_error_direction_signed_initial_deg": (
            pole_apex_error_direction_signed_initial_deg
        ),
        "rms_pole_residual_deg": rms_pole_residual_deg,
        "valid_pole": valid_pole,
        "apex_unit_vector": apex_unit_vector,
        "initial_apex_unit_vector": (
            initial_apex_unit_vector
        ),
    }

# ============================================================
# EJEMPLO EJECUTABLE
# ============================================================

def main() -> None:
    """
    Ejemplo completo.

    Se ejecuta únicamente cuando se llama:

        python open_cluster_pipeline.py

    No se ejecuta al importar el módulo.
    """

    config = ClusterSimulationConfig(
        n_members=455,

        # Centro en coordenadas galácticas.
        center_l_deg=221.0,
        center_b_deg=84.0,

        distance_pc=85.0,
        radius_pc=5.0,

        # Ápex en coordenadas ecuatoriales.
        apex_ra_deg=180.0,
        apex_dec_deg=30.0,

        speed_kms=25.0,

        # Dispersión interna por componente cartesiana.
        velocity_dispersion_1d_kms=0.5,

        # Se recomienda dejarlas en cero inicialmente.
        bulk_speed_sigma_kms=0.0,
        apex_lon_sigma_deg=0.0,
        apex_lat_sigma_deg=0.0,

        # Errores observacionales.
        parallax_error_mas=0.02,
        radial_velocity_error_kms=1.0,
        position_error_mas=0.1,

        pmra_error_distribution="log10_triangular",
        pmra_error_log10_left=-3.0,
        pmra_error_log10_mode=-1.7,
        pmra_error_log10_right=0.0,

        pm_error_slope=0.7,
        pmra_pmdec_corr=0.2,
        pm_error_slope_logscatter=0.08,

        seed=42,
        include_true_values=True,
    )

    # --------------------------------------------------------
    # 1. Simulación
    # --------------------------------------------------------

    catalog = simulate_cluster(
        config,
        add_noise=True,
    )

    print("\nCATÁLOGO SIMULADO")
    print(catalog.head())

    save_catalog(
        catalog,
        "results/mock_open_cluster_gaia.csv",
    )

    # --------------------------------------------------------
    # 2. Reconstrucción y ápex de una realización
    # --------------------------------------------------------

    reconstructed, apex = analyze_catalog_apex(
        catalog,
        true_apex_ra_deg=180.0,
        true_apex_dec_deg=30.0,
        method="mean",
    )

    print("\nÁPEX RECUPERADO")
    for key, value in apex.items():
        print(f"{key}: {value}")

    save_catalog(
        reconstructed,
        "results/reconstructed_cluster.csv",
    )

    # --------------------------------------------------------
    # 3. Monte Carlo
    # --------------------------------------------------------

    mc_config = MonteCarloConfig(
        n_iterations=100,
        seed=12345,
        resample_intrinsic_kinematics=True,
        resample_observational_noise=True,
        apex_method="mean",
    )

    mc_results = run_apex_monte_carlo(
        simulation_config=config,
        monte_carlo_config=mc_config,
    )

    mc_summary = summarize_monte_carlo(
        mc_results
    )

    print("\nRESUMEN MONTE CARLO")
    print(mc_summary)

    save_monte_carlo_results(
        results=mc_results,
        output_path="results/apex_monte_carlo.csv",
        summary_path="results/apex_monte_carlo_summary.csv",
    )


if __name__ == "__main__":
    main()