"""
open_cluster_simulator.py

Simulador cinemático tipo Gaia para cúmulos abiertos.

El modelo genera miembros de un cúmulo abierto en 3D, asigna una
velocidad espacial común dirigida hacia un ápex y proyecta esa cinemática
a observables astrométricos tipo Gaia:

    ra        [deg]
    dec       [deg]
    parallax  [mas]
    pmra      [mas / yr]  = mu_alpha * cos(dec)
    pmdec     [mas / yr]

El centro del cúmulo puede ingresarse en coordenadas ecuatoriales ICRS:

    center_ra_deg, center_dec_deg

o en coordenadas galácticas:

    center_l_deg, center_b_deg

El ápex también puede ingresarse en coordenadas ecuatoriales ICRS:

    apex_ra_deg, apex_dec_deg

o en coordenadas galácticas:

    apex_l_deg, apex_b_deg

Para cada dirección debe darse exactamente un sistema de coordenadas.
Si se pasan ambos sistemas para el centro o para el ápex, el código lanza
un error para evitar ambigüedades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord


KM_S_PER_ARCSEC_YR_PC = 4.74047


@dataclass(frozen=True)
class ClusterSimulationConfig:
    """
    Configuración física y observacional de la simulación.
    """

    n_members: int
    distance_pc: float
    radius_pc: float
    speed_kms: float

    # Centro del cúmulo en coordenadas ecuatoriales ICRS
    center_ra_deg: Optional[float] = None
    center_dec_deg: Optional[float] = None

    # Centro del cúmulo en coordenadas galácticas
    center_l_deg: Optional[float] = None
    center_b_deg: Optional[float] = None

    # Ápex en coordenadas ecuatoriales ICRS
    apex_ra_deg: Optional[float] = None
    apex_dec_deg: Optional[float] = None

    # Ápex en coordenadas galácticas
    apex_l_deg: Optional[float] = None
    apex_b_deg: Optional[float] = None

    # Dispersión en velocidad
    speed_sigma_kms: float = 0.0

    # Dispersión angular del ápex.
    # Si el ápex está en galácticas, corresponden a sigma_l y sigma_b.
    # Si el ápex está en ecuatoriales, corresponden a sigma_ra y sigma_dec.
    apex_l_sigma_deg: float = 0.0
    apex_b_sigma_deg: float = 0.0

    # Errores observacionales tipo Gaia
    parallax_error_mas: float = 0.0

    # Error común para pmra y pmdec, por compatibilidad
    proper_motion_error_masyr: float = 0.0

    # Errores separados opcionales para movimiento propio
    pmra_error_masyr: Optional[float] = None
    pmdec_error_masyr: Optional[float] = None

    position_error_mas: float = 0.0

    seed: Optional[int] = None
    include_true_values: bool = True


class OpenClusterSimulator:
    """
    Simulador cinemático de cúmulos abiertos tipo Gaia.
    """

    def __init__(self, config: ClusterSimulationConfig) -> None:
        """
        Inicializa el simulador.

        Parameters
        ----------
        config : ClusterSimulationConfig
            Configuración física y observacional del cúmulo.
        """

        self.config = config
        self._validate_config()
        self.rng = np.random.default_rng(config.seed)

    def simulate(self) -> pd.DataFrame:
        """
        Ejecuta la simulación completa.

        Returns
        -------
        pd.DataFrame
            Catálogo simulado con columnas principales:

            - source_id
            - ra
            - dec
            - parallax
            - pmra
            - pmdec

            También incluye columnas verdaderas y diagnósticas.
        """

        positions_pc = self._build_cluster_positions()
        velocities_kms = self._build_cluster_velocities()

        catalog = self._phase_space_to_gaia_observables(
            positions_pc=positions_pc,
            velocities_kms=velocities_kms,
        )

        catalog.insert(
            loc=0,
            column="source_id",
            value=np.arange(1, self.config.n_members + 1),
        )

        if self.config.include_true_values:
            catalog["ra_true"] = catalog["ra"]
            catalog["dec_true"] = catalog["dec"]
            catalog["parallax_true"] = catalog["parallax"]
            catalog["pmra_true"] = catalog["pmra"]
            catalog["pmdec_true"] = catalog["pmdec"]

        catalog = self._apply_observational_noise(catalog)

        return catalog

    def _validate_config(self) -> None:
        """
        Valida que los parámetros sean físicamente razonables y que las
        coordenadas del centro y del ápex estén definidas sin ambigüedad.
        """

        config = self.config

        if config.n_members <= 0:
            raise ValueError("n_members must be greater than zero.")

        if config.distance_pc <= 0.0:
            raise ValueError("distance_pc must be greater than zero.")

        if config.radius_pc < 0.0:
            raise ValueError("radius_pc must be non-negative.")

        if config.radius_pc >= config.distance_pc:
            raise ValueError(
                "radius_pc must be smaller than distance_pc. "
                "Otherwise some sources may have non-physical distances."
            )

        if config.speed_kms < 0.0:
            raise ValueError("speed_kms must be non-negative.")

        self._validate_coordinate_choice(
            object_name="cluster center",
            equatorial_lon=config.center_ra_deg,
            equatorial_lat=config.center_dec_deg,
            galactic_lon=config.center_l_deg,
            galactic_lat=config.center_b_deg,
        )

        self._validate_coordinate_choice(
            object_name="apex",
            equatorial_lon=config.apex_ra_deg,
            equatorial_lat=config.apex_dec_deg,
            galactic_lon=config.apex_l_deg,
            galactic_lat=config.apex_b_deg,
        )

        self._validate_latitude_range(
            name="center_dec_deg",
            value=config.center_dec_deg,
        )
        self._validate_latitude_range(
            name="center_b_deg",
            value=config.center_b_deg,
        )
        self._validate_latitude_range(
            name="apex_dec_deg",
            value=config.apex_dec_deg,
        )
        self._validate_latitude_range(
            name="apex_b_deg",
            value=config.apex_b_deg,
        )

        sigma_values = {
            "speed_sigma_kms": config.speed_sigma_kms,
            "apex_l_sigma_deg": config.apex_l_sigma_deg,
            "apex_b_sigma_deg": config.apex_b_sigma_deg,
            "parallax_error_mas": config.parallax_error_mas,
            "proper_motion_error_masyr": config.proper_motion_error_masyr,
            "position_error_mas": config.position_error_mas,
        }

        optional_sigma_values = {
            "pmra_error_masyr": config.pmra_error_masyr,
            "pmdec_error_masyr": config.pmdec_error_masyr,
        }

        for name, value in sigma_values.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        for name, value in optional_sigma_values.items():
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative.")

    @staticmethod
    def _validate_coordinate_choice(
        object_name: str,
        equatorial_lon: Optional[float],
        equatorial_lat: Optional[float],
        galactic_lon: Optional[float],
        galactic_lat: Optional[float],
    ) -> None:
        """
        Valida que una dirección esté definida exactamente en un sistema:
        ecuatorial ICRS o galáctico.
        """

        equatorial_any = (
            equatorial_lon is not None or equatorial_lat is not None
        )
        equatorial_complete = (
            equatorial_lon is not None and equatorial_lat is not None
        )

        galactic_any = galactic_lon is not None or galactic_lat is not None
        galactic_complete = (
            galactic_lon is not None and galactic_lat is not None
        )

        if equatorial_any and not equatorial_complete:
            raise ValueError(
                f"Incomplete equatorial coordinates for {object_name}. "
                "Both RA and Dec must be provided."
            )

        if galactic_any and not galactic_complete:
            raise ValueError(
                f"Incomplete Galactic coordinates for {object_name}. "
                "Both l and b must be provided."
            )

        if equatorial_complete and galactic_complete:
            raise ValueError(
                f"Ambiguous coordinates for {object_name}. "
                "Provide either equatorial coordinates or Galactic coordinates, "
                "not both."
            )

        if not equatorial_complete and not galactic_complete:
            raise ValueError(
                f"Missing coordinates for {object_name}. "
                "Provide either equatorial coordinates or Galactic coordinates."
            )

    @staticmethod
    def _validate_latitude_range(
        name: str,
        value: Optional[float],
    ) -> None:
        """
        Valida declinaciones o latitudes galácticas.
        """

        if value is None:
            return

        if not -90.0 <= value <= 90.0:
            raise ValueError(f"{name} must be between -90 and 90 degrees.")

    def _get_center_coord_icrs(self) -> SkyCoord:
        """
        Devuelve la coordenada del centro del cúmulo en ICRS,
        independientemente del sistema usado en la entrada.
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

    def _get_apex_frame(self) -> Literal["icrs", "galactic"]:
        """
        Indica en qué sistema fue definido el ápex.
        """

        if self.config.apex_ra_deg is not None:
            return "icrs"

        return "galactic"

    def _get_apex_coordinates_deg(self) -> tuple[float, float]:
        """
        Devuelve las dos coordenadas angulares del ápex en el sistema
        escogido por el usuario.

        Si el ápex está en ICRS, devuelve RA, Dec.
        Si el ápex está en galácticas, devuelve l, b.
        """

        config = self.config

        if config.apex_ra_deg is not None:
            return config.apex_ra_deg, config.apex_dec_deg

        return config.apex_l_deg, config.apex_b_deg

    def _build_cluster_positions(self) -> np.ndarray:
        """
        Genera posiciones cartesianas heliocéntricas ICRS en pc.

        Returns
        -------
        np.ndarray
            Arreglo de forma ``(n_members, 3)`` con posiciones en pc.
        """

        config = self.config

        center_coord = self._get_center_coord_icrs()
        center_xyz_pc = center_coord.cartesian.xyz.to_value(u.pc)

        offsets_pc = self._sample_uniform_sphere(
            n_points=config.n_members,
            radius_pc=config.radius_pc,
        )

        return center_xyz_pc[None, :] + offsets_pc

    def _build_cluster_velocities(self) -> np.ndarray:
        """
        Genera velocidades cartesianas heliocéntricas ICRS en km/s.

        Returns
        -------
        np.ndarray
            Arreglo de forma ``(n_members, 3)`` con velocidades en km/s.
        """

        config = self.config

        speeds_kms = self.rng.normal(
            loc=config.speed_kms,
            scale=config.speed_sigma_kms,
            size=config.n_members,
        )
        speeds_kms = np.clip(speeds_kms, a_min=0.0, a_max=None)

        apex_lon_deg, apex_lat_deg = self._get_apex_coordinates_deg()
        apex_frame = self._get_apex_frame()

        sampled_apex_lon_deg = self.rng.normal(
            loc=apex_lon_deg,
            scale=config.apex_l_sigma_deg,
            size=config.n_members,
        )
        sampled_apex_lon_deg = np.mod(sampled_apex_lon_deg, 360.0)

        sampled_apex_lat_deg = self.rng.normal(
            loc=apex_lat_deg,
            scale=config.apex_b_sigma_deg,
            size=config.n_members,
        )
        sampled_apex_lat_deg = np.clip(sampled_apex_lat_deg, -90.0, 90.0)

        apex_unit_vectors = self._apex_to_icrs_unit_vector(
            apex_lon_deg=sampled_apex_lon_deg,
            apex_lat_deg=sampled_apex_lat_deg,
            frame=apex_frame,
        )

        return speeds_kms[:, None] * apex_unit_vectors

    def _sample_uniform_sphere(
        self,
        n_points: int,
        radius_pc: float,
    ) -> np.ndarray:
        """
        Muestrea puntos uniformemente dentro de una esfera 3D.

        Parameters
        ----------
        n_points : int
            Número de puntos.

        radius_pc : float
            Radio máximo de la esfera, en pc.

        Returns
        -------
        np.ndarray
            Offsets cartesianos de forma ``(n_points, 3)`` en pc.
        """

        if radius_pc == 0.0:
            return np.zeros((n_points, 3), dtype=float)

        directions = self.rng.normal(size=(n_points, 3))
        norms = np.linalg.norm(directions, axis=1)

        if np.any(norms == 0.0):
            raise RuntimeError("Random direction with zero norm generated.")

        unit_directions = directions / norms[:, None]
        radii = radius_pc * self.rng.random(n_points) ** (1.0 / 3.0)

        return unit_directions * radii[:, None]

    @staticmethod
    def _apex_to_icrs_unit_vector(
        apex_lon_deg: np.ndarray,
        apex_lat_deg: np.ndarray,
        frame: Literal["icrs", "galactic"],
    ) -> np.ndarray:
        """
        Convierte direcciones de ápex a vectores unitarios ICRS.

        Parameters
        ----------
        apex_lon_deg : np.ndarray
            Coordenada longitudinal del ápex, en grados.
            Si frame='icrs', corresponde a RA.
            Si frame='galactic', corresponde a l.

        apex_lat_deg : np.ndarray
            Coordenada latitudinal del ápex, en grados.
            Si frame='icrs', corresponde a Dec.
            Si frame='galactic', corresponde a b.

        frame : {'icrs', 'galactic'}
            Sistema de coordenadas en el que se definió el ápex.

        Returns
        -------
        np.ndarray
            Vectores unitarios ICRS de forma ``(n_sources, 3)``.
        """

        if frame == "icrs":
            apex_coords = SkyCoord(
                ra=apex_lon_deg * u.deg,
                dec=apex_lat_deg * u.deg,
                frame="icrs",
            )
        elif frame == "galactic":
            apex_coords = SkyCoord(
                l=apex_lon_deg * u.deg,
                b=apex_lat_deg * u.deg,
                frame="galactic",
            ).icrs
        else:
            raise ValueError("frame must be either 'icrs' or 'galactic'.")

        icrs_cartesian = apex_coords.icrs.cartesian

        return np.column_stack(
            (
                icrs_cartesian.x.value,
                icrs_cartesian.y.value,
                icrs_cartesian.z.value,
            )
        )

    @staticmethod
    def _phase_space_to_gaia_observables(
        positions_pc: np.ndarray,
        velocities_kms: np.ndarray,
    ) -> pd.DataFrame:
        """
        Convierte fase espacial cartesiana ICRS a observables tipo Gaia.

        Parameters
        ----------
        positions_pc : np.ndarray
            Posiciones heliocéntricas cartesianas ICRS, en pc.
            Forma esperada: ``(n_sources, 3)``.

        velocities_kms : np.ndarray
            Velocidades heliocéntricas cartesianas ICRS, en km/s.
            Forma esperada: ``(n_sources, 3)``.

        Returns
        -------
        pd.DataFrame
            DataFrame con ra, dec, parallax, pmra y pmdec.
        """

        if positions_pc.shape != velocities_kms.shape:
            raise ValueError(
                "positions_pc and velocities_kms must have the same shape."
            )

        if positions_pc.ndim != 2 or positions_pc.shape[1] != 3:
            raise ValueError(
                "positions_pc and velocities_kms must have shape "
                "(n_sources, 3)."
            )

        x_coord = positions_pc[:, 0]
        y_coord = positions_pc[:, 1]
        z_coord = positions_pc[:, 2]

        distance_pc = np.linalg.norm(positions_pc, axis=1)

        if np.any(distance_pc <= 0.0):
            raise ValueError("All sources must have positive distance.")

        unit_position = positions_pc / distance_pc[:, None]

        ra_rad = np.arctan2(y_coord, x_coord) % (2.0 * np.pi)
        dec_rad = np.arcsin(np.clip(z_coord / distance_pc, -1.0, 1.0))

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

        velocity_ra_kms = np.sum(velocities_kms * basis_ra, axis=1)
        velocity_dec_kms = np.sum(velocities_kms * basis_dec, axis=1)

        radial_velocity_kms = np.sum(
            velocities_kms * unit_position,
            axis=1,
        )

        pmra_masyr = (
            1_000.0
            * velocity_ra_kms
            / (KM_S_PER_ARCSEC_YR_PC * distance_pc)
        )

        pmdec_masyr = (
            1_000.0
            * velocity_dec_kms
            / (KM_S_PER_ARCSEC_YR_PC * distance_pc)
        )

        return pd.DataFrame(
            {
                "ra": np.degrees(ra_rad),
                "dec": np.degrees(dec_rad),
                "parallax": 1_000.0 / distance_pc,
                "pmra": pmra_masyr,
                "pmdec": pmdec_masyr,
                "distance_pc_true": distance_pc,
                "radial_velocity_kms_true": radial_velocity_kms,
            }
        )

    def _apply_observational_noise(
        self,
        catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aplica ruido observacional gaussiano a las columnas tipo Gaia.

        Permite errores separados para pmra y pmdec si existen en la
        configuración:

            pmra_error_masyr
            pmdec_error_masyr

        Si esos valores son None, usa el error común:

            proper_motion_error_masyr

        Parameters
        ----------
        catalog : pd.DataFrame
            Catálogo sin ruido observacional.

        Returns
        -------
        pd.DataFrame
            Catálogo con ruido observacional.
        """

        config = self.config
        result = catalog.copy()

        if config.position_error_mas > 0.0:
            sigma_deg = config.position_error_mas / 3_600_000.0

            result["ra"] += self.rng.normal(
                loc=0.0,
                scale=sigma_deg,
                size=len(result),
            )

            result["dec"] += self.rng.normal(
                loc=0.0,
                scale=sigma_deg,
                size=len(result),
            )

            result["ra"] = np.mod(result["ra"], 360.0)
            result["dec"] = np.clip(result["dec"], -90.0, 90.0)

        if config.parallax_error_mas > 0.0:
            result["parallax"] += self.rng.normal(
                loc=0.0,
                scale=config.parallax_error_mas,
                size=len(result),
            )

        pmra_error_masyr = (
            config.pmra_error_masyr
            if config.pmra_error_masyr is not None
            else config.proper_motion_error_masyr
        )

        pmdec_error_masyr = (
            config.pmdec_error_masyr
            if config.pmdec_error_masyr is not None
            else config.proper_motion_error_masyr
        )

        if pmra_error_masyr > 0.0:
            result["pmra"] += self.rng.normal(
                loc=0.0,
                scale=pmra_error_masyr,
                size=len(result),
            )

        if pmdec_error_masyr > 0.0:
            result["pmdec"] += self.rng.normal(
                loc=0.0,
                scale=pmdec_error_masyr,
                size=len(result),
            )

        return result


# if __name__ == "__main__":
    # ============================================================
    # EJEMPLO 1:
    # Centro en coordenadas ecuatoriales ICRS.
    # Ápex en coordenadas galácticas.
    # ============================================================

    # config = ClusterSimulationConfig(
    #     n_members=455,
    #     center_ra_deg=90.0,
    #     center_dec_deg=0.0,
    #     distance_pc=85.0,
    #     radius_pc=5.0,
    #     apex_l_deg=123.0,
    #     apex_b_deg=27.0,
    #     speed_kms=25.0,
    #     speed_sigma_kms=0.0,
    #     apex_l_sigma_deg=0.0,
    #     apex_b_sigma_deg=0.0,
    #     parallax_error_mas=0.0,
    #     proper_motion_error_masyr=0.0,
    #     position_error_mas=0.0,
    #     seed=42,
    #     include_true_values=True,
    # )

    # ============================================================
    # EJEMPLO 2:
    # Centro en coordenadas galácticas.
    # Ápex en coordenadas ecuatoriales ICRS.
    #
    # Para usar este ejemplo, comenta el config anterior y
    # descomenta este bloque.
    # ============================================================

    # config = ClusterSimulationConfig(
    #     n_members=455,
    #     center_l_deg=221.0,
    #     center_b_deg=84.0,
    #     distance_pc=85.0,
    #     radius_pc=5.0,
    #     apex_ra_deg=180.0,
    #     apex_dec_deg=30.0,
    #     speed_kms=25.0,
    #     speed_sigma_kms=0.0,
    #     apex_l_sigma_deg=0.0,
    #     apex_b_sigma_deg=0.0,
    #     parallax_error_mas=0.0,
    #     proper_motion_error_masyr=0.0,
    #     position_error_mas=0.0,
    #     seed=42,
    #     include_true_values=True,
    # )

    # config = ClusterSimulationConfig(
    #     n_members=455,

    #     # Centro del cúmulo en ecuatoriales ICRS
    #     center_ra_deg=90.0,
    #     center_dec_deg=0.0,

    #     distance_pc=85.0,
    #     radius_pc=5.0,

    #     # Ápex en ecuatoriales ICRS
    #     apex_ra_deg=180.0,
    #     apex_dec_deg=30.0,

    #     speed_kms=25.0,
    #     speed_sigma_kms=0.0,

    #     # En este caso estas sigmas se interpretan como:
    #     # apex_l_sigma_deg -> sigma_RA
    #     # apex_b_sigma_deg -> sigma_Dec
    #     apex_l_sigma_deg=0.0,
    #     apex_b_sigma_deg=0.0,

    #     parallax_error_mas=0.0,
    #     proper_motion_error_masyr=0.0,
    #     position_error_mas=0.0,

    #     seed=42,
    #     include_true_values=True,
    # )

    # simulator = OpenClusterSimulator(config)
    # catalog = simulator.simulate()

    # print(catalog.head())

    # catalog.to_csv(
    #     "mock_open_cluster_gaia_equatorial_only.csv",
    #     index=False,
    # )

    # simulator = OpenClusterSimulator(config)
    # catalog = simulator.simulate()

    # print(catalog.head())

    # catalog.to_csv(
    #     "mock_open_cluster_gaia.csv",
    #     index=False,
    # )