from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

ProfileType = Literal["flat", "ascending", "descending", "stepped"]
InflowMethod = Literal["legacy_empirical", "aliev_2015_anisotropic"]
PwfMode = Literal["pwf_const", "pwf_variable_iterative"]


@dataclass(frozen=True)
class ReservoirParams:
    """Параметры пласта и флюида для физического блока (Шаг 1)."""

    p_res_mpa: float = 9.5
    t_res_c: float = 46.0
    k_mD: float = 345.0
    h_m: float = 40.0
    porosity: float = 0.15
    ae_default: float = 0.2
    macro_l: float = 1.0
    gamma_g: float = 0.693
    gas_composition_mol_frac: dict[str, float] = field(
        default_factory=lambda: {
            "CH4": 0.9623,
            "C2H6": 0.0302,
        }
    )
    # Названия методик:
    # ppc_method: composition_linear | composition_precise | specific_gravity_sutton
    # z_method: dranchuk_abou_kassem | aliev_zotov_two_parameter | papay
    # inflow_method: legacy_empirical | aliev_2015_anisotropic
    # pwf_mode: pwf_const | pwf_variable_iterative
    ppc_method: str = "composition_linear"
    z_method: str = "dranchuk_abou_kassem"
    inflow_method: InflowMethod = "legacy_empirical"
    pwf_mode: PwfMode = "pwf_variable_iterative"
    cgf_g_m3: float = 0.85
    skin: float = 0.0

    def __post_init__(self) -> None:
        """Проверяет базовые физические ограничения параметров пласта."""
        if not (0.0 < self.ae_default <= 1.0):
            raise ValueError("Параметр ae_default должен быть в диапазоне 0 < ae <= 1.")


@dataclass(frozen=True)
class WellboreParams:
    """Геометрия и гидравлические параметры ствола/колонн (Шаги 2-3)."""

    tubing_od_m: float = 0.114
    tubing_id_m: float = 0.102
    casing_id_m: float = 0.156
    roughness_m: float = 1.5e-5
    wellbore_radius_m: float = 0.057
    drainage_radius_m: float = 600.0
    entry_from_roof_m: float = 20.0
    ascending_angle_deg: float = 6.0
    # Параметры ступенчатого профиля (задаются вручную):
    # - stepped_segment_lengths_m: длины горизонтальных ступеней;
    # - stepped_step_heights_m: перепады высоты между соседними ступенями (положительные/отрицательные).
    stepped_segment_lengths_m: Sequence[float] = (300.0, 300.0, 300.0)
    stepped_step_heights_m: Sequence[float] = (1.5, 1.5)

    def __post_init__(self) -> None:
        """Проверяет корректность параметров ступенчатого профиля."""
        if any(length <= 0.0 for length in self.stepped_segment_lengths_m):
            raise ValueError("Все значения stepped_segment_lengths_m должны быть > 0.")
        expected_heights = max(len(self.stepped_segment_lengths_m) - 1, 0)
        if len(self.stepped_step_heights_m) != expected_heights:
            raise ValueError(
                "Длина stepped_step_heights_m должна быть равна (количество ступеней - 1), "
                "то есть len(stepped_segment_lengths_m) - 1."
            )


@dataclass(frozen=True)
class OperatingParams:
    """Режимные параметры расчета (заданный Pз на пятке и стандартные условия)."""

    # Фиксируем забойное давление на пятке для сравнения сценариев с вертикальным эталоном.
    p_wf_heel_mpa: float = 7.5
    std_p_pa: float = 101_325.0
    std_t_k: float = 293.15


@dataclass(frozen=True)
class CalibrationParams:
    """Масштабирующие коэффициенты для последующей калибровки модели (Шаг 7)."""

    a_scale: float = 1.0
    b_scale: float = 1.0


@dataclass(frozen=True)
class SweepParams:
    """Сетка параметров для многовариантного анализа (Шаг 5)."""

    lateral_lengths_m: Sequence[int] = tuple(range(100, 1501, 100))
    curvature_radii_m: Sequence[float] = (8.0, 40.0, 120.0)
    profiles: Sequence[ProfileType] = ("flat", "ascending", "descending", "stepped")
    anisotropy_values: Sequence[float] = (0.1, 0.2, 0.3)
    shoe_step_m: float = 100.0

    def __post_init__(self) -> None:
        """Проверяет диапазон перебираемых значений анизотропии."""
        if any((value <= 0.0 or value > 1.0) for value in self.anisotropy_values):
            raise ValueError("Все значения anisotropy_values должны быть в диапазоне 0 < ae <= 1.")


@dataclass(frozen=True)
class OutputParams:
    """Параметры вывода результатов."""

    output_dir: str = "outputs"


@dataclass(frozen=True)
class Scenario:
    """Описание одного расчетного сценария ЗБС."""

    lateral_length_m: float
    curvature_radius_m: float
    profile: ProfileType
    tubing_shoe_from_heel_m: float
    anisotropy_ae: float

    def __post_init__(self) -> None:
        """Проверяет корректность анизотропии для конкретного сценария."""
        if not (0.0 < self.anisotropy_ae <= 1.0):
            raise ValueError("Параметр anisotropy_ae должен быть в диапазоне 0 < ae <= 1.")


@dataclass(frozen=True)
class BaseConfig:
    """Корневая конфигурация проекта: объединяет все блоки параметров."""

    reservoir: ReservoirParams = field(default_factory=ReservoirParams)
    wellbore: WellboreParams = field(default_factory=WellboreParams)
    operating: OperatingParams = field(default_factory=OperatingParams)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    output: OutputParams = field(default_factory=OutputParams)
