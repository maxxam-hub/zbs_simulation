from __future__ import annotations

import math

from .geometry import resistance_term
from .models import BaseConfig, Scenario
from .reservoir import c_to_k, gas_properties, md_to_m2


def _pvt_for_inflow(cfg: BaseConfig, p_avg_pa: float) -> tuple[float, float, float]:
    """
    Возвращает PVT-параметры для расчета коэффициентов притока A/B.

    Возвращает:
    - t_k: пластовая температура в K;
    - z: коэффициент сверхсжимаемости;
    - mu: вязкость газа, Па*с.
    """
    t_k = c_to_k(cfg.reservoir.t_res_c)
    z, mu, _ = gas_properties(
        pressure_pa=p_avg_pa,
        temperature_k=t_k,
        gamma_g=cfg.reservoir.gamma_g,
        composition_mol_frac=cfg.reservoir.gas_composition_mol_frac,
        ppc_method=cfg.reservoir.ppc_method,
        z_method=cfg.reservoir.z_method,
    )
    return t_k, z, mu


def compute_ab_legacy(cfg: BaseConfig, scenario: Scenario, p_avg_pa: float) -> tuple[float, float]:
    """
    Считает A/B исходной эмпирической схемой проекта (без изменения прежней логики).

    Роль:
    - гарантирует обратную совместимость старых результатов.
    """
    # Шаг 1. Получаем PVT-блок.
    t_k, z, mu = _pvt_for_inflow(cfg, p_avg_pa)
    # Шаг 2. Считаем геометрическое сопротивление legacy-моделью.
    psi = resistance_term(cfg.reservoir, cfg.wellbore, scenario)
    k_m2 = md_to_m2(cfg.reservoir.k_mD)

    # Шаг 3. Линейный коэффициент A.
    a_coeff = (
        cfg.calibration.a_scale
        * mu
        * z
        * cfg.operating.std_p_pa
        * t_k
        * psi
        / (math.pi * k_m2 * cfg.reservoir.h_m * cfg.operating.std_t_k)
    )

    # Шаг 4. Квадратичный коэффициент B.
    b_coeff = (
        cfg.calibration.b_scale
        * cfg.reservoir.macro_l
        * z
        * cfg.operating.std_p_pa
        * t_k
        * (psi**1.1)
        / (
            math.pi
            * math.pi
            * k_m2
            * cfg.reservoir.h_m
            * max(scenario.lateral_length_m, 1.0)
            * cfg.operating.std_t_k
        )
    )
    return a_coeff, b_coeff


def _aliev_2015_resistance_term(cfg: BaseConfig, scenario: Scenario) -> float:
    """
    Возвращает геометрический множитель сопротивления для анизотропного ГС.

    Примечание:
    - реализована инженерная адаптация под гл. 3 Алиева (2015) для условий курсовой:
      явный учет анизотропии через ae = sqrt(kv/kh), с отдельным анизотропным членом.
    """
    # Шаг 1. Подготовка параметров геометрии.
    re = max(cfg.wellbore.drainage_radius_m, 1.0)
    rw = max(cfg.wellbore.wellbore_radius_m, 1e-4)
    h = max(cfg.reservoir.h_m, 0.1)
    lateral = max(scenario.lateral_length_m, 1.0)
    ae = scenario.anisotropy_ae
    beta = 1.0 / ae

    # Шаг 2. Геометрический член для горизонтального ствола (Joshi-подобная форма).
    half_l = 0.5 * lateral
    shape_factor = 0.5 + math.sqrt(0.25 + (2.0 * re / lateral) ** 4)
    a_eq = half_l * shape_factor
    geom_argument = (a_eq + math.sqrt(max(a_eq * a_eq - half_l * half_l, 1e-12))) / max(half_l, 1e-12)
    geom_term = math.log(max(geom_argument, 1.0000001))

    # Шаг 3. Явный анизотропный член через beta = sqrt(kh/kv) = 1/ae.
    anis_argument = max(beta * h / (2.0 * rw), 1.0000001)
    anis_term = (beta * h / lateral) * math.log(anis_argument)

    # Шаг 4. Поправки на положение входа и радиус кривизны.
    asymmetry = abs(cfg.wellbore.entry_from_roof_m - 0.5 * h) / max(0.5 * h, 0.1)
    asymmetry_mult = 1.0 + 0.4 * asymmetry
    curvature_mult = 1.0 + 0.25 * min(1.0, 12.0 / max(scenario.curvature_radius_m, 1.0))

    # Шаг 5. Сборка итогового сопротивления с skin.
    psi = (geom_term + anis_term + cfg.reservoir.skin) * asymmetry_mult * curvature_mult
    return max(psi, 0.2)


def compute_ab_aliev_2015(cfg: BaseConfig, scenario: Scenario, p_avg_pa: float) -> tuple[float, float]:
    """
    Считает A/B по анизотропной постановке (инженерная реализация под Алиев, 2015).

    Источники:
    - Алиев (2015), гл. 3, стр. 96-104, формулы 3.6-3.19;
    - двучленная форма притока по квадратам давлений.
    """
    # Шаг 1. Получаем PVT-параметры и сопротивление по анизотропной схеме.
    t_k, z, mu = _pvt_for_inflow(cfg, p_avg_pa)
    psi = _aliev_2015_resistance_term(cfg, scenario)
    k_m2 = md_to_m2(cfg.reservoir.k_mD)

    # Шаг 2. Линейный коэффициент A.
    a_coeff = (
        cfg.calibration.a_scale
        * mu
        * z
        * cfg.operating.std_p_pa
        * t_k
        * psi
        / (math.pi * k_m2 * cfg.reservoir.h_m * cfg.operating.std_t_k)
    )

    # Шаг 3. Квадратичный коэффициент B.
    b_coeff = (
        cfg.calibration.b_scale
        * cfg.reservoir.macro_l
        * z
        * cfg.operating.std_p_pa
        * t_k
        * (psi**1.1)
        / (
            math.pi
            * math.pi
            * k_m2
            * cfg.reservoir.h_m
            * max(scenario.lateral_length_m, 1.0)
            * cfg.operating.std_t_k
        )
    )
    return a_coeff, b_coeff


def compute_ab(cfg: BaseConfig, scenario: Scenario, p_avg_pa: float) -> tuple[float, float]:
    """
    Выбирает методику расчета A/B по конфигурации без изменения остального движка.
    """
    method = cfg.reservoir.inflow_method.strip().lower()
    if method == "legacy_empirical":
        return compute_ab_legacy(cfg, scenario, p_avg_pa)
    if method == "aliev_2015_anisotropic":
        return compute_ab_aliev_2015(cfg, scenario, p_avg_pa)
    raise ValueError(
        f"Неизвестный inflow_method='{cfg.reservoir.inflow_method}'. Допустимо: "
        "legacy_empirical | aliev_2015_anisotropic."
    )
