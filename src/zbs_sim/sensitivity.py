from __future__ import annotations

from .flow_engine import simulate_scenario, simulate_vertical_reference
from .geometry import curvature_class, generate_shoe_depths
from .models import BaseConfig, Scenario


def run_sensitivity(cfg: BaseConfig) -> list[dict]:
    """
    Выполняет многовариантный анализ по сетке параметров и возвращает таблицу сценариев.

    Роль в проекте:
    - Реализация шага "Sensitivity Analysis" из вашего плана.
    """
    # Шаг 1. Инициализируем контейнер результатов и считаем вертикальный эталон.
    records: list[dict] = []
    q_vertical_m3_day = simulate_vertical_reference(cfg)

    # Шаг 2. Перебираем все комбинации: ae, профиль, Rкр, Lг и глубина башмака.
    scenario_id = 0
    for ae in cfg.sweep.anisotropy_values:
        for profile in cfg.sweep.profiles:
            for radius in cfg.sweep.curvature_radii_m:
                for lateral in cfg.sweep.lateral_lengths_m:
                    # Шаг 2.1. Для каждого Lг формируем допустимые положения башмака НКТ.
                    shoes = generate_shoe_depths(float(lateral), cfg.sweep.shoe_step_m)
                    for shoe in shoes:
                        scenario_id += 1
                        # Шаг 2.2. Формируем объект сценария.
                        scenario = Scenario(
                            lateral_length_m=float(lateral),
                            curvature_radius_m=float(radius),
                            profile=profile,
                            tubing_shoe_from_heel_m=float(shoe),
                            anisotropy_ae=float(ae),
                        )
                        # Шаг 2.3. Запускаем расчетный движок.
                        result = simulate_scenario(cfg, scenario)
                        shoe_fraction = scenario.tubing_shoe_from_heel_m / scenario.lateral_length_m
                        # Шаг 2.4. Сохраняем метрики сценария в строку выходной таблицы.
                        records.append(
                            {
                                "scenario_id": scenario_id,
                                "ae": scenario.anisotropy_ae,
                                "profile": scenario.profile,
                                "profile_ru": "плоско-горизонтальный" if scenario.profile == "flat" else "восходящий",
                                "curvature_radius_m": scenario.curvature_radius_m,
                                "curvature_class": curvature_class(scenario.curvature_radius_m),
                                "lateral_length_m": scenario.lateral_length_m,
                                "tubing_shoe_from_heel_m": scenario.tubing_shoe_from_heel_m,
                                "shoe_fraction": shoe_fraction,
                                "q_std_m3_day": result.q_std_m3_day,
                                "q_std_th_m3_day": result.q_std_m3_day / 1000.0,
                                "p_heel_mpa": result.p_heel_mpa,
                                "p_avg_lateral_mpa": result.p_avg_lateral_mpa,
                                "p_toe_mpa": result.p_toe_mpa,
                                "delta_p_wellbore_kpa": result.delta_p_wellbore_kpa,
                                "a_coeff": result.a_coeff,
                                "b_coeff": result.b_coeff,
                                "iterations": result.iterations,
                                "q_vertical_ref_m3_day": q_vertical_m3_day,
                                "gain_vs_vertical_pct": (
                                    (result.q_std_m3_day - q_vertical_m3_day) / q_vertical_m3_day * 100.0
                                    if q_vertical_m3_day > 0.0
                                    else 0.0
                                ),
                            }
                        )

    # Шаг 3. Возвращаем полную таблицу сценариев.
    return records
