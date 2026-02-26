from __future__ import annotations

import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

from .completion import pressure_profile_heel_to_toe
from .flow_engine import simulate_scenario
from .models import BaseConfig, Scenario


def _isclose(left: float, right: float, tol: float = 1e-9) -> bool:
    """Проверяет близость чисел с заданным допуском."""
    return abs(left - right) <= tol


def _write_records_csv(path: Path, records: list[dict], fieldnames: list[str]) -> None:
    """
    Сохраняет набор записей в CSV.

    Роль в проекте:
    - Резервный экспорт данных для графиков, когда matplotlib недоступен.
    """
    # Шаг 1. Открываем выходной файл.
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Шаг 2. Записываем заголовок и строки.
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_rate_vs_length_plot(records: list[dict], cfg: BaseConfig, output_path: Path) -> bool:
    """
    Строит график зависимости дебита от длины ГС при разных Rкр.

    Роль в проекте:
    - Визуализация №1: поиск "полки" эффективности по Lг.
    """
    # Шаг 1. Отбираем сопоставимые сценарии (базовый ae, плоский профиль, башмак у забоя).
    selection = [
        row
        for row in records
        if _isclose(float(row["ae"]), cfg.reservoir.ae_default)
        and row["profile"] == "flat"
        and _isclose(float(row["shoe_fraction"]), 1.0)
    ]
    if not selection:
        return False

    # Шаг 2. Если matplotlib недоступен, сохраняем данные для построения графика в CSV.
    if plt is None:
        rows_ru = [
            {
                "Радиус_кривизны_м": row["curvature_radius_m"],
                "Длина_ГС_м": row["lateral_length_m"],
                "Дебит_тыс_стм3_сут": row["q_std_th_m3_day"],
            }
            for row in selection
        ]
        _write_records_csv(
            output_path.with_suffix(".csv"),
            rows_ru,
            ["Радиус_кривизны_м", "Длина_ГС_м", "Дебит_тыс_стм3_сут"],
        )
        return False

    # Шаг 3. Строим линии по каждому радиусу кривизны.
    plt.figure(figsize=(10, 5))
    radii = sorted({float(row["curvature_radius_m"]) for row in selection})
    for radius in radii:
        sub = sorted(
            [row for row in selection if _isclose(float(row["curvature_radius_m"]), radius)],
            key=lambda item: float(item["lateral_length_m"]),
        )
        xs = [float(row["lateral_length_m"]) for row in sub]
        ys = [float(row["q_std_th_m3_day"]) for row in sub]
        plt.plot(xs, ys, marker="o", label=f"Rкр={radius:.0f} м")

    # Шаг 4. Оформляем и сохраняем график.
    plt.xlabel("Длина горизонтального участка, м")
    plt.ylabel("Дебит, тыс. ст.м3/сут")
    plt.title("Зависимость дебита от длины ГС при разных радиусах кривизны")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def save_pressure_profile_plot(cfg: BaseConfig, output_path: Path) -> bool:
    """
    Строит профиль давления вдоль ствола при разных глубинах башмака НКТ.

    Роль в проекте:
    - Визуализация №2: обоснование оптимальной глубины спуска НКТ.
    """
    # Шаг 1. Выбираем типовой сценарий для иллюстрации гидравлики.
    lateral = 900.0
    radius = 40.0
    profile = "flat"
    ae = cfg.reservoir.ae_default
    shoe_depths = [0.0, 0.5 * lateral, lateral]

    # Шаг 2. При отсутствии matplotlib сохраняем профиль давления в CSV.
    if plt is None:
        fallback_rows: list[dict] = []
        for shoe in shoe_depths:
            scenario = Scenario(
                lateral_length_m=lateral,
                curvature_radius_m=radius,
                profile=profile,
                tubing_shoe_from_heel_m=shoe,
                anisotropy_ae=ae,
            )
            result = simulate_scenario(cfg, scenario)
            xs, ps = pressure_profile_heel_to_toe(cfg, scenario, q_std_m3s=result.q_std_m3_day / 86400.0)
            for x, p in zip(xs, ps):
                fallback_rows.append(
                    {
                        "Глубина_башмака_м": shoe,
                        "Расстояние_от_пятки_м": x,
                        "Давление_МПа": p,
                        "Дебит_тыс_стм3_сут": result.q_std_m3_day / 1000.0,
                    }
                )
        _write_records_csv(
            output_path.with_suffix(".csv"),
            fallback_rows,
            ["Глубина_башмака_м", "Расстояние_от_пятки_м", "Давление_МПа", "Дебит_тыс_стм3_сут"],
        )
        return False

    # Шаг 3. Строим кривую давления для каждого положения башмака.
    plt.figure(figsize=(10, 5))
    for shoe in shoe_depths:
        scenario = Scenario(
            lateral_length_m=lateral,
            curvature_radius_m=radius,
            profile=profile,
            tubing_shoe_from_heel_m=shoe,
            anisotropy_ae=ae,
        )
        result = simulate_scenario(cfg, scenario)
        xs, ps = pressure_profile_heel_to_toe(cfg, scenario, q_std_m3s=result.q_std_m3_day / 86400.0)
        label = f"Башмак={shoe:.0f} м; Q={result.q_std_m3_day/1000.0:.1f} тыс."
        plt.plot(xs, ps, label=label)

    # Шаг 4. Оформляем и сохраняем график.
    plt.xlabel("Расстояние от пятки к забою, м")
    plt.ylabel("Давление, МПа")
    plt.title("Профиль давления при разной глубине башмака НКТ")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def save_anisotropy_plot(records: list[dict], cfg: BaseConfig, output_path: Path) -> bool:
    """
    Строит влияние анизотропии на зависимость дебита от длины ГС.

    Роль в проекте:
    - Визуализация №3: анализ влияния ae на выбор Lг.
    """
    # Шаг 1. Отбираем сопоставимые сценарии (фиксированные Rкр, профиль и положение башмака).
    radius = 40.0
    selection = [
        row
        for row in records
        if _isclose(float(row["curvature_radius_m"]), radius)
        and row["profile"] == "flat"
        and _isclose(float(row["shoe_fraction"]), 1.0)
    ]
    if not selection:
        return False

    # Шаг 2. При отсутствии matplotlib сохраняем исходные данные в CSV.
    if plt is None:
        rows_ru = [
            {
                "Анизотропия_ae": row["ae"],
                "Длина_ГС_м": row["lateral_length_m"],
                "Дебит_тыс_стм3_сут": row["q_std_th_m3_day"],
            }
            for row in selection
        ]
        _write_records_csv(
            output_path.with_suffix(".csv"),
            rows_ru,
            ["Анизотропия_ae", "Длина_ГС_м", "Дебит_тыс_стм3_сут"],
        )
        return False

    # Шаг 3. Строим линии по значениям ae.
    plt.figure(figsize=(10, 5))
    ae_values = sorted({float(row["ae"]) for row in selection})
    for ae in ae_values:
        sub = sorted(
            [row for row in selection if _isclose(float(row["ae"]), ae)],
            key=lambda item: float(item["lateral_length_m"]),
        )
        xs = [float(row["lateral_length_m"]) for row in sub]
        ys = [float(row["q_std_th_m3_day"]) for row in sub]
        plt.plot(xs, ys, marker="o", label=f"ae={ae:.2f}")

    # Шаг 4. Оформляем и сохраняем график.
    plt.xlabel("Длина горизонтального участка, м")
    plt.ylabel("Дебит, тыс. ст.м3/сут")
    plt.title("Влияние анизотропии на выбор длины ГС")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True
