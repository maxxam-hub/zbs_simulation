from __future__ import annotations

import csv
import sys
from pathlib import Path

# Шаг 0. Подключаем локальный пакет src/ к sys.path для запуска скриптом.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zbs_sim import default_config, run_sensitivity
from zbs_sim.visualization import (
    save_anisotropy_plot,
    save_profile_pressure_comparison_plot,
    save_profile_rate_comparison_plot,
    save_pressure_profile_plot,
    save_rate_vs_length_plot,
)


def main() -> None:
    """
    Точка входа MVP-расчета.

    Роль в проекте:
    - Запускает многовариантный расчет, сохраняет таблицу и графики, печатает топ-сценарии.
    """
    # Шаг 1. Загружаем конфигурацию и создаем папку результатов.
    cfg = default_config()
    output_dir = ROOT / cfg.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Метод коэффициентов притока A/B: {cfg.reservoir.inflow_method}")
    print(f"Режим по забойному давлению: {cfg.reservoir.pwf_mode}")
    print(f"Метод псевдокритических параметров: {cfg.reservoir.ppc_method}")
    print(f"Метод расчета Z-фактора: {cfg.reservoir.z_method}")
    if cfg.reservoir.gas_composition_mol_frac:
        print("Используется расширенный компонентный состав газа.")
    else:
        print("Компонентный состав газа не задан, используются fallback-методы по относительной плотности газа.")

    # Шаг 2. Выполняем полный перебор сценариев.
    records = run_sensitivity(cfg)
    csv_path = output_dir / "scenario_results.csv"
    csv_path_ru = output_dir / "scenario_results_ru.csv"
    # Шаг 3. Сохраняем таблицу сценариев в CSV.
    if records:
        fieldnames = list(records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow(row)
        # Шаг 3.1. Формируем русскую версию таблицы для отчета.
        with csv_path_ru.open("w", newline="", encoding="utf-8") as file_ru:
            fields_ru = [
                "ID_сценария",
                "Метод_A_B",
                "Режим_Pз",
                "Метод_Pпкр",
                "Метод_Z",
                "Анизотропия_ae",
                "Профиль",
                "Класс_Rкр",
                "Rкр_м",
                "Lг_м",
                "Глубина_башмака_от_пятки_м",
                "Доля_покрытия_НКТ",
                "Дебит_стм3_сут",
                "Дебит_тыс_стм3_сут",
                "Pз_пятка_МПа",
                "P_сред_по_ГС_МПа",
                "P_забой_носок_МПа",
                "Потери_в_стволе_кПа",
                "Коэффициент_A",
                "Коэффициент_B",
                "Итераций",
                "Вертикальный_эталон_стм3_сут",
                "Прирост_к_вертикальной_проц",
            ]
            writer_ru = csv.DictWriter(file_ru, fieldnames=fields_ru)
            writer_ru.writeheader()
            for row in records:
                writer_ru.writerow(
                    {
                        "ID_сценария": row["scenario_id"],
                        "Метод_A_B": row["inflow_method"],
                        "Режим_Pз": row["pwf_mode"],
                        "Метод_Pпкр": row["ppc_method"],
                        "Метод_Z": row["z_method"],
                        "Анизотропия_ae": row["ae"],
                        "Профиль": row["profile_ru"],
                        "Класс_Rкр": row["curvature_class"],
                        "Rкр_м": row["curvature_radius_m"],
                        "Lг_м": row["lateral_length_m"],
                        "Глубина_башмака_от_пятки_м": row["tubing_shoe_from_heel_m"],
                        "Доля_покрытия_НКТ": row["shoe_fraction"],
                        "Дебит_стм3_сут": row["q_std_m3_day"],
                        "Дебит_тыс_стм3_сут": row["q_std_th_m3_day"],
                        "Pз_пятка_МПа": row["p_heel_mpa"],
                        "P_сред_по_ГС_МПа": row["p_avg_lateral_mpa"],
                        "P_забой_носок_МПа": row["p_toe_mpa"],
                        "Потери_в_стволе_кПа": row["delta_p_wellbore_kpa"],
                        "Коэффициент_A": row["a_coeff"],
                        "Коэффициент_B": row["b_coeff"],
                        "Итераций": row["iterations"],
                        "Вертикальный_эталон_стм3_сут": row["q_vertical_ref_m3_day"],
                        "Прирост_к_вертикальной_проц": row["gain_vs_vertical_pct"],
                    }
                )

    # Шаг 4. Формируем три целевых графика из задания.
    graph_1 = save_rate_vs_length_plot(records, cfg, output_dir / "plot_1_rate_vs_lateral_length.png")
    graph_2 = save_pressure_profile_plot(cfg, output_dir / "plot_2_pressure_profile_vs_shoe_depth.png")
    graph_3 = save_anisotropy_plot(records, cfg, output_dir / "plot_3_anisotropy_vs_lateral_length.png")
    graph_4 = save_profile_rate_comparison_plot(records, cfg, output_dir / "plot_4_profile_rate_comparison.png")
    graph_5 = save_profile_pressure_comparison_plot(cfg, output_dir / "plot_5_profile_pressure_comparison.png")

    # Шаг 5. Выводим краткую сводку по лучшим сценариям.
    top = sorted(records, key=lambda item: float(item["q_std_m3_day"]), reverse=True)[:10]
    print(f"Смоделировано сценариев: {len(records)}")
    print(f"Таблица результатов: {csv_path}")
    print(f"Таблица результатов (рус.): {csv_path_ru}")
    if not (graph_1 and graph_2 and graph_3 and graph_4 and graph_5):
        print("matplotlib недоступен: вместо PNG сохранены CSV-данные для построения графиков.")
    print("Топ-10 по прогнозному дебиту (ст.м3/сут):")
    for row in top:
        profile_ru = {
            "flat": "плоско-горизонтальный",
            "ascending": "восходящий",
            "descending": "нисходящий",
            "stepped": "ступенчатый",
        }.get(row["profile"], row["profile"])
        print(
            f"{int(row['scenario_id']):5d} | "
            f"Q={float(row['q_std_m3_day']):12.2f} | "
            f"L={float(row['lateral_length_m']):6.1f} | "
            f"R={float(row['curvature_radius_m']):6.1f} | "
            f"профиль={profile_ru:21s} | "
            f"башмак={float(row['tubing_shoe_from_heel_m']):6.1f} | "
            f"ae={float(row['ae']):.2f} | "
            f"прирост к вертик., %={float(row['gain_vs_vertical_pct']):8.2f}"
        )


if __name__ == "__main__":
    main()
