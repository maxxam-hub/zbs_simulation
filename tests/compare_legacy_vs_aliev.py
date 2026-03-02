from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zbs_sim import default_config
from zbs_sim.flow_engine import simulate_scenario
from zbs_sim.models import Scenario


def _run_one(cfg, scenario: Scenario) -> dict:
    result = simulate_scenario(cfg, scenario)
    return {
        "q_std_m3_day": result.q_std_m3_day,
        "p_avg_lateral_mpa": result.p_avg_lateral_mpa,
        "delta_p_wellbore_kpa": result.delta_p_wellbore_kpa,
        "a_coeff": result.a_coeff,
        "b_coeff": result.b_coeff,
    }


def main() -> None:
    cfg_base = default_config()
    scenarios = [
        Scenario(300.0, 8.0, "flat", 300.0, 0.1),
        Scenario(600.0, 40.0, "flat", 300.0, 0.2),
        Scenario(900.0, 120.0, "ascending", 450.0, 0.2),
        Scenario(1200.0, 40.0, "descending", 1200.0, 0.3),
        Scenario(1500.0, 120.0, "stepped", 1500.0, 0.3),
    ]

    rows: list[dict] = []
    for idx, scenario in enumerate(scenarios, start=1):
        for inflow_method in ("legacy_empirical", "aliev_2015_anisotropic"):
            for pwf_mode in ("pwf_const", "pwf_variable_iterative"):
                cfg = replace(
                    cfg_base,
                    reservoir=replace(cfg_base.reservoir, inflow_method=inflow_method, pwf_mode=pwf_mode),
                )
                metrics = _run_one(cfg, scenario)
                rows.append(
                    {
                        "case_id": idx,
                        "inflow_method": inflow_method,
                        "pwf_mode": pwf_mode,
                        "lateral_length_m": scenario.lateral_length_m,
                        "curvature_radius_m": scenario.curvature_radius_m,
                        "profile": scenario.profile,
                        "tubing_shoe_from_heel_m": scenario.tubing_shoe_from_heel_m,
                        "ae": scenario.anisotropy_ae,
                        **metrics,
                    }
                )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "method_compare_5_cases.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Сохранен сравнительный отчет: {output_path}")
    print(f"Всего строк: {len(rows)}")


if __name__ == "__main__":
    main()
