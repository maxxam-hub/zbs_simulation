from __future__ import annotations

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


def _calc_q(cfg, scenario: Scenario) -> float:
    result = simulate_scenario(cfg, scenario)
    return result.q_std_m3_day


def main() -> None:
    cfg_base = default_config()
    base_scenario = Scenario(
        lateral_length_m=900.0,
        curvature_radius_m=40.0,
        profile="flat",
        tubing_shoe_from_heel_m=900.0,
        anisotropy_ae=0.2,
    )

    # Проверка 1. Новый метод притока считается без ошибок.
    cfg_aliev = replace(cfg_base, reservoir=replace(cfg_base.reservoir, inflow_method="aliev_2015_anisotropic"))
    q_aliev = _calc_q(cfg_aliev, base_scenario)
    if q_aliev <= 0.0:
        raise AssertionError("Проверка 1 не пройдена: q(aliev_2015_anisotropic) <= 0.")

    # Проверка 2. В среднем pwf_variable_iterative дает меньший или равный дебит, чем pwf_const.
    cfg_const = replace(cfg_aliev, reservoir=replace(cfg_aliev.reservoir, pwf_mode="pwf_const"))
    cfg_iter = replace(cfg_aliev, reservoir=replace(cfg_aliev.reservoir, pwf_mode="pwf_variable_iterative"))
    q_const = _calc_q(cfg_const, base_scenario)
    q_iter = _calc_q(cfg_iter, base_scenario)
    if q_iter - q_const > 1e-6:
        raise AssertionError("Проверка 2 не пройдена: q_iterative > q_const.")

    # Проверка 3. Физические тренды по длине и анизотропии.
    scenario_short = replace(base_scenario, lateral_length_m=200.0, tubing_shoe_from_heel_m=200.0)
    scenario_long = replace(base_scenario, lateral_length_m=1200.0, tubing_shoe_from_heel_m=1200.0)
    q_short = _calc_q(cfg_iter, scenario_short)
    q_long = _calc_q(cfg_iter, scenario_long)
    if q_long + 1e-6 < q_short:
        raise AssertionError("Проверка 3.1 не пройдена: q(long) < q(short).")

    scenario_ae_low = replace(base_scenario, anisotropy_ae=0.1)
    scenario_ae_high = replace(base_scenario, anisotropy_ae=0.3)
    q_ae_low = _calc_q(cfg_iter, scenario_ae_low)
    q_ae_high = _calc_q(cfg_iter, scenario_ae_high)
    if q_ae_high + 1e-6 < q_ae_low:
        raise AssertionError("Проверка 3.2 не пройдена: q(ae=0.3) < q(ae=0.1).")

    # Проверка 4. Новые профили (descending, stepped) считаются без ошибок.
    scenario_desc = replace(base_scenario, profile="descending")
    scenario_step = replace(base_scenario, profile="stepped")
    q_desc = _calc_q(cfg_iter, scenario_desc)
    q_step = _calc_q(cfg_iter, scenario_step)
    if q_desc <= 0.0 or q_step <= 0.0:
        raise AssertionError("Проверка 4 не пройдена: descending/stepped дали некорректный дебит.")

    print("OK: проверки методик пройдены.")
    print(f"q_aliev={q_aliev:.3f}")
    print(f"q_const={q_const:.3f}, q_iter={q_iter:.3f}")
    print(f"q_short={q_short:.3f}, q_long={q_long:.3f}")
    print(f"q_ae_low={q_ae_low:.3f}, q_ae_high={q_ae_high:.3f}")
    print(f"q_desc={q_desc:.3f}, q_step={q_step:.3f}")


if __name__ == "__main__":
    main()
