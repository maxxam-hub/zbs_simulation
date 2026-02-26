from __future__ import annotations

import math
from dataclasses import dataclass

from .completion import WellboreLosses, compute_wellbore_losses
from .geometry import resistance_term, vertical_resistance_term
from .models import BaseConfig, Scenario
from .reservoir import c_to_k, gas_properties, md_to_m2, mpa_to_pa


@dataclass(frozen=True)
class FlowResult:
    """Результат расчета одного сценария ГС."""

    q_std_m3_day: float
    p_heel_mpa: float
    p_avg_lateral_mpa: float
    p_toe_mpa: float
    delta_p_wellbore_kpa: float
    a_coeff: float
    b_coeff: float
    iterations: int
    losses: WellboreLosses


def _quadratic_positive_root(delta_p2: float, a_coeff: float, b_coeff: float) -> float:
    """
    Решает двучленное уравнение притока и возвращает физически допустимый корень q >= 0.

    Формула:
    - delta_p2 = A*q + B*q^2
    """
    # Шаг 1. Обработка случаев без движущей силы или с некорректным A.
    if delta_p2 <= 0.0 or a_coeff <= 0.0:
        return 0.0
    # Шаг 2. Линейный частный случай при B<=0.
    if b_coeff <= 0.0:
        return delta_p2 / a_coeff
    # Шаг 3. Квадратное решение и выбор положительного корня.
    discriminant = a_coeff * a_coeff + 4.0 * b_coeff * delta_p2
    return max((-a_coeff + math.sqrt(max(discriminant, 0.0))) / (2.0 * b_coeff), 0.0)


def _coefficients_for_scenario(cfg: BaseConfig, scenario: Scenario, p_avg_pa: float) -> tuple[float, float]:
    """
    Вычисляет коэффициенты A и B двучленного уравнения притока для текущего сценария.

    Роль в проекте:
    - Центральная часть расчетного движка (Шаг 4 вашего плана).

    Источники:
    - «2.3 АН 2021», стр. 50: форма двучленного уравнения.
    - Алиев (1995), стр. 56: инженерный вид A/B-параметризации.
    - «муг хитёв.pdf», стр. 26: коэффициент макрошероховатости l для инерционного члена B.
    """
    # Шаг 1. Получаем PVT-свойства газа на среднем давлении сценария.
    t_k = c_to_k(cfg.reservoir.t_res_c)
    z, mu, _ = gas_properties(
        pressure_pa=p_avg_pa,
        temperature_k=t_k,
        gamma_g=cfg.reservoir.gamma_g,
        methane_mol_frac=cfg.reservoir.methane_mol_frac,
        ethane_mol_frac=cfg.reservoir.ethane_mol_frac,
        z_method=cfg.reservoir.z_method,
    )
    # Шаг 2. Переводим проницаемость в SI и считаем геометрический множитель сопротивления.
    k_m2 = md_to_m2(cfg.reservoir.k_mD)
    psi = resistance_term(cfg.reservoir, cfg.wellbore, scenario)

    # Шаг 3. Считаем линейный коэффициент A (вязкостная часть сопротивления).
    a = (
        cfg.calibration.a_scale
        * mu
        * z
        * cfg.operating.std_p_pa
        * t_k
        * psi
        / (math.pi * k_m2 * cfg.reservoir.h_m * cfg.operating.std_t_k)
    )

    # Шаг 4. Считаем квадратичный коэффициент B (инерционная часть).
    b = (
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
    return a, b


def simulate_scenario(cfg: BaseConfig, scenario: Scenario, max_iter: int = 30) -> FlowResult:
    """
    Рассчитывает дебит и давления для одного сценария ГС при заданном Pз на пятке.

    Роль в проекте:
    - Связывает приток из пласта и гидравлику ствола через итерации.
    """
    # Шаг 1. Подготавливаем давления в SI.
    p_res_pa = mpa_to_pa(cfg.reservoir.p_res_mpa)
    p_heel_pa = mpa_to_pa(cfg.operating.p_wf_heel_mpa)

    # Шаг 2. Получаем начальную оценку q без учета перерасчета среднего давления вдоль ствола.
    a0, b0 = _coefficients_for_scenario(cfg, scenario, 0.5 * (p_res_pa + p_heel_pa))
    q_std_m3s = _quadratic_positive_root(p_res_pa * p_res_pa - p_heel_pa * p_heel_pa, a0, b0)

    a_coeff = a0
    b_coeff = b0
    # Шаг 3. Считаем начальные потери давления в стволе.
    losses = compute_wellbore_losses(cfg, scenario, q_std_m3s, pressure_ref_pa=0.5 * (p_res_pa + p_heel_pa))

    # Шаг 4. Итерационно согласуем приток и стволовые потери.
    for iteration in range(1, max_iter + 1):
        # Шаг 4.1. Обновляем среднее давление по горизонтальному участку.
        p_toe_pa = p_heel_pa + losses.total_drop_pa
        p_avg_lateral_pa = max(0.5 * (p_heel_pa + p_toe_pa), 100_000.0)

        # Шаг 4.2. Пересчитываем дебит по обновленному перепаду квадратов давления.
        drawdown_p2 = p_res_pa * p_res_pa - p_avg_lateral_pa * p_avg_lateral_pa
        if drawdown_p2 <= 0.0:
            q_new = 0.0
        else:
            a_coeff, b_coeff = _coefficients_for_scenario(cfg, scenario, 0.5 * (p_res_pa + p_avg_lateral_pa))
            q_new = _quadratic_positive_root(drawdown_p2, a_coeff, b_coeff)

        # Шаг 4.3. Проверяем критерий сходимости по дебиту.
        if q_std_m3s > 0.0 and abs(q_new - q_std_m3s) / q_std_m3s < 1e-4:
            q_std_m3s = q_new
            break

        # Шаг 4.4. Выполняем релаксацию и пересчитываем потери.
        q_std_m3s = 0.5 * q_std_m3s + 0.5 * q_new
        losses = compute_wellbore_losses(cfg, scenario, q_std_m3s, pressure_ref_pa=max(p_avg_lateral_pa, 100_000.0))
    else:
        iteration = max_iter

    # Шаг 5. Финальный пересчет выходных параметров после завершения итераций.
    losses = compute_wellbore_losses(cfg, scenario, q_std_m3s, pressure_ref_pa=0.5 * (p_res_pa + p_heel_pa))
    p_toe_pa = p_heel_pa + losses.total_drop_pa
    p_avg_lateral_pa = 0.5 * (p_heel_pa + p_toe_pa)

    # Шаг 6. Упаковываем результат в удобную структуру.
    return FlowResult(
        q_std_m3_day=q_std_m3s * 86400.0,
        p_heel_mpa=cfg.operating.p_wf_heel_mpa,
        p_avg_lateral_mpa=p_avg_lateral_pa / 1_000_000.0,
        p_toe_mpa=p_toe_pa / 1_000_000.0,
        delta_p_wellbore_kpa=losses.total_drop_pa / 1000.0,
        a_coeff=a_coeff,
        b_coeff=b_coeff,
        iterations=iteration,
        losses=losses,
    )


def simulate_vertical_reference(cfg: BaseConfig) -> float:
    """
    Считает опорный дебит вертикальной скважины при том же заданном Pз.

    Роль в проекте:
    - Базовая точка сравнения эффективности вариантов ЗБС.
    """
    # Шаг 1. Подготавливаем входные давления.
    p_res_pa = mpa_to_pa(cfg.reservoir.p_res_mpa)
    p_wf_pa = mpa_to_pa(cfg.operating.p_wf_heel_mpa)
    p_avg_pa = 0.5 * (p_res_pa + p_wf_pa)

    # Шаг 2. Получаем PVT и геометрию вертикальной скважины.
    t_k = c_to_k(cfg.reservoir.t_res_c)
    z, mu, _ = gas_properties(
        pressure_pa=p_avg_pa,
        temperature_k=t_k,
        gamma_g=cfg.reservoir.gamma_g,
        methane_mol_frac=cfg.reservoir.methane_mol_frac,
        ethane_mol_frac=cfg.reservoir.ethane_mol_frac,
        z_method=cfg.reservoir.z_method,
    )
    k_m2 = md_to_m2(cfg.reservoir.k_mD)
    psi_v = vertical_resistance_term(cfg.reservoir, cfg.wellbore)

    # Шаг 3. Формируем коэффициенты A/B для вертикального эталона.
    a_v = (
        cfg.calibration.a_scale
        * mu
        * z
        * cfg.operating.std_p_pa
        * t_k
        * psi_v
        / (math.pi * k_m2 * cfg.reservoir.h_m * cfg.operating.std_t_k)
    )
    b_v = (
        cfg.calibration.b_scale
        * cfg.reservoir.macro_l
        * z
        * cfg.operating.std_p_pa
        * t_k
        * (psi_v**1.1)
        / (math.pi * math.pi * k_m2 * cfg.reservoir.h_m * cfg.reservoir.h_m * cfg.operating.std_t_k)
    )

    # Шаг 4. Решаем уравнение притока и возвращаем стандартный дебит за сутки.
    q_std_m3s = _quadratic_positive_root(p_res_pa * p_res_pa - p_wf_pa * p_wf_pa, a_v, b_v)
    return q_std_m3s * 86400.0
