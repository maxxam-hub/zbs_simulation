from __future__ import annotations

import math
from typing import List

from .models import ReservoirParams, Scenario, WellboreParams


def curvature_class(radius_m: float) -> str:
    """
    Классифицирует радиус кривизны для отчетности сценариев.

    Роль в проекте:
    - Используется в выходной таблице для быстрой группировки кейсов.
    """
    # Шаг 1. Определяем класс "малый радиус" по технологическому диапазону.
    if 4.0 <= radius_m < 12.0:
        return "малый"
    # Шаг 2. Определяем класс "средний радиус".
    if 12.0 <= radius_m <= 150.0:
        return "средний"
    # Шаг 3. Все остальные случаи помечаем как "вне диапазона".
    return "вне диапазона"


def generate_shoe_depths(lateral_length_m: float, step_m: float) -> List[float]:
    """
    Формирует набор глубин башмака НКТ вдоль горизонтального участка.

    Роль в проекте:
    - Обеспечивает перебор сценариев по глубине спуска НКТ.
    """
    # Шаг 1. Обрабатываем вырожденный случай отсутствия горизонтального участка.
    if lateral_length_m <= 0:
        return [0.0]
    # Шаг 2. Добавляем начало ГС (башмак на пятке).
    values = [0.0]
    current = step_m
    # Шаг 3. Добавляем промежуточные положения башмака с заданным шагом.
    while current < lateral_length_m:
        values.append(round(current, 6))
        current += step_m
    # Шаг 4. Добавляем крайнее положение (башмак у забоя).
    values.append(round(lateral_length_m, 6))
    # Шаг 5. Удаляем повторы, сохраняя исходный порядок.
    unique: List[float] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def toe_elevation_gain_m(scenario: Scenario, wellbore: WellboreParams) -> float:
    """
    Рассчитывает набор/сброс высоты от пятки к носку ГС.

    Роль в проекте:
    - Нужен для гравитационной части потерь давления в стволе.
    """
    # Шаг 1. Берем высоту носка ствола относительно пятки на конечной длине.
    return elevation_at_distance_m(scenario, wellbore, scenario.lateral_length_m)


def _scaled_stepped_lengths(scenario: Scenario, wellbore: WellboreParams) -> List[float]:
    """
    Возвращает длины ступеней, масштабированные под фактическую длину ГС сценария.

    Почему так:
    - пользователь задает шаблон ступеней в конфиге;
    - при переборе Lг шаблон масштабируется, сохраняя относительную структуру.
    """
    lateral = max(scenario.lateral_length_m, 0.0)
    if lateral <= 0.0:
        return []
    base = [float(value) for value in wellbore.stepped_segment_lengths_m if float(value) > 0.0]
    if not base:
        return [lateral]
    total = sum(base)
    if total <= 0.0:
        return [lateral]
    scale = lateral / total
    return [value * scale for value in base]


def elevation_at_distance_m(scenario: Scenario, wellbore: WellboreParams, distance_m: float) -> float:
    """
    Возвращает высоту ствола относительно пятки в точке distance_m вдоль ГС.

    Профили:
    - flat: постоянная высота;
    - ascending: линейный набор высоты;
    - descending: линейное снижение высоты;
    - stepped: горизонтальные ступени с заданными перепадами между ними.
    """
    lateral = max(scenario.lateral_length_m, 0.0)
    if lateral <= 0.0:
        return 0.0
    x = min(max(distance_m, 0.0), lateral)

    # Шаг 1. Плоский профиль.
    if scenario.profile == "flat":
        return 0.0

    # Шаг 2. Линейные профили по углу.
    if scenario.profile in ("ascending", "descending"):
        sign = 1.0 if scenario.profile == "ascending" else -1.0
        angle_rad = math.radians(abs(wellbore.ascending_angle_deg))
        return sign * x * math.tan(angle_rad)

    # Шаг 3. Ступенчатый профиль.
    if scenario.profile == "stepped":
        lengths = _scaled_stepped_lengths(scenario, wellbore)
        if not lengths:
            return 0.0
        heights = list(float(value) for value in wellbore.stepped_step_heights_m)
        expected = max(len(lengths) - 1, 0)
        if len(heights) < expected:
            heights.extend([0.0] * (expected - len(heights)))
        elif len(heights) > expected:
            heights = heights[:expected]

        current_x = 0.0
        elevation = 0.0
        for index, segment_len in enumerate(lengths):
            next_x = current_x + segment_len
            if x < next_x - 1e-9:
                return elevation
            if index < len(heights):
                elevation += heights[index]
            current_x = next_x
        return elevation

    raise ValueError(f"Неизвестный профиль '{scenario.profile}'.")


def resistance_term(
    reservoir: ReservoirParams,
    wellbore: WellboreParams,
    scenario: Scenario,
) -> float:
    """
    Оценивает интегральный геометрический множитель сопротивления притоку (psi) для ГС.

    Роль в проекте:
    - Входит в коэффициенты A и B двучленного уравнения притока.
    - Учитывает длину ГС, анизотропию, асимметрию в пласте и радиус кривизны.

    Источники:
    - Алиев (2015), стр. 14, 18-19, 65: влияние геометрии ГС и положения ствола в пласте.
    """
    # Шаг 1. Базовый логарифмический член по радиусу дренирования и радиусу ствола.
    drainage_ratio = max(wellbore.drainage_radius_m / wellbore.wellbore_radius_m, 1.0001)
    base = math.log(drainage_ratio)

    # Шаг 2. Снижение сопротивления с ростом длины ГС.
    lateral_gain = 1.0 + math.log1p(scenario.lateral_length_m / max(reservoir.h_m, 0.1))
    # Шаг 3. Учет анизотропии через ae = sqrt(kv/kh): меньший ae -> выше сопротивление.
    if not (0.0 < scenario.anisotropy_ae <= 1.0):
        raise ValueError("Параметр anisotropy_ae должен быть в диапазоне 0 < ae <= 1.")
    anisotropy_mult = 1.0 / max(scenario.anisotropy_ae, 0.05)

    # Шаг 4. Учет асимметрии положения ствола в разрезе пласта (h1 относительно середины пласта).
    asymmetry = abs(wellbore.entry_from_roof_m - 0.5 * reservoir.h_m) / max(0.5 * reservoir.h_m, 0.1)
    asymmetry_mult = 1.0 + 0.6 * asymmetry

    # Шаг 5. Учет дополнительного ущерба проводимости при малом радиусе кривизны.
    curvature_mult = 1.0 + 0.35 * min(1.0, 12.0 / max(scenario.curvature_radius_m, 1.0))

    # Шаг 6. Сборка итогового множителя с учетом skin.
    psi = base * anisotropy_mult * asymmetry_mult * curvature_mult / lateral_gain + reservoir.skin
    # Шаг 7. Ограничение снизу для устойчивости решения.
    return max(psi, 0.2)


def vertical_resistance_term(reservoir: ReservoirParams, wellbore: WellboreParams) -> float:
    """
    Оценивает множитель сопротивления для опорной вертикальной скважины.

    Роль в проекте:
    - Нужен как базовый сценарий для расчета прироста/падения дебита ГС.
    """
    # Шаг 1. Базовый логарифмический член по классической вертикальной геометрии.
    drainage_ratio = max(wellbore.drainage_radius_m / wellbore.wellbore_radius_m, 1.0001)
    # Шаг 2. Добавляем skin и ограничиваем снизу.
    return max(math.log(drainage_ratio) + reservoir.skin, 0.2)
