from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .geometry import toe_elevation_gain_m
from .models import BaseConfig, Scenario
from .reservoir import c_to_k, gas_properties, mpa_to_pa

GRAVITY = 9.80665


@dataclass(frozen=True)
class WellboreLosses:
    """Потери давления в стволе: отдельно в НКТ и в кольцевом пространстве."""

    tubing_drop_pa: float
    annulus_drop_pa: float

    @property
    def total_drop_pa(self) -> float:
        """Суммарные потери давления по всему горизонтальному участку."""
        return self.tubing_drop_pa + self.annulus_drop_pa


def _friction_factor(reynolds: float, rel_roughness: float) -> float:
    """
    Возвращает коэффициент трения Darcy.

    Почему эта зависимость:
    - Ламинарный режим: формула Пуазейля f=64/Re.
    - Турбулентный режим: явная аппроксимация Colebrook (Swamee-Jain).
    - Это базовый промышленный подход для гидравлики НКТ/затруба.
    """
    # Шаг 1. Защита от невалидного Re.
    if reynolds <= 0:
        return 0.0
    # Шаг 2. Ламинарный режим.
    if reynolds < 2320.0:
        return 64.0 / reynolds
    # Шаг 3. Турбулентный режим через явную формулу.
    return 0.25 / (math.log10(rel_roughness / 3.7 + 5.74 / (reynolds**0.9)) ** 2)


def _segment_drop_pa(
    q_std_m3s: float,
    length_m: float,
    hydraulic_diameter_m: float,
    flow_area_m2: float,
    delta_z_flow_m: float,
    pressure_ref_pa: float,
    cfg: BaseConfig,
) -> float:
    """
    Считает потери давления на одном сегменте (НКТ или затруб).

    Роль в проекте:
    - Физическое ядро шага "Completion Module": раздельная гидравлика по сегментам.

    Ссылки по методике:
    - Darcy-Weisbach + Colebrook для трения в трубах.
    - Алиев (2015), стр. 81-82: необходимость раздельного учета участков НКТ/затруба.
    """
    # Шаг 1. Быстрый выход для пустого/невалидного сегмента.
    if length_m <= 0.0 or hydraulic_diameter_m <= 0.0 or flow_area_m2 <= 0.0:
        return 0.0

    # Шаг 2. Получаем PVT-свойства газа на опорном давлении сегмента.
    t_k = c_to_k(cfg.reservoir.t_res_c)
    z, mu, rho = gas_properties(
        pressure_pa=pressure_ref_pa,
        temperature_k=t_k,
        gamma_g=cfg.reservoir.gamma_g,
        methane_mol_frac=cfg.reservoir.methane_mol_frac,
        ethane_mol_frac=cfg.reservoir.ethane_mol_frac,
        z_method=cfg.reservoir.z_method,
    )

    # Шаг 3. Переводим стандартный расход в пластовые условия и считаем скорость.
    q_in_situ_m3s = q_std_m3s * (cfg.operating.std_p_pa / pressure_ref_pa) * (z * t_k / cfg.operating.std_t_k)
    velocity = q_in_situ_m3s / flow_area_m2

    # Шаг 4. Считаем Re и коэффициент трения.
    reynolds = rho * abs(velocity) * hydraulic_diameter_m / max(mu, 1e-12)
    rel_rough = cfg.wellbore.roughness_m / hydraulic_diameter_m
    friction = _friction_factor(reynolds, rel_rough)

    # Шаг 5. Считаем потери на трение и гравитацию.
    dp_friction = friction * (length_m / hydraulic_diameter_m) * (rho * velocity * abs(velocity) / 2.0)
    dp_gravity = rho * GRAVITY * delta_z_flow_m
    # Шаг 6. Возвращаем суммарный перепад на сегменте.
    return dp_friction + dp_gravity


def compute_wellbore_losses(
    cfg: BaseConfig,
    scenario: Scenario,
    q_std_m3s: float,
    pressure_ref_pa: float | None = None,
) -> WellboreLosses:
    """
    Считает суммарные потери давления по горизонтальному стволу.

    Роль в проекте:
    - Реализует сегментацию "НКТ + кольцевое пространство" для заданной глубины башмака.
    """
    # Шаг 1. Выбираем опорное давление, если оно не передано явно.
    if pressure_ref_pa is None:
        pressure_ref_pa = mpa_to_pa(cfg.operating.p_wf_heel_mpa)

    # Шаг 2. Определяем длины сегментов НКТ и затруба.
    lateral = max(scenario.lateral_length_m, 0.0)
    shoe = min(max(scenario.tubing_shoe_from_heel_m, 0.0), lateral)
    tubing_len = shoe
    annulus_len = max(lateral - shoe, 0.0)

    # Шаг 3. Оцениваем высотный перепад (важно для восходящего профиля).
    toe_gain = toe_elevation_gain_m(scenario, cfg.wellbore)
    dz_total_flow = -toe_gain

    # Шаг 4. Распределяем гравитационную составляющую по сегментам пропорционально длине.
    dz_tubing = dz_total_flow * (tubing_len / lateral) if lateral > 0.0 else 0.0
    dz_annulus = dz_total_flow * (annulus_len / lateral) if lateral > 0.0 else 0.0

    # Шаг 5. Считаем площади течения для НКТ и затрубного пространства.
    tubing_area = math.pi * (cfg.wellbore.tubing_id_m**2) / 4.0
    annulus_area = math.pi * (cfg.wellbore.casing_id_m**2 - cfg.wellbore.tubing_od_m**2) / 4.0

    # Шаг 6. Считаем потери по каждому сегменту.
    dp_tubing = _segment_drop_pa(
        q_std_m3s=q_std_m3s,
        length_m=tubing_len,
        hydraulic_diameter_m=cfg.wellbore.tubing_id_m,
        flow_area_m2=tubing_area,
        delta_z_flow_m=dz_tubing,
        pressure_ref_pa=pressure_ref_pa,
        cfg=cfg,
    )
    dp_annulus = _segment_drop_pa(
        q_std_m3s=q_std_m3s,
        length_m=annulus_len,
        hydraulic_diameter_m=max(cfg.wellbore.casing_id_m - cfg.wellbore.tubing_od_m, 1e-4),
        flow_area_m2=annulus_area,
        delta_z_flow_m=dz_annulus,
        pressure_ref_pa=pressure_ref_pa,
        cfg=cfg,
    )

    # Шаг 7. Возвращаем структуру с отдельными и суммарными потерями.
    return WellboreLosses(tubing_drop_pa=dp_tubing, annulus_drop_pa=dp_annulus)


def pressure_profile_heel_to_toe(
    cfg: BaseConfig,
    scenario: Scenario,
    q_std_m3s: float,
    points: int = 120,
) -> Tuple[List[float], List[float]]:
    """
    Строит профиль давления от пятки к забою для выбранного сценария.

    Роль в проекте:
    - Основа графика распределения давления по стволу (обоснование глубины башмака НКТ).
    """
    # Шаг 1. Проверка вырожденного случая без ГС.
    lateral = max(scenario.lateral_length_m, 0.0)
    if lateral == 0.0:
        return [0.0], [cfg.operating.p_wf_heel_mpa]

    # Шаг 2. Получаем потери давления по сегментам.
    p_heel_pa = mpa_to_pa(cfg.operating.p_wf_heel_mpa)
    losses = compute_wellbore_losses(cfg, scenario, q_std_m3s, pressure_ref_pa=p_heel_pa)

    # Шаг 3. Разбиваем ствол на НКТ и затруб.
    shoe = min(max(scenario.tubing_shoe_from_heel_m, 0.0), lateral)
    tubing_len = shoe
    annulus_len = max(lateral - shoe, 0.0)

    # Шаг 4. Формируем сетку координат вдоль ствола.
    if points < 2:
        points = 2
    step = lateral / (points - 1)
    xs = [i * step for i in range(points)]
    ps_pa: List[float] = []

    # Шаг 5. На каждой точке рассчитываем давление кусочно-линейно по сегментам.
    p_shoe_pa = p_heel_pa + losses.tubing_drop_pa
    for x in xs:
        if tubing_len > 0.0 and x <= tubing_len:
            p = p_heel_pa + losses.tubing_drop_pa * (x / tubing_len)
        elif annulus_len > 0.0 and x > tubing_len:
            p = p_shoe_pa + losses.annulus_drop_pa * ((x - tubing_len) / annulus_len)
        else:
            p = p_heel_pa
        ps_pa.append(p)

    # Шаг 6. Переводим давление в МПа для отчетности и графиков.
    ps_mpa = [value / 1_000_000.0 for value in ps_pa]
    return xs, ps_mpa
