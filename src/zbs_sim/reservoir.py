from __future__ import annotations

import math
from typing import Dict, Tuple

# Константы пересчета единиц и физические константы (SI).
MPA_TO_PA = 1_000_000.0
MD_TO_M2 = 9.869233e-16
KELVIN_OFFSET = 273.15
R_UNIVERSAL = 8.314462618
M_AIR = 0.0289652
PSIA_TO_PA = 6894.757293168

# Критические параметры компонентов (минимальный набор для текущего состава).
# Значения заданы в SI.
# Источник по методике использования в смеси: Инструкция (1980), гл. II.3,
# формулы (II.12)-(II.13): псевдокритические параметры как сумма xi*Pкр,i и xi*Tкр,i.
COMPONENT_CRITICAL_PROPS: Dict[str, Tuple[float, float]] = {
    "CH4": (190.56, 4.5992e6),
    "C2H6": (305.32, 4.8720e6),
}


def mpa_to_pa(value_mpa: float) -> float:
    """Переводит давление из МПа в Па для внутренних расчетов в SI."""
    return value_mpa * MPA_TO_PA


def c_to_k(value_c: float) -> float:
    """Переводит температуру из C в K для PVT-корреляций."""
    return value_c + KELVIN_OFFSET


def md_to_m2(value_md: float) -> float:
    """Переводит проницаемость из мД в м²."""
    return value_md * MD_TO_M2


def _build_composition(methane_mol_frac: float | None, ethane_mol_frac: float | None) -> Dict[str, float] | None:
    """
    Собирает молярный состав из доступных полей конфигурации.

    Роль в проекте:
    - Позволяет использовать составной расчет псевдокритических параметров,
      если заданы доли CH4/C2H6.
    """
    # Шаг 1. Нормализуем отсутствующие значения к нулю.
    ch4 = max(methane_mol_frac or 0.0, 0.0)
    c2h6 = max(ethane_mol_frac or 0.0, 0.0)
    total = ch4 + c2h6

    # Шаг 2. Если состав не задан, возвращаем None и далее используем Sutton.
    if total <= 0.0:
        return None

    # Шаг 3. Нормируем известные компоненты до единицы.
    return {"CH4": ch4 / total, "C2H6": c2h6 / total}


def pseudo_critical_sutton_pa_k(gamma_g: float) -> Tuple[float, float]:
    """
    Возвращает псевдокритические параметры газа по Sutton.

    Почему эта зависимость:
    - Быстрый и устойчивый инженерный baseline, когда известна только gamma_g.

    Ссылки:
    - Sutton, R.P. (1985), ppc/tpc from gas specific gravity.
    """
    # Шаг 1. Расчет псевдокритической температуры (R) по gamma_g.
    t_pc_r = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g**2
    # Шаг 2. Расчет псевдокритического давления (psia) по gamma_g.
    p_pc_psia = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g**2
    # Шаг 3. Перевод в SI.
    t_pc_k = t_pc_r * (5.0 / 9.0)
    p_pc_pa = p_pc_psia * PSIA_TO_PA
    return p_pc_pa, t_pc_k


def pseudo_critical_composition_pa_k(composition_mol_frac: Dict[str, float] | None) -> Tuple[float, float] | None:
    """
    Возвращает псевдокритические параметры по составу (правило Кея).

    Почему эта зависимость:
    - При известном составе физически предпочтительнее SG-корреляций уровня Sutton.
    - Прямо соответствует подходу из пособия:
      Инструкция (1980), гл. II.3, формулы (II.12)-(II.13).
    """
    # Шаг 1. Проверяем наличие состава.
    if not composition_mol_frac:
        return None

    # Шаг 2. Оставляем только компоненты, для которых есть критические параметры.
    filtered: Dict[str, float] = {}
    for component, value in composition_mol_frac.items():
        if component in COMPONENT_CRITICAL_PROPS and value > 0.0:
            filtered[component] = value

    if not filtered:
        return None

    # Шаг 3. Нормируем доли и считаем Pпкр/Tпкр как сумму xi*Pкр,i и xi*Tкр,i.
    total = sum(filtered.values())
    t_pc_k = 0.0
    p_pc_pa = 0.0
    for component, value in filtered.items():
        xi = value / total
        t_crit_k, p_crit_pa = COMPONENT_CRITICAL_PROPS[component]
        t_pc_k += xi * t_crit_k
        p_pc_pa += xi * p_crit_pa
    return p_pc_pa, t_pc_k


def pseudo_critical_pa_k(
    gamma_g: float,
    methane_mol_frac: float | None = None,
    ethane_mol_frac: float | None = None,
) -> Tuple[float, float]:
    """
    Возвращает псевдокритические параметры газа с приоритетом состава.

    Логика выбора:
    - Если доступен состав CH4/C2H6 -> используем составной метод (правило Кея, Инструкция 1980).
    - Если состава нет -> fallback на Sutton.
    """
    # Шаг 1. Пытаемся посчитать псевдокритические параметры по составу.
    composition = _build_composition(methane_mol_frac, ethane_mol_frac)
    pseudo_from_composition = pseudo_critical_composition_pa_k(composition)
    if pseudo_from_composition is not None:
        return pseudo_from_composition

    # Шаг 2. Если состав недоступен, используем Sutton.
    return pseudo_critical_sutton_pa_k(gamma_g)


def z_factor_papay(
    pressure_pa: float,
    temperature_k: float,
    gamma_g: float,
    methane_mol_frac: float | None = None,
    ethane_mol_frac: float | None = None,
) -> float:
    """
    Оценивает коэффициент сверхсжимаемости Z по корреляции Papay.

    Роль в проекте:
    - Сохранен как альтернативный/контрольный метод.
    """
    # Шаг 1. Получаем псевдокритические параметры выбранным способом.
    p_pc_pa, t_pc_k = pseudo_critical_pa_k(
        gamma_g=gamma_g,
        methane_mol_frac=methane_mol_frac,
        ethane_mol_frac=ethane_mol_frac,
    )

    # Шаг 2. Считаем приведенные параметры и ограничиваем снизу для численной устойчивости.
    pr = max(pressure_pa / p_pc_pa, 0.01)
    tr = max(temperature_k / t_pc_k, 1.01)

    # Шаг 3. Рассчитываем Z по формуле Papay.
    z = 1.0 - (3.53 * pr) / (10.0 ** (0.9813 * tr)) + (0.274 * pr * pr) / (10.0 ** (0.8157 * tr))
    return max(z, 0.2)


def _dak_z_from_rhor(rho_r: float, t_pr: float) -> float:
    """
    Вычисляет Z по Dranchuk-Abou-Kassem для заданной приведенной плотности.

    Ссылки:
    - Dranchuk, Abou-Kassem (1975), Hall-Yarborough type explicit form in rho_r.
    """
    a1 = 0.3265
    a2 = -1.0700
    a3 = -0.5339
    a4 = 0.01569
    a5 = -0.05165
    a6 = 0.5475
    a7 = -0.7361
    a8 = 0.1844
    a9 = 0.1056
    a10 = 0.6134
    a11 = 0.7210

    # Шаг 1. Линейные/квадратичные члены по rho_r.
    term_1 = (a1 + a2 / t_pr + a3 / (t_pr**3) + a4 / (t_pr**4) + a5 / (t_pr**5)) * rho_r
    term_2 = (a6 + a7 / t_pr + a8 / (t_pr**2)) * (rho_r**2)

    # Шаг 2. Высокоплотностный член.
    term_3 = -a9 * (a7 / t_pr + a8 / (t_pr**2)) * (rho_r**5)

    # Шаг 3. Экспоненциальный член.
    term_4 = a10 * (1.0 + a11 * (rho_r**2)) * ((rho_r**2) / (t_pr**3)) * math.exp(-a11 * (rho_r**2))

    return 1.0 + term_1 + term_2 + term_3 + term_4


def _dak_reduced_density(p_pr: float, t_pr: float, max_iter: int = 40) -> float:
    """
    Решает неявное уравнение DAK относительно приведенной плотности rho_r.

    Роль в проекте:
    - Позволяет получить более точный Z(P,T), чем Papay, без графиков.
    """
    # Шаг 1. Стартовая оценка rho_r из допущения Z≈1.
    rho_r = max(0.27 * p_pr / max(t_pr, 1e-9), 1e-6)

    # Шаг 2. Итерации Ньютона с численной производной.
    for _ in range(max_iter):
        z = max(_dak_z_from_rhor(rho_r, t_pr), 0.05)
        f = rho_r - 0.27 * p_pr / (z * t_pr)

        eps = max(1e-6 * rho_r, 1e-7)
        z_eps = max(_dak_z_from_rhor(rho_r + eps, t_pr), 0.05)
        f_eps = (rho_r + eps) - 0.27 * p_pr / (z_eps * t_pr)
        derivative = (f_eps - f) / eps

        if abs(derivative) < 1e-12:
            break

        candidate = rho_r - f / derivative
        if not math.isfinite(candidate) or candidate <= 0.0:
            candidate = 0.5 * rho_r

        if abs(candidate - rho_r) / max(rho_r, 1e-9) < 1e-7:
            rho_r = candidate
            break
        rho_r = candidate

    return max(rho_r, 1e-6)


def z_factor_dak(
    pressure_pa: float,
    temperature_k: float,
    gamma_g: float,
    methane_mol_frac: float | None = None,
    ethane_mol_frac: float | None = None,
) -> float:
    """
    Оценивает коэффициент сверхсжимаемости Z по Dranchuk-Abou-Kassem.

    Почему выбрано как дефолт:
    - Более детальная аналитическая аппроксимация по сравнению с Papay.
    - Не требует ручного чтения графиков из старых методик.
    - Удобна для автоматического многовариантного анализа.

    Сопоставление с пособием:
    - В Инструкции (1980), гл. II.5, предложены графические и аналитические методы.
    - Для кода и серии сценариев DAK обычно практичнее и воспроизводимее.
    """
    # Шаг 1. Получаем псевдокритические параметры выбранным способом.
    p_pc_pa, t_pc_k = pseudo_critical_pa_k(
        gamma_g=gamma_g,
        methane_mol_frac=methane_mol_frac,
        ethane_mol_frac=ethane_mol_frac,
    )

    # Шаг 2. Считаем приведенные параметры.
    p_pr = max(pressure_pa / p_pc_pa, 1e-6)
    t_pr = max(temperature_k / t_pc_k, 1.01)

    # Шаг 3. Решаем неявное уравнение по rho_r и получаем Z.
    rho_r = _dak_reduced_density(p_pr, t_pr)
    z = _dak_z_from_rhor(rho_r, t_pr)
    return max(min(z, 3.0), 0.2)


def gas_density_kgm3(pressure_pa: float, temperature_k: float, z_factor: float, gamma_g: float) -> float:
    """
    Вычисляет плотность газа по уравнению состояния реального газа.

    Эта функция нужна в гидравлике ствола для Re, потерь трения и гравитационной составляющей.
    """
    # Шаг 1. Переводим относительную плотность в молярную массу газа.
    molar_mass = gamma_g * M_AIR
    # Шаг 2. Считаем плотность через pM/(ZRT).
    return pressure_pa * molar_mass / (z_factor * R_UNIVERSAL * temperature_k)


def gas_viscosity_pa_s_lee(
    pressure_pa: float,
    temperature_k: float,
    z_factor: float,
    gamma_g: float,
) -> float:
    """
    Оценивает вязкость газа по Lee-Gonzalez-Eakin.

    Почему эта зависимость:
    - Стандартная промышленная корреляция для инженерных расчетов сухого газа.
    - Использует P, T и плотность газа, что соответствует вашему ТЗ по μ(P,T).

    Ссылки:
    - Lee, Gonzalez, Eakin (1966), gas viscosity correlation.
    - «2.3 АН 2021», стр. 50-51 (требование учитывать μ(P,T)).
    """
    # Шаг 1. Определяем молекулярную массу и текущую плотность газа.
    molar_mass_g = gamma_g * 28.9652
    rho_kgm3 = gas_density_kgm3(pressure_pa, temperature_k, z_factor, gamma_g)
    rho_gcm3 = rho_kgm3 / 1000.0

    # Шаг 2. Вычисляем эмпирические коэффициенты корреляции Lee.
    k = ((9.379 + 0.01607 * molar_mass_g) * temperature_k**1.5) / (
        209.2 + 19.26 * molar_mass_g + temperature_k
    )
    x = 3.448 + (986.4 / temperature_k) + 0.01009 * molar_mass_g
    y = 2.447 - 0.2224 * x

    # Шаг 3. Получаем вязкость (сP) и переводим в Па*с.
    mu_cp = 1e-4 * k * math.exp(x * (rho_gcm3**y))
    return mu_cp * 1e-3


def gas_properties(
    pressure_pa: float,
    temperature_k: float,
    gamma_g: float,
    methane_mol_frac: float | None = None,
    ethane_mol_frac: float | None = None,
    z_method: str = "dak",
) -> Tuple[float, float, float]:
    """
    Комплексный вызов блока PVT.

    Возвращает:
    - Z(P,T),
    - μ(P,T),
    - ρ(P,T).

    Роль в проекте:
    - Используется и в притоке (коэффициенты A/B), и в гидравлике ствола.
    """
    # Шаг 1. Считаем коэффициент сверхсжимаемости выбранной корреляцией.
    method = z_method.lower().strip()
    if method == "papay":
        z = z_factor_papay(
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            gamma_g=gamma_g,
            methane_mol_frac=methane_mol_frac,
            ethane_mol_frac=ethane_mol_frac,
        )
    else:
        z = z_factor_dak(
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            gamma_g=gamma_g,
            methane_mol_frac=methane_mol_frac,
            ethane_mol_frac=ethane_mol_frac,
        )

    # Шаг 2. Считаем вязкость.
    mu = gas_viscosity_pa_s_lee(pressure_pa, temperature_k, z, gamma_g)

    # Шаг 3. Считаем плотность.
    rho = gas_density_kgm3(pressure_pa, temperature_k, z, gamma_g)
    return z, mu, rho
