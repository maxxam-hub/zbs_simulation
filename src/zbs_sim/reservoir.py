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

# Критические параметры компонентов (Tc [K], Pc [Pa]).
# Набор расширен, чтобы можно было задавать более детальный состав газа.
COMPONENT_CRITICAL_PROPS: Dict[str, Tuple[float, float]] = {
    "CH4": (190.56, 4.5992e6),
    "C2H6": (305.32, 4.8720e6),
    "C3H8": (369.83, 4.2480e6),
    "IC4H10": (407.85, 3.6480e6),
    "NC4H10": (425.12, 3.7960e6),
    "IC5H12": (460.35, 3.3700e6),
    "NC5H12": (469.70, 3.3700e6),
    "NC6H14": (507.60, 3.0250e6),
    "N2": (126.20, 3.3958e6),
    "CO2": (304.20, 7.3773e6),
    "H2S": (373.20, 8.9600e6),
    "HE": (5.20, 0.2270e6),
    "AR": (150.86, 4.8630e6),
}

# Допустимые названия методик задаются явно и без алиасов:
# ppc_method:
# - composition_linear
# - composition_precise
# - specific_gravity_sutton
# z_method:
# - dranchuk_abou_kassem
# - aliev_zotov_two_parameter
# - papay


def mpa_to_pa(value_mpa: float) -> float:
    """Переводит давление из МПа в Па для внутренних расчетов в SI."""
    return value_mpa * MPA_TO_PA


def c_to_k(value_c: float) -> float:
    """Переводит температуру из C в K для PVT-корреляций."""
    return value_c + KELVIN_OFFSET


def md_to_m2(value_md: float) -> float:
    """Переводит проницаемость из мД в м²."""
    return value_md * MD_TO_M2


def _build_composition(
    composition_mol_frac: Dict[str, float] | None = None,
) -> Dict[str, float] | None:
    """
    Собирает молярный состав из доступных входов и нормирует его.

    Логика:
    - если передан расширенный словарь состава, используем его;
    - иначе возвращаем None (дальше сработает fallback на методы по gamma_g).
    """
    # Шаг 1. Приоритетно читаем расширенный состав, если он передан.
    if composition_mol_frac:
        collected: Dict[str, float] = {}
        for raw_name, value in composition_mol_frac.items():
            canonical = raw_name.strip().upper()
            if canonical not in COMPONENT_CRITICAL_PROPS:
                continue
            if value <= 0.0:
                continue
            collected[canonical] = collected.get(canonical, 0.0) + float(value)

        total = sum(collected.values())
        if total > 0.0:
            return {name: val / total for name, val in collected.items()}

    # Шаг 2. Если состав не задан, возвращаем None.
    return None


def pseudo_critical_sutton_pa_k(gamma_g: float) -> Tuple[float, float]:
    """
    Возвращает псевдокритические параметры газа по Sutton.

    Роль:
    - fallback-метод, когда компонентный состав неизвестен.
    """
    # Шаг 1. Расчет псевдокритической температуры (R) по gamma_g.
    t_pc_r = 169.2 + 349.5 * gamma_g - 74.0 * gamma_g**2
    # Шаг 2. Расчет псевдокритического давления (psia) по gamma_g.
    p_pc_psia = 756.8 - 131.0 * gamma_g - 3.6 * gamma_g**2
    # Шаг 3. Перевод в SI.
    t_pc_k = t_pc_r * (5.0 / 9.0)
    p_pc_pa = p_pc_psia * PSIA_TO_PA
    return p_pc_pa, t_pc_k


def pseudo_critical_by_composition_linear_pa_k(
    composition_mol_frac: Dict[str, float] | None,
) -> Tuple[float, float] | None:
    """
    Псевдокритические параметры по формулам (II.12)-(II.13), Инструкция 1980.

    Формулы:
    - pпкр = Σ xi * pкр,i
    - Tпкр = Σ xi * Tкр,i
    """
    if not composition_mol_frac:
        return None

    # Шаг 1. Суммируем вклад каждого компонента.
    t_pc_k = 0.0
    p_pc_pa = 0.0
    for component, xi in composition_mol_frac.items():
        t_crit_k, p_crit_pa = COMPONENT_CRITICAL_PROPS[component]
        t_pc_k += xi * t_crit_k
        p_pc_pa += xi * p_crit_pa

    if t_pc_k <= 0.0 or p_pc_pa <= 0.0:
        return None
    return p_pc_pa, t_pc_k


def pseudo_critical_by_composition_precise_pa_k(
    composition_mol_frac: Dict[str, float] | None,
) -> Tuple[float, float] | None:
    """
    Псевдокритические параметры по формулам (II.14), Инструкция 1980.

    Формулы:
    - pпкр = K^2 / J^2
    - Tпкр = K^2 / J
    - K = Σ xi * (Tкр,i / sqrt(pкр,i))
    - J = 1/8 Σi Σj xi xj [ (Tкр,i/pкр,i)^(1/3) + (Tкр,j/pкр,j)^(1/3) ]^3
    """
    if not composition_mol_frac:
        return None

    # Шаг 1. Считаем коэффициент K.
    k_value = 0.0
    for component, xi in composition_mol_frac.items():
        t_crit_k, p_crit_pa = COMPONENT_CRITICAL_PROPS[component]
        k_value += xi * (t_crit_k / math.sqrt(p_crit_pa))

    # Шаг 2. Считаем коэффициент J двойной суммой.
    j_value = 0.0
    items = list(composition_mol_frac.items())
    for component_i, xi in items:
        t_crit_i, p_crit_i = COMPONENT_CRITICAL_PROPS[component_i]
        theta_i = (t_crit_i / p_crit_i) ** (1.0 / 3.0)
        for component_j, xj in items:
            t_crit_j, p_crit_j = COMPONENT_CRITICAL_PROPS[component_j]
            theta_j = (t_crit_j / p_crit_j) ** (1.0 / 3.0)
            j_value += xi * xj * ((theta_i + theta_j) ** 3)
    j_value *= 0.125

    if j_value <= 0.0:
        return None

    # Шаг 3. Получаем псевдокритические параметры.
    t_pc_k = (k_value**2) / j_value
    p_pc_pa = t_pc_k / j_value
    if t_pc_k <= 0.0 or p_pc_pa <= 0.0:
        return None
    return p_pc_pa, t_pc_k


def pseudo_critical_pa_k(
    gamma_g: float,
    composition_mol_frac: Dict[str, float] | None = None,
    method: str = "composition_linear",
) -> Tuple[float, float]:
    """
    Возвращает псевдокритические параметры газа выбранным методом.

    Доступные методы:
    - `composition_linear` (по составу, аддитивно);
    - `composition_precise` (по составу, повышенная точность);
    - `specific_gravity_sutton` (по относительной плотности).
    """
    # Шаг 1. Нормируем признак метода.
    method_norm = method.strip().lower()

    # Шаг 2. Для составных методов собираем компонентный состав.
    composition = _build_composition(
        composition_mol_frac=composition_mol_frac,
    )

    # Шаг 3. Вычисляем по выбранной схеме.
    if method_norm == "composition_precise":
        ppc_tpc = pseudo_critical_by_composition_precise_pa_k(composition)
        if ppc_tpc is not None:
            return ppc_tpc
        return pseudo_critical_sutton_pa_k(gamma_g)

    if method_norm == "specific_gravity_sutton":
        return pseudo_critical_sutton_pa_k(gamma_g)

    if method_norm != "composition_linear":
        raise ValueError(
            f"Неизвестный ppc_method='{method}'. Допустимо: "
            "composition_linear | composition_precise | specific_gravity_sutton."
        )

    ppc_tpc = pseudo_critical_by_composition_linear_pa_k(composition)
    if ppc_tpc is not None:
        return ppc_tpc
    return pseudo_critical_sutton_pa_k(gamma_g)


def z_factor_papay(
    pressure_pa: float,
    temperature_k: float,
    gamma_g: float,
    composition_mol_frac: Dict[str, float] | None = None,
    ppc_method: str = "composition_linear",
) -> float:
    """Оценивает коэффициент сверхсжимаемости Z по корреляции Papay."""
    # Шаг 1. Получаем псевдокритические параметры выбранным способом.
    p_pc_pa, t_pc_k = pseudo_critical_pa_k(
        gamma_g=gamma_g,
        composition_mol_frac=composition_mol_frac,
        method=ppc_method,
    )

    # Шаг 2. Считаем приведенные параметры.
    pr = max(pressure_pa / p_pc_pa, 0.01)
    tr = max(temperature_k / t_pc_k, 1.01)

    # Шаг 3. Рассчитываем Z по формуле Papay.
    z = 1.0 - (3.53 * pr) / (10.0 ** (0.9813 * tr)) + (0.274 * pr * pr) / (10.0 ** (0.8157 * tr))
    return max(z, 0.2)


def z_factor_aliev_zotov_two_parameter(
    pressure_pa: float,
    temperature_k: float,
    gamma_g: float,
    composition_mol_frac: Dict[str, float] | None = None,
    ppc_method: str = "composition_linear",
    max_iter: int = 80,
) -> float:
    """
    Оценивает Z по формуле (II.21), Инструкция 1980.

    Используется неявное уравнение:
    - z = 1/(1-h) - (a*2/b*) * h/(1+h),
    где h = p*b*/z,
      b* = 0.08677 * Tпкр / (pпкр*T),
      a*2 = 0.42787 * Tпкр^2.5 / (pпкр*T^2.5).
    """
    # Шаг 1. Получаем псевдокритические параметры газа.
    p_pc_pa, t_pc_k = pseudo_critical_pa_k(
        gamma_g=gamma_g,
        composition_mol_frac=composition_mol_frac,
        method=ppc_method,
    )

    # Шаг 2. Рассчитываем коэффициенты уравнения II.21.
    b_star = 0.08677 * t_pc_k / (p_pc_pa * temperature_k)
    a_star_2 = 0.42787 * (t_pc_k**2.5) / (p_pc_pa * (temperature_k**2.5))
    ratio = a_star_2 / max(b_star, 1e-18)

    # Шаг 3. Итерационно решаем неявное уравнение относительно z.
    z = z_factor_papay(
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
        gamma_g=gamma_g,
        composition_mol_frac=composition_mol_frac,
        ppc_method=ppc_method,
    )

    for _ in range(max_iter):
        h = pressure_pa * b_star / max(z, 1e-9)
        # Ограничиваем h, чтобы не попадать в особую точку 1/(1-h).
        h = min(max(h, -0.99), 0.99)

        z_new = 1.0 / (1.0 - h) - ratio * (h / (1.0 + h))
        if not math.isfinite(z_new):
            z_new = z
        z_new = max(min(z_new, 3.0), 0.2)

        if abs(z_new - z) / max(z, 1e-9) < 1e-7:
            z = z_new
            break
        z = 0.5 * z + 0.5 * z_new

    return max(min(z, 3.0), 0.2)


def _dak_z_from_rhor(rho_r: float, t_pr: float) -> float:
    """Вычисляет Z по Dranchuk-Abou-Kassem для заданной приведенной плотности."""
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
    """Решает неявное уравнение DAK относительно приведенной плотности rho_r."""
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
    composition_mol_frac: Dict[str, float] | None = None,
    ppc_method: str = "composition_linear",
) -> float:
    """Оценивает коэффициент сверхсжимаемости Z по Dranchuk-Abou-Kassem."""
    # Шаг 1. Получаем псевдокритические параметры выбранным способом.
    p_pc_pa, t_pc_k = pseudo_critical_pa_k(
        gamma_g=gamma_g,
        composition_mol_frac=composition_mol_frac,
        method=ppc_method,
    )

    # Шаг 2. Считаем приведенные параметры.
    p_pr = max(pressure_pa / p_pc_pa, 1e-6)
    t_pr = max(temperature_k / t_pc_k, 1.01)

    # Шаг 3. Решаем неявное уравнение по rho_r и получаем Z.
    rho_r = _dak_reduced_density(p_pr, t_pr)
    z = _dak_z_from_rhor(rho_r, t_pr)
    return max(min(z, 3.0), 0.2)


def gas_density_kgm3(pressure_pa: float, temperature_k: float, z_factor: float, gamma_g: float) -> float:
    """Вычисляет плотность газа по уравнению состояния реального газа."""
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
    """Оценивает вязкость газа по Lee-Gonzalez-Eakin."""
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
    composition_mol_frac: Dict[str, float] | None = None,
    ppc_method: str = "composition_linear",
    z_method: str = "dranchuk_abou_kassem",
) -> Tuple[float, float, float]:
    """
    Комплексный вызов блока PVT.

    Возвращает:
    - Z(P,T),
    - μ(P,T),
    - ρ(P,T).
    """
    # Шаг 1. Считаем коэффициент сверхсжимаемости выбранной корреляцией.
    method = z_method.lower().strip()
    if method == "papay":
        z = z_factor_papay(
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            gamma_g=gamma_g,
            composition_mol_frac=composition_mol_frac,
            ppc_method=ppc_method,
        )
    elif method == "aliev_zotov_two_parameter":
        z = z_factor_aliev_zotov_two_parameter(
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            gamma_g=gamma_g,
            composition_mol_frac=composition_mol_frac,
            ppc_method=ppc_method,
        )
    elif method == "dranchuk_abou_kassem":
        z = z_factor_dak(
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            gamma_g=gamma_g,
            composition_mol_frac=composition_mol_frac,
            ppc_method=ppc_method,
        )
    else:
        raise ValueError(
            f"Неизвестный z_method='{z_method}'. Допустимо: "
            "dranchuk_abou_kassem | aliev_zotov_two_parameter | papay."
        )

    # Шаг 2. Считаем вязкость.
    mu = gas_viscosity_pa_s_lee(pressure_pa, temperature_k, z, gamma_g)

    # Шаг 3. Считаем плотность.
    rho = gas_density_kgm3(pressure_pa, temperature_k, z, gamma_g)
    return z, mu, rho
