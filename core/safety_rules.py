"""
core/safety_rules.py  —  Team Alpha
Deterministic safety guardrails evaluated BEFORE any actuation command
is dispatched by the Supervisor.

Design principle
----------------
These are hard-coded rules with zero ML involvement.  They run after
the LLM produces an action payload and act as the final veto layer.
If ANY rule fires, the action is blocked and a SafetyViolation is raised.

Usage
-----
    from core.safety_rules import enforce_safety, SafetyViolation

    try:
        enforce_safety(action_payload)
    except SafetyViolation as e:
        # Log, alert, return safe refusal to user
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────────
# Domain Constants  (agronomic safety limits)
# ─────────────────────────────────────────────────────────────

class NutrientLimits:
    """Maximum single-application doses in kg/hectare."""
    NITROGEN_MAX:   float = 200.0
    PHOSPHORUS_MAX: float = 100.0
    POTASSIUM_MAX:  float = 150.0


class IrrigationLimits:
    """Water volume limits in litres/m²."""
    WATER_MAX_PER_APPLICATION: float = 50.0


class PesticideLimits:
    """Pesticide concentration limits in mg/L."""
    CONCENTRATION_MAX: float = 500.0


class TemperatureLimits:
    """Operating temperature window in °C for actuation."""
    ACTUATION_TEMP_MIN: float = 5.0
    ACTUATION_TEMP_MAX: float = 45.0


class HumidityLimits:
    """Relative humidity % bounds for spray actuation."""
    SPRAY_HUMIDITY_MIN: float = 20.0  # Below this: spray drifts, wasted
    SPRAY_HUMIDITY_MAX: float = 95.0  # Above this: fungal risk with wet application


# ─────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────

@dataclass
class SafetyViolation(Exception):
    """
    Raised when an action payload violates one or more safety rules.

    Attributes
    ----------
    violations : list of str
        Human-readable descriptions of each rule that fired.
    action_payload : dict
        The original payload that triggered the violation (for logging).
    """
    violations:     List[str]
    action_payload: Dict[str, Any]

    def __str__(self) -> str:  # noqa: D105
        lines = ["SafetyViolation — action BLOCKED."]
        for i, v in enumerate(self.violations, 1):
            lines.append(f"  [{i}] {v}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Individual Rule Functions
# ─────────────────────────────────────────────────────────────

def _check_nitrogen(payload: Dict[str, Any], violations: List[str]) -> None:
    n = payload.get("nitrogen_kg_ha")
    if n is not None and float(n) > NutrientLimits.NITROGEN_MAX:
        violations.append(
            f"Nitrogen dose {n} kg/ha exceeds maximum "
            f"{NutrientLimits.NITROGEN_MAX} kg/ha."
        )


def _check_phosphorus(payload: Dict[str, Any], violations: List[str]) -> None:
    p = payload.get("phosphorus_kg_ha")
    if p is not None and float(p) > NutrientLimits.PHOSPHORUS_MAX:
        violations.append(
            f"Phosphorus dose {p} kg/ha exceeds maximum "
            f"{NutrientLimits.PHOSPHORUS_MAX} kg/ha."
        )


def _check_potassium(payload: Dict[str, Any], violations: List[str]) -> None:
    k = payload.get("potassium_kg_ha")
    if k is not None and float(k) > NutrientLimits.POTASSIUM_MAX:
        violations.append(
            f"Potassium dose {k} kg/ha exceeds maximum "
            f"{NutrientLimits.POTASSIUM_MAX} kg/ha."
        )


def _check_water(payload: Dict[str, Any], violations: List[str]) -> None:
    w = payload.get("water_litres_m2")
    if w is not None and float(w) > IrrigationLimits.WATER_MAX_PER_APPLICATION:
        violations.append(
            f"Water application {w} L/m² exceeds maximum "
            f"{IrrigationLimits.WATER_MAX_PER_APPLICATION} L/m²."
        )


def _check_pesticide(payload: Dict[str, Any], violations: List[str]) -> None:
    c = payload.get("pesticide_concentration_mg_l")
    if c is not None and float(c) > PesticideLimits.CONCENTRATION_MAX:
        violations.append(
            f"Pesticide concentration {c} mg/L exceeds maximum "
            f"{PesticideLimits.CONCENTRATION_MAX} mg/L."
        )


def _check_temperature_for_actuation(
    payload: Dict[str, Any], violations: List[str]
) -> None:
    t = payload.get("ambient_temperature_c")
    if t is None:
        return
    t = float(t)
    if t < TemperatureLimits.ACTUATION_TEMP_MIN:
        violations.append(
            f"Ambient temperature {t}°C is below minimum actuation "
            f"threshold {TemperatureLimits.ACTUATION_TEMP_MIN}°C."
        )
    elif t > TemperatureLimits.ACTUATION_TEMP_MAX:
        violations.append(
            f"Ambient temperature {t}°C exceeds maximum actuation "
            f"threshold {TemperatureLimits.ACTUATION_TEMP_MAX}°C."
        )


def _check_spray_humidity(
    payload: Dict[str, Any], violations: List[str]
) -> None:
    """Only applied when the action involves spraying."""
    action_type = str(payload.get("action_type", "")).lower()
    if "spray" not in action_type and "pesticide" not in action_type:
        return
    h = payload.get("ambient_humidity_pct")
    if h is None:
        return
    h = float(h)
    if h < HumidityLimits.SPRAY_HUMIDITY_MIN:
        violations.append(
            f"Humidity {h}% is below minimum for spray actuation "
            f"({HumidityLimits.SPRAY_HUMIDITY_MIN}%). "
            "Spray will drift — action blocked."
        )
    elif h > HumidityLimits.SPRAY_HUMIDITY_MAX:
        violations.append(
            f"Humidity {h}% exceeds maximum for spray actuation "
            f"({HumidityLimits.SPRAY_HUMIDITY_MAX}%). "
            "Fungal infection risk — action blocked."
        )


def _check_negative_values(
    payload: Dict[str, Any], violations: List[str]
) -> None:
    """Any numeric dose value must be non-negative."""
    dose_keys = (
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha",
        "water_litres_m2",
        "pesticide_concentration_mg_l",
    )
    for key in dose_keys:
        val = payload.get(key)
        if val is not None and float(val) < 0:
            violations.append(
                f"Dose value '{key}' is negative ({val}). "
                "Negative dosing is physically invalid."
            )


# ─────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────

_RULE_FUNCTIONS = [
    _check_negative_values,
    _check_nitrogen,
    _check_phosphorus,
    _check_potassium,
    _check_water,
    _check_pesticide,
    _check_temperature_for_actuation,
    _check_spray_humidity,
]


def enforce_safety(action_payload: Dict[str, Any]) -> None:
    """
    Runs all registered safety rules against an action payload.

    Parameters
    ----------
    action_payload : dict
        The actuation parameters the Supervisor intends to execute.
        Recognised keys (all optional — unrecognised keys are ignored):
          - action_type                   : str  (e.g. "deploy_fertilizer", "spray_pesticide")
          - nitrogen_kg_ha                : float
          - phosphorus_kg_ha              : float
          - potassium_kg_ha               : float
          - water_litres_m2               : float
          - pesticide_concentration_mg_l  : float
          - ambient_temperature_c         : float
          - ambient_humidity_pct          : float

    Raises
    ------
    SafetyViolation
        If any rule fires. Contains the full list of violations.

    Returns
    -------
    None
        If all rules pass — action is safe to execute.
    """
    violations: List[str] = []

    for rule_fn in _RULE_FUNCTIONS:
        rule_fn(action_payload, violations)

    if violations:
        raise SafetyViolation(
            violations=violations,
            action_payload=action_payload,
        )


def get_safety_summary() -> Dict[str, Any]:
    """
    Returns the current safety limits as a dict.
    Used by the Supervisor to inject limits into the LLM system prompt.
    """
    return {
        "nutrient_limits_kg_ha": {
            "nitrogen_max":   NutrientLimits.NITROGEN_MAX,
            "phosphorus_max": NutrientLimits.PHOSPHORUS_MAX,
            "potassium_max":  NutrientLimits.POTASSIUM_MAX,
        },
        "irrigation_limits": {
            "water_max_litres_m2": IrrigationLimits.WATER_MAX_PER_APPLICATION,
        },
        "pesticide_limits": {
            "concentration_max_mg_l": PesticideLimits.CONCENTRATION_MAX,
        },
        "actuation_temperature_c": {
            "min": TemperatureLimits.ACTUATION_TEMP_MIN,
            "max": TemperatureLimits.ACTUATION_TEMP_MAX,
        },
        "spray_humidity_pct": {
            "min": HumidityLimits.SPRAY_HUMIDITY_MIN,
            "max": HumidityLimits.SPRAY_HUMIDITY_MAX,
        },
    }