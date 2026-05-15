"""
agents/base.py
Abstract base contract for all AgriBot sub-agents.
Every agent MUST return a dict that strictly conforms to the AgentResponse schema.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal


# ─────────────────────────────────────────────
# JSON API CONTRACT  (enforced at runtime)
# ─────────────────────────────────────────────
REQUIRED_KEYS: frozenset[str] = frozenset({
    "unit",
    "status",
    "finding",
    "confidence_score",
    "recommended_action",
})

VALID_STATUSES: frozenset[str] = frozenset({
    "Nominal",
    "Warning",
    "Action Required",
    "Error",
})


def validate_response(response: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    Validates that an agent response conforms to the JSON API contract.
    Raises ValueError on contract violation so the Supervisor can catch it.
    Returns the response unchanged if valid.
    """
    missing = REQUIRED_KEYS - response.keys()
    if missing:
        raise ValueError(
            f"[{agent_name}] Contract violation — missing keys: {missing}"
        )
    if response["status"] not in VALID_STATUSES:
        raise ValueError(
            f"[{agent_name}] Contract violation — invalid status '{response['status']}'. "
            f"Must be one of {VALID_STATUSES}"
        )
    if not isinstance(response["confidence_score"], (int, float)):
        raise ValueError(
            f"[{agent_name}] Contract violation — confidence_score must be numeric"
        )
    return response


class BaseAgent(ABC):
    """
    Abstract base class for all AgriBot agents.

    Subclasses MUST implement `evaluate()` and return a dict that passes
    `validate_response()`. The `run()` method enforces this automatically.

    AgentResponse schema
    ────────────────────
    {
        "unit":               str,   # Agent identifier, e.g. "MicroClimateAgent"
        "status":             str,   # One of: Nominal | Warning | Action Required | Error
        "finding":            str,   # Human-readable summary for the Supervisor LLM
        "confidence_score":   float, # [0.0, 1.0]
        "recommended_action": str,   # Machine-readable action token
        # Optional keys agents may append:
        "action_parameters":  dict,  # Payload for actuation / chaining
        "anomaly_detected":   bool,  # Reflex trigger flag
        "forecast":           dict,  # Weather agent forward predictions
    }
    """

    @abstractmethod
    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core inference method.

        Parameters
        ----------
        input_data : dict
            Sensor telemetry or image payload passed by the Supervisor.

        Returns
        -------
        dict
            Must conform to AgentResponse schema.
        """
        ...

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public entry point called by the Supervisor.
        Wraps `evaluate()` with contract validation and top-level exception guard.
        """
        agent_name = self.__class__.__name__
        try:
            response = self.evaluate(input_data)
            return validate_response(response, agent_name)
        except ValueError:
            # Contract violations propagate — Supervisor must handle them
            raise
        except Exception as exc:
            # Unexpected runtime failure — return a well-formed Error response
            return {
                "unit": agent_name,
                "status": "Error",
                "finding": f"Unhandled exception in {agent_name}: {exc}",
                "confidence_score": 0.0,
                "recommended_action": "system_reset",
            }