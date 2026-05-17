"""
core/supervisor.py  —  Team Alpha
SupervisorBrain: LLM orchestrator powered by Llama 3.3 70B via Groq API.

Responsibilities
----------------
1. Parse user intent and route to the correct sub-agent(s).
2. Aggregate sub-agent JSON responses into a coherent LLM prompt.
3. Execute Groq function-calling to invoke tools deterministically.
4. Pass all actuation payloads through safety_rules.enforce_safety().
5. Log every decision event to ChromaDB via MemoryNode.
6. Return a structured final reply dict to app.py.

Final reply schema
------------------
{
    "reply":          str,           # LLM-generated natural language response
    "tool_calls_made": list[dict],   # [{name, arguments, result}, ...]
    "safety_blocked": bool,          # True if enforce_safety raised
    "safety_details": str | None,    # Violation description if blocked
    "agent_reports":  list[dict],    # Raw AgentResponse dicts from sub-agents
}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq

from core.safety_rules import enforce_safety, get_safety_summary, SafetyViolation

# MemoryNode imported lazily to avoid ChromaDB startup cost when only
# safety or routing is needed in unit tests.
_memory_node = None

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Tool / Function Definitions  (Groq function-calling schema)
# ─────────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_soil",
            "description": (
                "Run the SoilIntelligenceAgent ensemble (GRU + LSTM + 1D-CNN) "
                "on real-time NPK and moisture sensor data. "
                "Returns soil health status and recommended action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nitrogen":    {"type": "number", "description": "Nitrogen level (mg/kg)"},
                    "phosphorus":  {"type": "number", "description": "Phosphorus level (mg/kg)"},
                    "potassium":   {"type": "number", "description": "Potassium level (mg/kg)"},
                    "moisture":    {"type": "number", "description": "Soil moisture (%)"},
                },
                "required": ["nitrogen", "phosphorus", "potassium", "moisture"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_weather",
            "description": (
                "Run the MicroClimateAgent (1D-CNN anomaly reflex + GRU/LSTM forecasting) "
                "on a 20-step sliding window of temperature/humidity telemetry. "
                "Returns anomaly flag and next-step forecast."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {
                        "type": "array",
                        "description": "List of 20 [temperature_c, humidity_pct] pairs (chronological).",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "minItems": 20,
                        "maxItems": 20,
                    },
                },
                "required": ["window"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_crop_image",
            "description": (
                "Run the CropPathologyAgent (DenseNet121) on a tomato image "
                "to classify its state as Reject, Ripe, or Unripe. "
                "Triggers when a visual crop assessment is required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute or relative path to the image file.",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agronomy_pipeline",
            "description": (
                "Run the MasterAgronomyAgent sequential pipeline "
                "(ConvNeXt Soil Vision → Random Forest Crop Recommender → XGBoost Fertilizer). "
                "Call this when soil health is poor or the user requests a fertilizer recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nitrogen":    {"type": "number"},
                    "phosphorus":  {"type": "number"},
                    "potassium":   {"type": "number"},
                    "temperature": {"type": "number", "description": "°C"},
                    "humidity":    {"type": "number", "description": "%"},
                    "ph":          {"type": "number", "description": "Soil pH"},
                    "rainfall":    {"type": "number", "description": "mm"},
                },
                "required": ["nitrogen", "phosphorus", "potassium",
                             "temperature", "humidity", "ph", "rainfall"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Query the ChromaDB vector store for semantically similar past events. "
                "Use this to answer questions about historical sensor readings, "
                "past diagnoses, or previous recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search past events.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of past events to retrieve (default 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────
# System Prompt Builder
# ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    safety = get_safety_summary()
    return f"""You are the AgriBot Supervisor — the central intelligence of a precision agriculture decision support system.

## Your Role
Route user queries to the correct specialist agents, aggregate their JSON reports, and issue clear, actionable recommendations to the farmer.

## Available Agents
- **SoilIntelligenceAgent** (analyze_soil): Binary health classification on NPK + moisture time-series.
- **MicroClimateAgent** (analyze_weather): Anomaly detection + next-step temperature/humidity forecast.
- **CropPathologyAgent** (analyze_crop_image): Tomato ripeness/quality classification via DenseNet121. Returns one of three classes: Reject (0), Ripe (1), Unripe (2).
- **MasterAgronomyAgent** (run_agronomy_pipeline): Full soil-type → crop recommendation → fertilizer chain via XGBoost and Random Forest.
- **MemoryNode** (recall_memory): Semantic retrieval of past events from ChromaDB.

## Routing Rules
1. Soil NPK / moisture query → call `analyze_soil`.
2. Temperature / humidity / weather query → call `analyze_weather`.
3. Tomato image uploaded → call `analyze_crop_image`.
4. Fertilizer / crop recommendation → call `run_agronomy_pipeline`.
5. "What happened before / history / past" → call `recall_memory`.
6. Multiple concerns → call multiple tools sequentially.

## Safety Limits (NEVER recommend exceeding these)
{json.dumps(safety, indent=2)}

## Output Rules
- Always cite which agent produced each finding.
- If a safety limit would be exceeded, refuse the action and explain why.
- Be concise. Farmers need clear numbers and direct actions, not verbose explanations.
- Never fabricate sensor readings. Only report what the agents return.
"""


# ─────────────────────────────────────────────────────────────
# SupervisorBrain
# ─────────────────────────────────────────────────────────────

class SupervisorBrain:
    """
    LLM Supervisor powered by Llama 3.3 70B via Groq.

    Parameters
    ----------
    agents : dict
        Keys: "soil", "weather", "vision", "agronomy"
        Values: instantiated agent objects with a `.run()` method.
    memory : MemoryNode | None
        ChromaDB memory node. If None, memory tools are disabled.
    groq_api_key : str | None
        Overrides GROQ_API_KEY env var (useful for testing).
    model : str
        Groq model ID. Default: llama-3.3-70b-versatile.
    max_tool_rounds : int
        Maximum sequential tool-call rounds before forcing a final answer.
    """

    MODEL = "llama-3.3-70b-versatile"
    MAX_TOOL_ROUNDS = 5

    def __init__(
        self,
        agents: Dict[str, Any],
        memory=None,
        groq_api_key: str | None = None,
        model: str | None = None,
        max_tool_rounds: int | None = None,
    ) -> None:
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Set it in .env or pass groq_api_key= explicitly."
            )

        self.client         = Groq(api_key=api_key)
        self.agents         = agents
        self.memory         = memory
        self.model          = model or self.MODEL
        self.max_tool_rounds = max_tool_rounds or self.MAX_TOOL_ROUNDS
        self._system_prompt  = _build_system_prompt()

    # ── public entry point ────────────────────────────────────

    def chat(
        self,
        user_message: str,
        history: List[Dict[str, str]] | None = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Main agentic loop.

        Parameters
        ----------
        user_message : str
        history      : list of {role, content} for multi-turn context
        verbose      : if True, prints tool call details to stdout

        Returns
        -------
        dict — final reply schema (see module docstring)
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        tool_calls_made: List[Dict[str, Any]] = []
        agent_reports:   List[Dict[str, Any]] = []
        safety_blocked = False
        safety_details: str | None = None

        for round_n in range(self.max_tool_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
            )

            message = response.choices[0].message

            # ── No more tool calls → final answer ──────────────
            if not message.tool_calls:
                break

            # ── Process each tool call ─────────────────────────
            messages.append(message)   # assistant message with tool_calls

            for tc in message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                if verbose:
                    print(f"[Round {round_n+1}] Tool call: {fn_name}({fn_args})")

                result, report, blocked, detail = self._dispatch(fn_name, fn_args)

                if blocked:
                    safety_blocked = True
                    safety_details = detail

                if report:
                    agent_reports.append(report)

                tool_calls_made.append({
                    "name":      fn_name,
                    "arguments": fn_args,
                    "result":    result,
                })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result),
                })

        # ── Log to memory ──────────────────────────────────────
        if self.memory and agent_reports:
            self._log_to_memory(user_message, agent_reports)

        # ── Final LLM synthesis ────────────────────────────────
        final_text = (
            message.content
            if message.content
            else self._force_final_answer(messages)
        )

        return {
            "reply":           final_text,
            "tool_calls_made": tool_calls_made,
            "safety_blocked":  safety_blocked,
            "safety_details":  safety_details,
            "agent_reports":   agent_reports,
        }

    # ── private: tool dispatch ────────────────────────────────

    def _dispatch(
        self, fn_name: str, fn_args: Dict[str, Any]
    ):
        """
        Routes a tool call to the correct agent, enforces safety, returns
        (result_dict, agent_report_or_None, safety_blocked, safety_detail).
        """
        try:
            if fn_name == "analyze_soil":
                agent = self.agents.get("soil")
                if agent is None:
                    return {"error": "SoilAgent not loaded"}, None, False, None
                report = agent.run(fn_args)
                return report, report, False, None

            elif fn_name == "analyze_weather":
                agent = self.agents.get("weather")
                if agent is None:
                    return {"error": "WeatherAgent not loaded"}, None, False, None
                report = agent.run(fn_args)
                return report, report, False, None

            elif fn_name == "analyze_crop_image":
                agent = self.agents.get("vision")
                if agent is None:
                    return {"error": "VisionAgent not loaded"}, None, False, None
                report = agent.run(fn_args)
                return report, report, False, None

            elif fn_name == "run_agronomy_pipeline":
                agent = self.agents.get("agronomy")
                if agent is None:
                    return {"error": "AgronomyAgent not loaded"}, None, False, None
                report = agent.run(fn_args)

                # Safety gate: extract and validate actuation payload
                action_params = report.get("action_parameters", {})
                safety_payload = self._build_safety_payload(fn_args, action_params)
                try:
                    enforce_safety(safety_payload)
                except SafetyViolation as sv:
                    report["status"] = "Blocked"
                    report["finding"] += f"\n\n⛔ Safety guardrail blocked actuation: {sv}"
                    return report, report, True, str(sv)

                return report, report, False, None

            elif fn_name == "recall_memory":
                if self.memory is None:
                    return {"result": "Memory system not available."}, None, False, None
                results = self.memory.query(
                    fn_args["query"],
                    n_results=fn_args.get("n_results", 3),
                )
                return {"memory_results": results}, None, False, None

            else:
                return {"error": f"Unknown tool: {fn_name}"}, None, False, None

        except Exception as exc:
            return {"error": f"Tool execution failed: {exc}"}, None, False, None

    def _force_final_answer(self, messages: List[Dict]) -> str:
        """
        Requests a final natural language answer when the LLM stopped
        without generating text (edge case after max tool rounds).
        """
        messages.append({
            "role": "user",
            "content": (
                "Based on all the agent reports above, provide your final "
                "recommendation to the farmer in plain language."
            ),
        })
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    def _log_to_memory(
        self,
        user_message: str,
        agent_reports: List[Dict[str, Any]],
    ) -> None:
        """Serialises the session event and stores it in ChromaDB."""
        try:
            event_text = (
                f"User query: {user_message}\n"
                + "\n".join(
                    f"[{r.get('unit', 'unknown')}] {r.get('finding', '')}"
                    for r in agent_reports
                )
            )
            self.memory.store(event_text)
        except Exception:
            pass  # Memory failure must never crash the main loop

    @staticmethod
    def _build_safety_payload(
        fn_args: Dict[str, Any],
        action_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combines LLM-extracted args + agent action_parameters into the
        flat dict expected by enforce_safety().

        action_type is read from fn_args["recommended_action"] (the agent's
        machine-readable action token, e.g. "deploy_fertilizer") with a safe
        fallback to "deploy_fertilizer" if absent.  The old code incorrectly
        read action_params.get("fertilizer"), which is the fertilizer *name*.
        """
        return {
            "action_type":           action_params.get("recommended_action", "deploy_fertilizer"),
            "nitrogen_kg_ha":        fn_args.get("nitrogen"),
            "phosphorus_kg_ha":      fn_args.get("phosphorus"),
            "potassium_kg_ha":       fn_args.get("potassium"),
            "ambient_temperature_c": fn_args.get("temperature"),
            "ambient_humidity_pct":  fn_args.get("humidity"),
        }