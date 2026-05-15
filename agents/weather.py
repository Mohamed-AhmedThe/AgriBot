"""
agents/weather.py  —  Team Gamma
MicroClimateAgent: sliding-window micro-climate forecasting and anomaly detection.

Architecture (verified against saved .pth state dicts)
------------------------------------------------------
All three models are REGRESSORS outputting (next_temp, next_humidity):

  WeatherGRURegressor   — single-layer GRU,  hidden=64
  WeatherLSTMRegressor  — single-layer LSTM, hidden=64
  Weather1DCNNRegressor — Conv1d(2,64,k=3,p=1) -> MaxPool1d(k=3,s=2) -> fc1(576,32) -> fc2(32,2)

Anomaly detection strategy
--------------------------
There is no sigmoid classifier. Anomalies are detected via two independent signals:
  1. Inter-model disagreement  — if GRU/LSTM/CNN forecasts diverge beyond DISAGREEMENT_THRESHOLD
     the ensemble is uncertain, which signals sensor noise or a regime shift.
  2. Deterministic threshold checks — hard agronomic limits on the forecast values themselves.

Input contract  (from Supervisor)
----------------------------------
{
    "window": [[temp_c, humidity_pct], ...]   # exactly 20 pairs, chronological
}

Output contract  (AgentResponse)
---------------------------------
{
    "unit":               "MicroClimateAgent",
    "status":             "Nominal" | "Warning" | "Error",
    "finding":            str,
    "confidence_score":   float,
    "recommended_action": "none" | "issue_weather_alert" | "check_sensors",
    "anomaly_detected":   bool,
    "forecast": {
        "next_temperature_c":  float,
        "next_humidity_pct":   float,
        "model_agreement":     float,
        "individual_forecasts": {
            "gru":  [float, float],
            "lstm": [float, float],
            "cnn":  [float, float]
        }
    }
}
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List

from agents.base import BaseAgent

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
WINDOW_SIZE: int   = 20
N_FEATURES:  int   = 2

DISAGREEMENT_THRESHOLD: float = 0.15

TEMP_ALERT_HIGH: float = 40.0
TEMP_ALERT_LOW:  float = 10.0
HUM_ALERT_HIGH:  float = 90.0
HUM_ALERT_LOW:   float = 20.0


# ----------------------------------------------------------------
# MicroClimateAgent
# ----------------------------------------------------------------

class MicroClimateAgent(BaseAgent):
    """
    Parameters
    ----------
    models_dict : dict
        Keys "gru", "lstm", "cnn" -> loaded nn.Module instances in eval() mode.
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted on [Temperature, Humidity] training data.
    device : torch.device
    """

    def __init__(
        self,
        models_dict: Dict[str, nn.Module],
        scaler,
        device: torch.device,
    ) -> None:
        self.gru    = models_dict.get("gru")
        self.lstm   = models_dict.get("lstm")
        self.cnn    = models_dict.get("cnn")
        self.scaler = scaler
        self.device = device

        active = [k for k, v in models_dict.items() if v is not None]
        if not active:
            raise ValueError("MicroClimateAgent: no models provided.")
        print(f"[MicroClimateAgent] Active models: {active}")

    # -- public entry point ---------------------------------------

    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            window = self._parse_window(input_data)
            scaled = self._scale(window)

            individual_scaled = self._run_all_models(scaled)

            mean_scaled = np.mean(list(individual_scaled.values()), axis=0)
            mean_real   = self._inverse_scale(mean_scaled)

            next_temp = round(float(mean_real[0]), 2)
            next_hum  = round(float(mean_real[1]), 2)

            disagreement   = self._compute_disagreement(individual_scaled)
            anomaly_disagr = disagreement > DISAGREEMENT_THRESHOLD
            threshold_msg  = self._check_thresholds(next_temp, next_hum)
            anomaly_flag   = anomaly_disagr or (threshold_msg is not None)

            status, action = ("Warning", "issue_weather_alert") if anomaly_flag \
                             else ("Nominal", "none")

            agreement_score = round(float(
                np.clip(1.0 - disagreement / DISAGREEMENT_THRESHOLD, 0.0, 1.0)
            ), 3)

            individual_real = {
                name: [round(float(v[0]), 2), round(float(v[1]), 2)]
                for name, v in {
                    n: self._inverse_scale(p)
                    for n, p in individual_scaled.items()
                }.items()
            }

            finding = self._build_finding(
                next_temp, next_hum, anomaly_disagr, disagreement, threshold_msg
            )

            return {
                "unit":               "MicroClimateAgent",
                "status":             status,
                "finding":            finding,
                "confidence_score":   agreement_score,
                "recommended_action": action,
                "anomaly_detected":   anomaly_flag,
                "forecast": {
                    "next_temperature_c":   next_temp,
                    "next_humidity_pct":    next_hum,
                    "model_agreement":      agreement_score,
                    "individual_forecasts": individual_real,
                },
            }

        except (ValueError, KeyError) as exc:
            return {
                "unit":               "MicroClimateAgent",
                "status":             "Error",
                "finding":            f"Input validation failed: {exc}",
                "confidence_score":   0.0,
                "recommended_action": "check_sensors",
                "anomaly_detected":   False,
                "forecast":           {},
            }

    # -- private helpers ------------------------------------------

    def _parse_window(self, input_data: Dict[str, Any]) -> np.ndarray:
        raw = input_data.get("window")
        if raw is None:
            raise ValueError("Missing key 'window' in input_data")
        arr = np.array(raw, dtype=np.float32)
        if arr.shape != (WINDOW_SIZE, N_FEATURES):
            raise ValueError(
                f"'window' must be shape ({WINDOW_SIZE}, {N_FEATURES}), got {arr.shape}"
            )
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("'window' contains NaN or Inf values")
        return arr

    def _scale(self, window: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return window
        return self.scaler.transform(window).astype(np.float32)

    def _inverse_scale(self, vec: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return vec
        return self.scaler.inverse_transform(vec.reshape(1, -1)).flatten()

    def _run_all_models(self, scaled: np.ndarray) -> Dict[str, np.ndarray]:
        """
        GRU/LSTM expect : (batch=1, seq=20, features=2)
        CNN expects      : (batch=1, features=2, seq=20)  -- transposed
        """
        results: Dict[str, np.ndarray] = {}
        rnn_tensor = torch.tensor(scaled[np.newaxis], device=self.device)
        cnn_tensor = torch.tensor(scaled.T[np.newaxis], device=self.device)

        with torch.no_grad():
            if self.gru is not None:
                results["gru"]  = self.gru(rnn_tensor).cpu().numpy().flatten()
            if self.lstm is not None:
                results["lstm"] = self.lstm(rnn_tensor).cpu().numpy().flatten()
            if self.cnn is not None:
                results["cnn"]  = self.cnn(cnn_tensor).cpu().numpy().flatten()

        return results

    @staticmethod
    def _compute_disagreement(preds: Dict[str, np.ndarray]) -> float:
        if len(preds) < 2:
            return 0.0
        values   = list(preds.values())
        max_diff = 0.0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                diff = float(np.abs(values[i] - values[j]).max())
                if diff > max_diff:
                    max_diff = diff
        return max_diff

    @staticmethod
    def _check_thresholds(temp: float, hum: float):
        breaches: List[str] = []
        if temp >= TEMP_ALERT_HIGH:
            breaches.append(f"Temperature critical HIGH ({temp}C >= {TEMP_ALERT_HIGH}C)")
        elif temp <= TEMP_ALERT_LOW:
            breaches.append(f"Temperature critical LOW ({temp}C <= {TEMP_ALERT_LOW}C)")
        if hum >= HUM_ALERT_HIGH:
            breaches.append(f"Humidity critical HIGH ({hum}% >= {HUM_ALERT_HIGH}%)")
        elif hum <= HUM_ALERT_LOW:
            breaches.append(f"Humidity critical LOW ({hum}% <= {HUM_ALERT_LOW}%)")
        return " | ".join(breaches) if breaches else None

    @staticmethod
    def _build_finding(next_temp, next_hum, anomaly_disagr, disagreement, threshold_msg):
        parts = [f"Forecast: next temperature {next_temp}C, next humidity {next_hum}%."]
        if anomaly_disagr:
            parts.append(
                f"Model disagreement {disagreement:.3f} exceeds threshold "
                f"{DISAGREEMENT_THRESHOLD} -- possible sensor noise or regime shift."
            )
        if threshold_msg:
            parts.append(f"Threshold breach -- {threshold_msg}.")
        if not anomaly_disagr and not threshold_msg:
            parts.append("Micro-climate within nominal operating parameters.")
        return " ".join(parts)