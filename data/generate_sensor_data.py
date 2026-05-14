"""
data/generate_sensor_data.py  —  Team Gamma
Generates a 12,000-row synthetic hardware telemetry CSV matching Table 3
of Murad et al. (AgriEngineering 8(1), 2026).

Usage
-----
    python data/generate_sensor_data.py                  # writes data/soil_weather.csv
    python data/generate_sensor_data.py --rows 5000      # custom row count
    python data/generate_sensor_data.py --out data/test.csv

Output schema
-------------
Timestamp, Potassium, Phosphorus, Nitrogen, SoilMoisture,
Humidity, Temperature, Distance, SoilHealth
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Generation Config
# ─────────────────────────────────────────────────────────────

class TelemetryConfig:
    RANDOM_SEED:   int      = 42
    N_SAMPLES:     int      = 12_000
    START_TIME:    datetime = datetime(2025, 8, 1, 6, 0, 0)
    INTERVAL:      timedelta = timedelta(minutes=5)

    # NPK / moisture  (mean, std, min, max)
    POTASSIUM:     tuple = (95.0,  45.0,  10.0,  200.0)
    PHOSPHORUS:    tuple = (22.0,  15.0,   3.0,   80.0)
    NITROGEN:      tuple = (42.0,  25.0,   5.0,  100.0)
    SOIL_MOISTURE: tuple = (55.0,  20.0,  15.0,   95.0)

    # Temperature / humidity (diurnal sinusoid + Gaussian noise)
    TEMP_BASE:     tuple = (30.0,   2.0,  20.0,   42.0)  # (mean, std, min, max)
    TEMP_DIURNAL_AMP: float = 5.0                         # ± °C peak-to-peak / 2

    HUM_BASE:      tuple = (60.0,   8.0,  25.0,   95.0)
    HUM_DIURNAL_AMP: float = 15.0                         # inverse phase to temp

    # Distance sensor (0 = crop present, >0 = obstacle/gap in cm)
    DISTANCE_ZERO_PROB: float = 0.92
    DISTANCE_RANGE:     tuple = (10, 400)                  # cm

    # Soil health labelling thresholds (deterministic rule — mirrors paper)
    SOIL_HEALTH_THRESHOLDS: dict = dict(
        nitrogen_min=30.0,
        phosphorus_min=15.0,
        potassium_min=60.0,
        moisture_min=30.0,
        moisture_max=85.0,
    )


# ─────────────────────────────────────────────────────────────
# Core Generator
# ─────────────────────────────────────────────────────────────

def _diurnal_phase(hour: int, amplitude: float, phase_offset: int = 6) -> float:
    """
    Sinusoidal diurnal variation peaking at (phase_offset + 6) h.
    Positive amplitude → peak at mid-day (temperature).
    Negative amplitude → trough at mid-day (humidity).
    """
    return amplitude * np.sin((hour - phase_offset) / 24.0 * 2.0 * np.pi)


def generate_telemetry(cfg: TelemetryConfig | None = None) -> pd.DataFrame:
    """
    Generates synthetic sensor telemetry.

    Parameters
    ----------
    cfg : TelemetryConfig, optional
        Override defaults for testing.

    Returns
    -------
    pd.DataFrame
        Shape (N_SAMPLES, 9) with columns matching Table 3 schema.
    """
    if cfg is None:
        cfg = TelemetryConfig()

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    rows: list[dict] = []

    for i in range(cfg.N_SAMPLES):
        ts   = cfg.START_TIME + i * cfg.INTERVAL
        hour = ts.hour

        # ── Soil sensors ──────────────────────────────────────
        potassium = float(np.clip(
            rng.normal(*cfg.POTASSIUM[:2]), *cfg.POTASSIUM[2:]))
        phosphorus = float(np.clip(
            rng.normal(*cfg.PHOSPHORUS[:2]), *cfg.PHOSPHORUS[2:]))
        nitrogen = float(np.clip(
            rng.normal(*cfg.NITROGEN[:2]), *cfg.NITROGEN[2:]))
        soil_moisture = float(np.clip(
            rng.normal(*cfg.SOIL_MOISTURE[:2]), *cfg.SOIL_MOISTURE[2:]))

        # ── Climate sensors (diurnal + noise) ─────────────────
        temperature = float(np.clip(
            rng.normal(*cfg.TEMP_BASE[:2]) + _diurnal_phase(hour, cfg.TEMP_DIURNAL_AMP),
            *cfg.TEMP_BASE[2:]))
        humidity = float(np.clip(
            rng.normal(*cfg.HUM_BASE[:2]) - _diurnal_phase(hour, cfg.HUM_DIURNAL_AMP),
            *cfg.HUM_BASE[2:]))

        # ── Distance sensor ───────────────────────────────────
        distance = (
            0 if rng.random() < cfg.DISTANCE_ZERO_PROB
            else int(rng.integers(*cfg.DISTANCE_RANGE))
        )

        rows.append({
            "Timestamp":    ts.strftime("%-d %B %Y %H:%M")
                            if os.name != "nt"
                            else ts.strftime("%#d %B %Y %H:%M"),
            "Potassium":    round(potassium,    1),
            "Phosphorus":   round(phosphorus,   1),
            "Nitrogen":     round(nitrogen,     1),
            "SoilMoisture": round(soil_moisture, 1),
            "Humidity":     int(round(humidity)),
            "Temperature":  int(round(temperature)),
            "Distance":     distance,
        })

    df = pd.DataFrame(rows)

    # ── Deterministic SoilHealth label ────────────────────────
    t = cfg.SOIL_HEALTH_THRESHOLDS
    df["SoilHealth"] = (
        (df["Nitrogen"]     > t["nitrogen_min"])   &
        (df["Phosphorus"]   > t["phosphorus_min"]) &
        (df["Potassium"]    > t["potassium_min"])  &
        (df["SoilMoisture"] > t["moisture_min"])   &
        (df["SoilMoisture"] < t["moisture_max"])
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# Sliding-Window Builder  (used by WeatherAgent training/inference)
# ─────────────────────────────────────────────────────────────

def build_weather_windows(
    df: pd.DataFrame,
    window_size: int = 20,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a flat telemetry DataFrame into overlapping windows for
    time-series model training.

    Parameters
    ----------
    df           : DataFrame containing at least Temperature and Humidity columns.
    window_size  : Number of timesteps per window (must match model training config).
    feature_cols : Columns to use. Defaults to ["Temperature", "Humidity"].

    Returns
    -------
    X : ndarray, shape (n_windows, window_size, n_features)
    y : ndarray, shape (n_windows, n_features)  — next-step targets
    """
    if feature_cols is None:
        feature_cols = ["Temperature", "Humidity"]

    data   = df[feature_cols].to_numpy(dtype=np.float32)
    X_list = []
    y_list = []

    for i in range(len(data) - window_size):
        X_list.append(data[i : i + window_size])
        y_list.append(data[i + window_size])

    return np.array(X_list), np.array(y_list)


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AgriBot synthetic sensor telemetry CSV."
    )
    parser.add_argument(
        "--rows", type=int, default=TelemetryConfig.N_SAMPLES,
        help=f"Number of rows to generate (default: {TelemetryConfig.N_SAMPLES})",
    )
    parser.add_argument(
        "--out", type=str, default="data/soil_weather.csv",
        help="Output CSV path (default: data/soil_weather.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=TelemetryConfig.RANDOM_SEED,
        help=f"Random seed (default: {TelemetryConfig.RANDOM_SEED})",
    )
    args = parser.parse_args()

    cfg           = TelemetryConfig()
    cfg.N_SAMPLES = args.rows
    cfg.RANDOM_SEED = args.seed

    print(f"Generating {cfg.N_SAMPLES:,} rows …")
    df = generate_telemetry(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    balance = df["SoilHealth"].value_counts().to_dict()
    healthy_pct = balance.get(1, 0) / len(df) * 100
    print(f"✓  Written {len(df):,} rows → {out_path}")
    print(f"   SoilHealth balance: {balance}  ({healthy_pct:.1f}% healthy)")
    print(f"   Temperature range : {df['Temperature'].min()}–{df['Temperature'].max()} °C")
    print(f"   Humidity range    : {df['Humidity'].min()}–{df['Humidity'].max()} %")


if __name__ == "__main__":
    main()