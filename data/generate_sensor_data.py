"""
data/generate_sensor_data.py  —  Team Gamma
Generates a 12,000-row synthetic hardware telemetry CSV.

Columns: Timestamp, Potassium, Phosphorous, Nitrogen, Moisture,
         PH, Rainfall, Humidity, Temperature, Distance, SoilHealth

Usage
-----
    python data/generate_sensor_data.py                  # writes data/soil_weather.csv
    python data/generate_sensor_data.py --rows 5000
    python data/generate_sensor_data.py --out data/test.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Generation Config
# ─────────────────────────────────────────────────────────────

class TelemetryConfig:
    RANDOM_SEED:   int       = 42
    N_SAMPLES:     int       = 12_000
    START_TIME:    datetime  = datetime(2025, 8, 1, 6, 0, 0)
    INTERVAL:      timedelta = timedelta(minutes=5)

    # NPK / moisture / pH / Rainfall  (mean, std, min, max)
    POTASSIUM:   tuple = (95.0,  45.0,  10.0,  200.0)
    PHOSPHOROUS: tuple = (22.0,  15.0,   3.0,   80.0)  # spelled with 'o' — matches XGBoost training
    NITROGEN:    tuple = (42.0,  25.0,   5.0,  100.0)
    MOISTURE:    tuple = (55.0,  20.0,  15.0,   95.0)
    PH:          tuple = ( 6.5,   0.5,   4.0,    9.0)
    RAINFALL:    tuple = (100.0, 40.0,   0.0,  300.0)

    # Temperature / humidity (diurnal sinusoid + Gaussian noise)
    TEMP_BASE:        tuple = (30.0,  2.0, 20.0, 42.0)
    TEMP_DIURNAL_AMP: float = 5.0

    HUM_BASE:        tuple = (60.0,  8.0, 25.0, 95.0)
    HUM_DIURNAL_AMP: float = 15.0   # inverse phase to temperature

    # Distance sensor (0 = crop present, >0 = obstacle/gap in cm)
    DISTANCE_ZERO_PROB: float = 0.92
    DISTANCE_RANGE:     tuple = (10, 400)

    # Hardware failure simulation — 0.5% chance a sensor reads 999.0
    SENSOR_FAILURE_PROB: float = 0.005

    # Deterministic SoilHealth label thresholds
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
    """Sinusoidal diurnal variation. Positive = peak at mid-day (temp), negative = trough (hum)."""
    return amplitude * np.sin((hour - phase_offset) / 24.0 * 2.0 * np.pi)


def generate_telemetry(cfg: TelemetryConfig | None = None) -> pd.DataFrame:
    """
    Generates synthetic sensor telemetry.

    Returns
    -------
    pd.DataFrame  shape (N_SAMPLES, 11)
    """
    if cfg is None:
        cfg = TelemetryConfig()

    rng  = np.random.default_rng(cfg.RANDOM_SEED)
    rows: list[dict] = []

    for i in range(cfg.N_SAMPLES):
        ts   = cfg.START_TIME + i * cfg.INTERVAL
        hour = ts.hour

        # ── Soil sensors ──────────────────────────────────────
        potassium   = float(np.clip(rng.normal(*cfg.POTASSIUM[:2]),   *cfg.POTASSIUM[2:]))
        phosphorous = float(np.clip(rng.normal(*cfg.PHOSPHOROUS[:2]), *cfg.PHOSPHOROUS[2:]))
        nitrogen    = float(np.clip(rng.normal(*cfg.NITROGEN[:2]),    *cfg.NITROGEN[2:]))
        moisture    = float(np.clip(rng.normal(*cfg.MOISTURE[:2]),    *cfg.MOISTURE[2:]))
        ph          = float(np.clip(rng.normal(*cfg.PH[:2]),          *cfg.PH[2:]))
        rainfall    = float(np.clip(rng.normal(*cfg.RAINFALL[:2]),    *cfg.RAINFALL[2:]))

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

        # ── Hardware failure injection ────────────────────────
        if rng.random() < cfg.SENSOR_FAILURE_PROB:
            temperature = 999.0   # simulates a broken thermistor

        rows.append({
            "Timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Potassium":   round(potassium,   2),
            "Phosphorous": round(phosphorous, 2),
            "Nitrogen":    round(nitrogen,    2),
            "Moisture":    round(moisture,    2),
            "PH":          round(ph,          2),
            "Rainfall":    round(rainfall,    2),
            "Humidity":    round(humidity,    2),
            "Temperature": round(temperature, 2),
            "Distance":    distance,
        })

    df = pd.DataFrame(rows)

    # ── Deterministic SoilHealth label ────────────────────────
    t = cfg.SOIL_HEALTH_THRESHOLDS
    df["SoilHealth"] = (
        (df["Nitrogen"]    > t["nitrogen_min"])   &
        (df["Phosphorous"] > t["phosphorus_min"]) &
        (df["Potassium"]   > t["potassium_min"])  &
        (df["Moisture"]    > t["moisture_min"])   &
        (df["Moisture"]    < t["moisture_max"])
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
    Converts a flat telemetry DataFrame into overlapping windows.

    Returns
    -------
    X : ndarray  (n_windows, window_size, n_features)
    y : ndarray  (n_windows, n_features)  — next-step targets
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
    parser.add_argument("--rows", type=int, default=TelemetryConfig.N_SAMPLES)
    parser.add_argument("--out",  type=str, default="data/soil_weather.csv")
    parser.add_argument("--seed", type=int, default=TelemetryConfig.RANDOM_SEED)
    args = parser.parse_args()

    cfg             = TelemetryConfig()
    cfg.N_SAMPLES   = args.rows
    cfg.RANDOM_SEED = args.seed

    print(f"Generating {cfg.N_SAMPLES:,} rows ...")
    df = generate_telemetry(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    balance     = df["SoilHealth"].value_counts().to_dict()
    healthy_pct = balance.get(1, 0) / len(df) * 100
    failures    = (df["Temperature"] == 999.0).sum()

    print(f"  Written      : {len(df):,} rows -> {out_path}")
    print(f"  SoilHealth   : {balance}  ({healthy_pct:.1f}% healthy)")
    print(f"  Temp range   : {df.loc[df['Temperature'] != 999.0, 'Temperature'].min()}"
          f"–{df.loc[df['Temperature'] != 999.0, 'Temperature'].max()} C")
    print(f"  Humidity range: {df['Humidity'].min()}–{df['Humidity'].max()} %")
    print(f"  Sensor faults : {failures} rows ({failures/len(df)*100:.2f}%)")


if __name__ == "__main__":
    main()