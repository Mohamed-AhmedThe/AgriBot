import os
import torch
import torch.nn as nn
import torchvision.models as vision_models
import xgboost as xgb
import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "models/weights"

# ============================================================
# TEAM BETA: 1. SOIL HEALTH ENSEMBLE (TIME-SERIES)
# ============================================================
class SoilGRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=32, batch_first=True)
        self.fc  = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class SoilLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
        self.fc   = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class Soil1DCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=1)
        self.fc   = nn.Linear(16, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.relu(self.conv(x))
        out = out.mean(dim=2)
        return self.fc(out).squeeze(-1)

def load_soil_models():
    print(f"[Models] Loading Soil Health Ensemble onto {DEVICE}...")
    gru_model  = SoilGRUClassifier().to(DEVICE)
    lstm_model = SoilLSTMClassifier().to(DEVICE)
    cnn_model  = Soil1DCNNClassifier().to(DEVICE)

    try:
        gru_model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "soil_gru.pth"), map_location=DEVICE))
        lstm_model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "soil_lstm.pth"), map_location=DEVICE))
        cnn_model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "soil_cnn.pth"), map_location=DEVICE))
    except FileNotFoundError:
        print("[Warning] Time-Series Soil weights missing. Using untrained initialization.")

    gru_model.eval()
    lstm_model.eval()
    cnn_model.eval()

    return {"gru": gru_model, "lstm": lstm_model, "cnn": cnn_model}


# ============================================================
# TEAM BETA: 2. STRATEGY & VISION PIPELINE (SEQUENTIAL)
# ============================================================
def load_soil_vision_model():
    print(f"[Models] Loading ConvNeXt-Tiny onto {DEVICE}...")
    model = vision_models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)

    try:
        model.load_state_dict(torch.load(
            os.path.join(MODELS_DIR, "convnext_soil.pth"), map_location=DEVICE))
    except FileNotFoundError:
        print("[Warning] convnext_soil.pth not found. Using untrained initialization.")

    model.eval()
    return model

def load_agronomy_strategy_models():
    print("[Models] Loading Agronomy Strategy Models and Encoders...")
    xgb_model = xgb.XGBClassifier()
    rf_model, crop_le, fert_le, fert_cols = None, None, None, None

    try:
        xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_fertilizer.json"))
        rf_model  = joblib.load(os.path.join(MODELS_DIR, "rf_crop_recommender.pkl"))
        crop_le   = joblib.load(os.path.join(MODELS_DIR, "crop_label_encoder.pkl"))
        fert_le   = joblib.load(os.path.join(MODELS_DIR, "fert_label_encoder.pkl"))
        fert_cols = joblib.load(os.path.join(MODELS_DIR, "fert_feature_columns.pkl"))
    except FileNotFoundError as e:
        print(f"[Warning] Strategy file missing: {e}. Fallback logic will be used.")
        xgb_model = None

    return {
        "xgboost":       xgb_model,
        "random_forest": rf_model,
        "crop_encoder":  crop_le,
        "fert_encoder":  fert_le,
        "fert_columns":  fert_cols,
    }


# ============================================================
# TEAM GAMMA: WEATHER ENSEMBLE (TIME-SERIES)
# ============================================================
# Architectures verified against saved state_dict keys/shapes.
#
# GRU / LSTM — single-layer regressors
#   Input  : (batch, seq_len=20, features=2)
#   Output : (batch, 2)  →  [next_temp, next_humidity]
#   Keys   : {gru|lstm}.weight_ih_l0 [192|256, 2]
#             {gru|lstm}.weight_hh_l0 [192|256, 64]
#             fc.weight [2, 64]
#
# CNN — 1-layer conv regressor (NOT a classifier)
#   Input  : (batch, channels=2, seq_len=20)  ← transposed before forward()
#   Output : (batch, 2)  →  [next_temp, next_humidity]
#   Keys   : conv1.weight [64, 2, 3]
#             fc1.weight  [32, 576]   (576 = 64 * 9, padding=1 on seq_len=20)
#             fc2.weight  [2, 32]
#
# All three are regressors. Anomaly detection in MicroClimateAgent uses
# inter-model disagreement + deterministic threshold checks, not a
# sigmoid classifier output.

class WeatherGRURegressor(nn.Module):
    """
    Single-layer GRU regressor.
    hidden_size=64, num_layers=1 (verified from weight_hh_l0 shape [192,64]).
    """
    def __init__(self, input_size: int = 2, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class WeatherLSTMRegressor(nn.Module):
    """
    Single-layer LSTM regressor.
    hidden_size=64, num_layers=1 (verified from weight_hh_l0 shape [256,64]).
    """
    def __init__(self, input_size: int = 2, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class Weather1DCNNRegressor(nn.Module):
    """
    1D-CNN regressor — layer names match saved state dict exactly.

    Verified architecture (reverse-engineered from fc1.weight shape [32, 576]):
      conv1      : Conv1d(2, 64, kernel_size=3, padding=1)  → (batch, 64, 20)
      MaxPool1d  : kernel_size=3, stride=2                  → (batch, 64,  9)
      flatten    :                                          → (batch, 576)
      fc1        : Linear(576, 32)
      fc2        : Linear(32, 2)

    Input  : (batch, channels=2, seq_len=20)  — MicroClimateAgent transposes before calling
    Output : (batch, 2)  →  [next_temp, next_humidity]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=3, stride=2)
        self.fc1   = nn.Linear(576, 32)
        self.fc2   = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, 2, 20)
        x = torch.relu(self.conv1(x))   # → (batch, 64, 20)
        x = self.pool(x)                # → (batch, 64,  9)
        x = x.flatten(1)               # → (batch, 576)
        x = torch.relu(self.fc1(x))    # → (batch,  32)
        return self.fc2(x)             # → (batch,   2)


def load_weather_models() -> dict:
    """
    Loads the three weather models + MinMaxScaler from models/weights/.

    Expected files
    --------------
    weather_gru.pth      — WeatherGRURegressor state dict
    weather_lstm.pth     — WeatherLSTMRegressor state dict
    weather_cnn.pth      — Weather1DCNNClassifier state dict
    weather_scaler.pkl   — sklearn MinMaxScaler fitted on Temperature + Humidity

    Returns
    -------
    dict with keys: "gru", "lstm", "cnn", "scaler"
    All nn.Modules are in eval() mode on DEVICE.
    "scaler" is None if the pkl is missing (triggers fallback in MicroClimateAgent).
    """
    print(f"[Models] Loading Weather Ensemble onto {DEVICE}...")

    gru_model  = WeatherGRURegressor().to(DEVICE)
    lstm_model = WeatherLSTMRegressor().to(DEVICE)
    cnn_model  = Weather1DCNNRegressor().to(DEVICE)
    scaler     = None

    weight_files = {
        "gru":  ("weather_gru.pth",  gru_model),
        "lstm": ("weather_lstm.pth", lstm_model),
        "cnn":  ("weather_cnn.pth",  cnn_model),
    }

    for key, (filename, model) in weight_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"  ✓ Loaded {filename}")
        else:
            print(f"  [Warning] {filename} not found — using untrained {key.upper()}.")

    gru_model.eval()
    lstm_model.eval()
    cnn_model.eval()

    scaler_path = os.path.join(MODELS_DIR, "weather_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("  ✓ Loaded weather_scaler.pkl")
    else:
        print("  [Warning] weather_scaler.pkl not found — raw inputs will be used (reduced accuracy).")
        scaler = _build_fallback_scaler()

    return {
        "gru":    gru_model,
        "lstm":   lstm_model,
        "cnn":    cnn_model,
        "scaler": scaler,
    }


def _build_fallback_scaler():
    """
    Returns a MinMaxScaler pre-fitted on the synthetic data's known ranges
    (Temperature: 20–42°C, Humidity: 25–95%).
    Used only when weather_scaler.pkl is missing — prevents a hard crash.
    """
    try:
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        scaler = MinMaxScaler()
        # Fit on the exact min/max from TelemetryConfig
        scaler.fit(np.array([[20.0, 25.0], [42.0, 95.0]]))
        print("  [Fallback] Built MinMaxScaler from hardcoded synthetic data ranges.")
        return scaler
    except ImportError:
        return None