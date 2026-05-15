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
    gru_model = SoilGRUClassifier().to(DEVICE)
    lstm_model = SoilLSTMClassifier().to(DEVICE)
    cnn_model = Soil1DCNNClassifier().to(DEVICE)
    
    try:
        gru_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "soil_gru.pth"), map_location=DEVICE))
        lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "soil_lstm.pth"), map_location=DEVICE))
        cnn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "soil_cnn.pth"), map_location=DEVICE))
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
    # 4 classes for Soil types (e.g., Alluvial, Black, Clay, Red)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
    
    try:
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "convnext_soil.pth"), map_location=DEVICE))
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
        rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_crop_recommender.pkl"))
        
        # Load encoders and feature maps to translate words to numbers and back
        crop_le = joblib.load(os.path.join(MODELS_DIR, "crop_label_encoder.pkl"))
        fert_le = joblib.load(os.path.join(MODELS_DIR, "fert_label_encoder.pkl"))
        fert_cols = joblib.load(os.path.join(MODELS_DIR, "fert_feature_columns.pkl"))
    except FileNotFoundError as e:
        print(f"[Warning] Strategy file missing: {e}. Fallback logic will be used.")
        xgb_model = None

    return {
        "xgboost": xgb_model, 
        "random_forest": rf_model,
        "crop_encoder": crop_le,
        "fert_encoder": fert_le,
        "fert_columns": fert_cols
    }


# ============================================================
# TEAM DELTA: CROP PATHOLOGY VISION MODELS
# ============================================================
def load_pathology_vision_models():
    """Loads DenseNet121 for Team Delta's Vision Node."""
    print(f"[Models] Loading Crop Pathology Vision Models onto {DEVICE}...")
    
    densenet_model = None
    
    # Initialize DenseNet121 (4 output classes for the diseases)
    try:
        densenet_model = vision_models.densenet121(weights=None)
        num_ftrs = densenet_model.classifier.in_features
        densenet_model.classifier = nn.Linear(num_ftrs, 4)
        densenet_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "densenet_pathology.pth"), map_location=DEVICE))
        densenet_model.eval()
        print("  ✓ Loaded densenet_pathology.pth")
    except FileNotFoundError:
        print("  [Warning] densenet_pathology.pth not found. Vision scans will use fallback.")

    return {
        "densenet": densenet_model
    }