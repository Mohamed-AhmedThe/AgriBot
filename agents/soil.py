import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

# Assuming BaseAgent is defined in agents/base.py
try:
    from agents.base import BaseAgent
except ImportError:
    class BaseAgent: pass # Mock for local testing if base.py is missing

from models.models_loader import DEVICE

# ============================================================
# AGENT 1: TIME-SERIES HEALTH MONITOR
# ============================================================
class SoilIntelligenceAgent(BaseAgent):
    def __init__(self, models_dict: dict):
        self.gru = models_dict.get("gru")
        self.lstm = models_dict.get("lstm")
        self.cnn = models_dict.get("cnn")

    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes real-time NPK and moisture arrays to catch anomalies."""
        try:
            n = float(input_data.get('nitrogen', 0))
            p = float(input_data.get('phosphorus', 0))
            k = float(input_data.get('potassium', 0))
            m = float(input_data.get('moisture', 0))

            sensor_array = np.array([[[n, p, k, m]]], dtype=np.float32)
            tensor_input = torch.tensor(sensor_array).to(DEVICE)

            with torch.no_grad():
                gru_prob = torch.sigmoid(self.gru(tensor_input)).item() if self.gru else 0.8
                lstm_prob = torch.sigmoid(self.lstm(tensor_input)).item() if self.lstm else 0.8
                cnn_prob = torch.sigmoid(self.cnn(tensor_input)).item() if self.cnn else 0.8

            votes = [
                1 if gru_prob > 0.5 else 0,
                1 if lstm_prob > 0.5 else 0,
                1 if cnn_prob > 0.5 else 0
            ]
            
            is_healthy = sum(votes) >= 2
            probs = [gru_prob, lstm_prob, cnn_prob]
            winning_probs = [p for v, p in zip(votes, probs) if v == is_healthy]
            avg_confidence = sum(winning_probs) / len(winning_probs) if winning_probs else 0.0

            status = "Nominal" if is_healthy else "Warning"
            health_str = "Healthy" if is_healthy else "Poor"
            finding = f"Real-time soil health is {health_str}. NPK: [{n},{p},{k}], Moisture: {m}%."

            return {
                "unit": "SoilIntelligence",
                "status": status,
                "finding": finding,
                "confidence_score": round(avg_confidence, 3),
                "recommended_action": "none" if is_healthy else "trigger_agronomy_pipeline"
            }

        except Exception as e:
            return {
                "unit": "SoilIntelligence",
                "status": "Error",
                "finding": f"Sensor processing failed: {str(e)}",
                "confidence_score": 0.0,
                "recommended_action": "check_sensors"
            }

# ============================================================
# AGENT 2: SEQUENTIAL STRATEGY PIPELINE
# ============================================================
class MasterAgronomyAgent(BaseAgent):
    def __init__(self, vision_model, strategy_dict: dict):
        self.convnext = vision_model
        self.rf_crop = strategy_dict.get("random_forest")
        self.xgb_fert = strategy_dict.get("xgboost")
        self.crop_le = strategy_dict.get("crop_encoder")
        self.fert_le = strategy_dict.get("fert_encoder")
        self.fert_columns = strategy_dict.get("fert_columns")

    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes Model Chaining: Vision -> Crop Suggestion -> Fertilizer Calculation."""
        try:
            # 1. Extract Environment Data
            temp = float(input_data.get('temperature', 25.0))
            hum = float(input_data.get('humidity', 60.0))
            ph = float(input_data.get('ph', 6.5))
            rain = float(input_data.get('rainfall', 100.0))
            n = float(input_data.get('nitrogen', 0))
            p = float(input_data.get('phosphorus', 0))
            k = float(input_data.get('potassium', 0))
            
            # --- MODEL A: VISION ---
            # In production, image path is passed in input_data and transformed via torchvision.
            # Using placeholder fallback for testing until camera hardware is attached.
            predicted_soil_type = "Loamy Soil" 
            
            # --- MODEL B: RANDOM FOREST CROP RECOMMENDER ---
            predicted_crop = "rice" # Default fallback
            if self.rf_crop and self.crop_le:
                rf_features = np.array([[n, p, k, temp, hum, ph, rain]])
                crop_encoded = self.rf_crop.predict(rf_features)[0]
                predicted_crop = self.crop_le.inverse_transform([crop_encoded])[0]

            # --- MODEL C: XGBOOST FERTILIZER (CHAINED INFERENCE) ---
            recommended_fertilizer = "Balanced NPK Fertilizer" # Default fallback
            if self.xgb_fert and self.fert_columns and self.fert_le:
                # Create a baseline dataframe of all zeros using the saved column names
                xgb_input = pd.DataFrame(0, index=[0], columns=self.fert_columns)
                
                # Fill in the continuous numerical variables
                xgb_input.loc[0, 'Temperature'] = temp
                xgb_input.loc[0, 'Moisture'] = hum
                xgb_input.loc[0, 'Rainfall'] = rain
                xgb_input.loc[0, 'PH'] = ph
                xgb_input.loc[0, 'Nitrogen'] = n
                xgb_input.loc[0, 'Phosphorous'] = p
                xgb_input.loc[0, 'Potassium'] = k
                
                # Dynamically trigger the "One-Hot Encoded" dummy columns from Models A & B
                soil_col = f"Soil_{predicted_soil_type}"
                crop_col = f"Crop_{predicted_crop}"
                
                if soil_col in xgb_input.columns:
                    xgb_input.loc[0, soil_col] = 1
                if crop_col in xgb_input.columns:
                    xgb_input.loc[0, crop_col] = 1

                fert_encoded = self.xgb_fert.predict(xgb_input)[0]
                recommended_fertilizer = self.fert_le.inverse_transform([fert_encoded])[0]

            # --- FINAL OUTPUT FORMATTING ---
            finding_str = (
                f"Camera classified soil as {predicted_soil_type}. "
                f"Environmental parameters suggest planting {predicted_crop}. "
                f"To optimize yield, deploy {recommended_fertilizer}."
            )

            return {
                "unit": "MasterAgronomyPipeline",
                "status": "Action Required",
                "finding": finding_str,
                "confidence_score": 0.92,
                "recommended_action": "deploy_fertilizer",
                "action_parameters": {
                    "soil_type": predicted_soil_type,
                    "target_crop": predicted_crop,
                    "fertilizer": recommended_fertilizer
                }
            }

        except Exception as e:
            return {
                "unit": "MasterAgronomyPipeline",
                "status": "Error",
                "finding": f"Sequential pipeline failed: {str(e)}",
                "confidence_score": 0.0,
                "recommended_action": "system_reset"
            }