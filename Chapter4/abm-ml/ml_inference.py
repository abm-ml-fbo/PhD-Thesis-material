"""ml_inference.py — inference wrapper for ABM notebooks.

Usage:
from ml_inference import predict_inspection_result
direction = predict_inspection_result(feature_dict)  # -1/0/+1
"""
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from catboost import CatBoostClassifier

MODEL_FILE = 'final_catboost_model.cbm'
META_FILE  = 'final_catboost_model.meta.json'

class _Predictor:
    def __init__(self, model_path=MODEL_FILE, meta_path=META_FILE):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.meta = json.loads(Path(meta_path).read_text())
        self.features = self.meta['feature_order']
        self.target_mapping = {int(k): v for k, v in self.meta['target_mapping'].items()}

    def predict(self, row: Dict[str, Any]) -> int:
        df = pd.DataFrame([[row[k] for k in self.features]], columns=self.features)
        pred = int(self.model.predict(df)[0])  # 0/1/2
        return self.target_mapping.get(pred, 0)  # -> -1/0/+1

_predictor = None

def predict_inspection_result(features: Dict[str, Any]) -> int:
    global _predictor
    if _predictor is None:
        _predictor = _Predictor()
    return _predictor.predict(features)
