import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CATEGORY_COLS = [
    "FOOD",
    "TRANSPORT",
    "SHOPPING",
    "ENTERTAINMENT",
    "UTILITIES",
    "HEALTH",
    "EDUCATION",
    "OTHER",
]

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "budget" / "profile_budget_model.cbm"
METADATA_PATH = BASE_DIR / "models" / "budget" / "profile_budget_metadata.json"

class BudgetPredictor:
    def __init__(self):
        self.model = CatBoostRegressor()
        self.model.load_model(str(MODEL_PATH))

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def validate_profile(self, profile: dict):
        for col in self.metadata["categorical_cols"]:
            value = profile.get(col)
            allowed_values = self.metadata["allowed_values"].get(col, [])

            if value not in allowed_values:
                raise ValueError(
                    f"Invalid value for {col}: {value}. "
                    f"Allowed values are: {allowed_values}"
                )
            
    def normalize_ratios(self, ratios):
        ratios = np.clip(np.array(ratios, dtype=float), 0, None)

        total = ratios.sum()

        if total == 0:
            return np.ones(len(ratios)) / len(ratios)
        
        return ratios / total
    
    def round_money(self, amount: float, base: int = 50_000) -> int:
        return int(round(amount / base) * base)
    
    def fix_total_sum(self, amounts: dict, total_budget: int):
        current_sum = sum(amounts.values())
        delta = total_budget - current_sum

        amounts["OTHER"] += delta

        return amounts
    
    def predict(self, total_budget: int, profile: dict):
        self.validate_profile(profile)

        row = {
            "total_budget": total_budget,
            **profile,
        }

        X = pd.DataFrame([row])
        X = X[self.metadata["feature_cols"]]

        preds = self.model.predict(X)
        ratios = self.normalize_ratios(preds[0])

        amounts = {}

        for i, category in enumerate(CATEGORY_COLS):
            amounts[category] = self.round_money(ratios[i] * total_budget)

        amounts = self.fix_total_sum(amounts, total_budget)

        categories = []

        for category in CATEGORY_COLS:
            categories.append({
                "category": category,
                "ratio": round(amounts[category] /total_budget, 4),
                "amount": int(amounts[category]),
            })
        
        return {
            "modelVersion": "PROFILE_BASED_CATBOOST",
            "totalBudget": total_budget,
            "categories": categories,
        }
