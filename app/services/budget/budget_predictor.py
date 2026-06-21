from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


CATEGORY_COLS = [
    "FOOD",
    "TRANSPORTATION",
    "CLOTHING",
    "UTILITIES",
    "ENTERTAINMENT",
    "HEALTH",
    "EDUCATION",
    "SHOPPING",
    "OTHER",
]

BASE_DIR = Path(__file__).resolve().parents[2]

PROFILE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "budget"
    / "profile_budget_model.cbm"
)

PROFILE_METADATA_PATH = (
    BASE_DIR
    / "models"
    / "budget"
    / "profile_budget_metadata.json"
)

HISTORY_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "budget"
    / "history_budget_model.cbm"
)

HISTORY_METADATA_PATH = (
    BASE_DIR
    / "models"
    / "budget"
    / "history_budget_metadata.json"
)


class BudgetPredictor:

    def __init__(self) -> None:
        (
            self.profile_model,
            self.profile_metadata,
        ) = self._load_artifacts(
            PROFILE_MODEL_PATH,
            PROFILE_METADATA_PATH,
        )

        (
            self.history_model,
            self.history_metadata,
        ) = self._load_artifacts(
            HISTORY_MODEL_PATH,
            HISTORY_METADATA_PATH,
        )

        self.profile_category_cols = (
            self.profile_metadata.get(
                "category_cols",
                CATEGORY_COLS,
            )
        )

        self.history_category_cols = (
            self.history_metadata.get(
                "category_cols",
                CATEGORY_COLS,
            )
        )

        self._validate_category_schema()

    def _load_artifacts(
        self,
        model_path: Path,
        metadata_path: Path,
    ) -> tuple[CatBoostRegressor, dict[str, Any]]:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Budget model not found: {model_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Budget metadata not found: {metadata_path}"
            )

        model = CatBoostRegressor()
        model.load_model(str(model_path))

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        if "feature_cols" not in metadata:
            raise ValueError(
                f"Missing feature_cols in metadata: "
                f"{metadata_path}"
            )

        return model, metadata

    def _validate_category_schema(self) -> None:
        if self.profile_category_cols != CATEGORY_COLS:
            raise ValueError(
                "Profile model category schema does not match "
                f"application schema. "
                f"Expected={CATEGORY_COLS}, "
                f"actual={self.profile_category_cols}"
            )

        if self.history_category_cols != CATEGORY_COLS:
            raise ValueError(
                "History model category schema does not match "
                f"application schema. "
                f"Expected={CATEGORY_COLS}, "
                f"actual={self.history_category_cols}"
            )

    def validate_profile(
        self,
        profile: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        categorical_cols = metadata.get(
            "categorical_cols",
            [],
        )

        allowed_values_map = metadata.get(
            "allowed_values",
            {},
        )

        for column in categorical_cols:
            value = profile.get(column)

            allowed_values = allowed_values_map.get(
                column,
                [],
            )

            if value is None:
                raise ValueError(
                    f"Missing required profile field: {column}"
                )

            if allowed_values and value not in allowed_values:
                raise ValueError(
                    f"Invalid value for {column}: {value}. "
                    f"Allowed values are: {allowed_values}"
                )

    def _has_usable_history(
        self,
        history_features: dict[str, Any] | None,
    ) -> bool:
        if not isinstance(history_features, dict):
            return False

        expected_columns = [
            f"{category}_avg_3m_ratio"
            for category in CATEGORY_COLS
        ]

        total_ratio = 0.0
        found_feature = False

        for column in expected_columns:
            raw_value = history_features.get(column)

            if raw_value is None:
                continue

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(value):
                continue

            found_feature = True
            total_ratio += max(value, 0.0)

        return found_feature and total_ratio > 0

    def _build_feature_frame(
        self,
        total_budget: int,
        profile: dict[str, Any],
        metadata: dict[str, Any],
        history_features: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        row: dict[str, Any] = {
            "total_budget": total_budget,
            **profile,
        }

        if history_features:
            row.update(history_features)

        feature_cols = metadata["feature_cols"]

        # /*
        #  * Lịch sử có thể thiếu một category nếu dữ liệu cũ
        #  * chưa gửi đầy đủ. Khi đó dùng tỷ lệ 0.
        #  */
        for column in feature_cols:
            if (
                column.endswith("_avg_3m_ratio")
                and column not in row
            ):
                row[column] = 0.0

        missing_features = [
            column
            for column in feature_cols
            if column not in row
        ]

        if missing_features:
            raise ValueError(
                "Missing budget model features: "
                f"{missing_features}"
            )

        frame = pd.DataFrame([row])

        return frame[feature_cols]

    def normalize_ratios(
        self,
        ratios: Any,
    ) -> np.ndarray:
        normalized = np.clip(
            np.asarray(ratios, dtype=float),
            0,
            None,
        )

        total = normalized.sum()

        if total <= 0:
            return (
                np.ones(len(normalized), dtype=float)
                / len(normalized)
            )

        return normalized / total

    def round_money(
        self,
        amount: float,
        base: int = 50_000,
    ) -> int:
        return int(
            round(amount / base) * base
        )

    def fix_total_sum(
        self,
        amounts: dict[str, int],
        total_budget: int,
        category_cols: list[str],
    ) -> dict[str, int]:
        current_sum = sum(amounts.values())
        delta = total_budget - current_sum

        if delta == 0:
            return amounts

        # /*
        #  * Nếu còn thiếu sau làm tròn, cộng vào OTHER.
        #  * Nếu đang dư, trừ khỏi category có số tiền lớn nhất
        #  * để tránh OTHER bị âm.
        #  */
        if delta > 0:
            correction_category = (
                "OTHER"
                if "OTHER" in amounts
                else category_cols[-1]
            )
        else:
            correction_category = max(
                amounts,
                key=amounts.get,
            )

        amounts[correction_category] += delta

        if amounts[correction_category] < 0:
            raise ValueError(
                "Cannot correct rounded budget amounts "
                "without producing a negative category amount"
            )

        return amounts

    def _predict_with_model(
        self,
        *,
        model: CatBoostRegressor,
        metadata: dict[str, Any],
        category_cols: list[str],
        model_version: str,
        total_budget: int,
        profile: dict[str, Any],
        history_features: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.validate_profile(
            profile,
            metadata,
        )

        features = self._build_feature_frame(
            total_budget=total_budget,
            profile=profile,
            metadata=metadata,
            history_features=history_features,
        )

        raw_predictions = model.predict(features)

        ratios = self.normalize_ratios(
            np.asarray(raw_predictions).reshape(-1)
        )

        if len(ratios) != len(category_cols):
            raise ValueError(
                "Budget model output size does not match "
                "category count. "
                f"Output={len(ratios)}, "
                f"categories={len(category_cols)}"
            )

        amounts: dict[str, int] = {}

        for index, category in enumerate(
            category_cols
        ):
            amounts[category] = self.round_money(
                ratios[index] * total_budget
            )

        amounts = self.fix_total_sum(
            amounts=amounts,
            total_budget=total_budget,
            category_cols=category_cols,
        )

        categories = []

        for category in category_cols:
            category_amount = int(amounts[category])

            categories.append(
                {
                    "category": category,
                    "ratio": round(
                        category_amount / total_budget,
                        4,
                    ),
                    "amount": category_amount,
                }
            )

        return {
            "modelVersion": model_version,
            "totalBudget": total_budget,
            "categories": categories,
        }

    def predict(
        self,
        total_budget: int,
        profile: dict[str, Any],
        history_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if total_budget <= 0:
            raise ValueError(
                "total_budget must be greater than 0"
            )

        if not isinstance(profile, dict):
            raise ValueError(
                "profile must be a dictionary"
            )

        # /*
        #  * Có lịch sử hợp lệ:
        #  * dùng model history.
        #  *
        #  * Không có lịch sử:
        #  * chỉ dùng model profile.
        #  */
        if self._has_usable_history(history_features):
            return self._predict_with_model(
                model=self.history_model,
                metadata=self.history_metadata,
                category_cols=self.history_category_cols,
                model_version="HISTORY_BASED_CATBOOST",
                total_budget=total_budget,
                profile=profile,
                history_features=history_features,
            )

        return self._predict_with_model(
            model=self.profile_model,
            metadata=self.profile_metadata,
            category_cols=self.profile_category_cols,
            model_version="PROFILE_BASED_CATBOOST",
            total_budget=total_budget,
            profile=profile,
            history_features=None,
        )
