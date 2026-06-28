"""LLM-based budget allocation.

Asks gpt-5-nano to split a monthly budget across the fixed spending
categories, then reuses BudgetPredictor's deterministic math (normalize →
round to 50k → fix sum) so the returned amounts always sum exactly to the
total budget. If the LLM call fails or returns unusable output, callers fall
back to the CatBoost BudgetPredictor.

Output shape matches BudgetPredictor.predict():
    {"modelVersion": str, "totalBudget": int, "categories": [
        {"category": str, "ratio": float, "amount": int}, ...]}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.budget.budget_predictor import CATEGORY_COLS, BudgetPredictor
from app.services.llm.service import LLMProvider, LLMService

logger = logging.getLogger(__name__)

MODEL_VERSION = "LLM_GPT5_NANO"


def _strip_json_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    s = _strip_json_fence(raw)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def _build_prompt(
    total_budget: int,
    currency: str,
    profile: dict[str, Any],
    history_features: dict[str, Any] | None,
) -> str:
    categories = ", ".join(CATEGORY_COLS)
    profile_json = json.dumps(profile, ensure_ascii=False)
    history_json = (
        json.dumps(history_features, ensure_ascii=False)
        if history_features
        else "null"
    )

    return (
        "Bạn là cố vấn tài chính cá nhân của SmartMoney.\n"
        "Nhiệm vụ: phân bổ ngân sách chi tiêu hàng tháng cho từng danh mục.\n\n"
        f"Tổng ngân sách: {total_budget} {currency}\n"
        f"Hồ sơ người dùng: {profile_json}\n"
        f"Lịch sử chi tiêu (tỷ lệ trung bình 3 tháng, có thể null): {history_json}\n\n"
        f"Danh mục cho phép: {categories}\n\n"
        "Quy tắc:\n"
        "- Trả về DUY NHẤT một đối tượng JSON, khóa là tên danh mục, giá trị là TỶ LỆ (số thực 0..1).\n"
        "- Phải có ĐỦ tất cả danh mục ở trên; tổng các tỷ lệ xấp xỉ 1.0.\n"
        "- Ưu tiên lịch sử chi tiêu nếu có; nếu không, dựa vào hồ sơ người dùng.\n"
        "- Không thêm chú thích hay markdown.\n\n"
        "Ví dụ định dạng:\n"
        '{"FOOD": 0.3, "TRANSPORTATION": 0.1, "CLOTHING": 0.05, "UTILITIES": 0.15, '
        '"ENTERTAINMENT": 0.1, "HEALTH": 0.05, "EDUCATION": 0.1, "SHOPPING": 0.1, "OTHER": 0.05}'
    )


class LLMBudgetAllocator:
    """LLM-first budget allocator with CatBoost fallback."""

    def __init__(
        self,
        llm_service: LLMService,
        fallback_predictor: BudgetPredictor,
    ) -> None:
        self._llm = llm_service
        self._fallback = fallback_predictor

    async def allocate(
        self,
        total_budget: int,
        profile: dict[str, Any],
        history_features: dict[str, Any] | None = None,
        currency: str = "VND",
    ) -> dict[str, Any]:
        if total_budget <= 0:
            raise ValueError("total_budget must be greater than 0")
        if not isinstance(profile, dict) or not profile:
            raise ValueError("profile must be a non-empty dictionary")

        try:
            ratios = await self._llm_ratios(
                total_budget,
                currency,
                profile,
                history_features,
            )
            return self._finalize(total_budget, ratios)
        except Exception as e:
            logger.warning(
                "LLM budget allocation failed (%s); falling back to CatBoost",
                e,
            )
            return self._fallback.predict(
                total_budget=total_budget,
                profile=profile,
                history_features=history_features,
            )

    async def _llm_ratios(
        self,
        total_budget: int,
        currency: str,
        profile: dict[str, Any],
        history_features: dict[str, Any] | None,
    ) -> dict[str, float]:
        prompt = _build_prompt(total_budget, currency, profile, history_features)

        try:
            raw = await self._llm.generate(
                prompt=prompt,
                provider=LLMProvider.OPENAI,
                response_format={"type": "json_object"},
            )
        except TypeError:
            # Older OpenAI SDKs may not accept response_format.
            raw = await self._llm.generate(prompt=prompt, provider=LLMProvider.OPENAI)

        obj = _parse_json_object(raw)

        ratios: dict[str, float] = {}
        for category in CATEGORY_COLS:
            value = obj.get(category)
            try:
                ratios[category] = max(float(value), 0.0)
            except (TypeError, ValueError):
                ratios[category] = 0.0

        if sum(ratios.values()) <= 0:
            raise ValueError(f"LLM returned no usable ratios: {raw!r}")

        return ratios

    def _finalize(
        self,
        total_budget: int,
        ratios_map: dict[str, float],
    ) -> dict[str, Any]:
        # Reuse the predictor's math so LLM and CatBoost paths behave identically.
        ratios = self._fallback.normalize_ratios(
            [ratios_map.get(c, 0.0) for c in CATEGORY_COLS]
        )

        amounts: dict[str, int] = {}
        for index, category in enumerate(CATEGORY_COLS):
            amounts[category] = self._fallback.round_money(
                ratios[index] * total_budget
            )

        amounts = self._fallback.fix_total_sum(
            amounts=amounts,
            total_budget=total_budget,
            category_cols=CATEGORY_COLS,
        )

        categories = []
        for category in CATEGORY_COLS:
            amount = int(amounts[category])
            categories.append(
                {
                    "category": category,
                    "ratio": round(amount / total_budget, 4),
                    "amount": amount,
                }
            )

        return {
            "modelVersion": MODEL_VERSION,
            "totalBudget": total_budget,
            "categories": categories,
        }
