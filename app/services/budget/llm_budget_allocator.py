"""LLM-based budget allocation.

Asks gpt-5-nano to split a monthly budget across the fixed spending
categories, then reuses BudgetPredictor's deterministic math (normalize →
round to 50k → fix sum) so the returned amounts always sum exactly to the
total budget.

Input shape (fetched by the AI budget pipeline):
    {
        "user_id": str,
        "month": "YYYY-MM",
        "financialSetup": {
            "safe_spending": float,          # total monthly budget (VND)
            "savingPace": str,               # e.g. BALANCED
            "interventionLevel": str,        # e.g. GENTLE
            "focusMode": str,                # SAVE_MODE | REDUCE_SPENDING | TRACK_ONLY
        },
        "spendingHistory": {                 # may be null / empty
            "monthsObserved": int,
            "lastMonthTotalSpend": float,
            "averageMonthlySpend": float,
            "categories": {
                "FOOD": {"lastMonthRatio": float, "averageRatio": float},
                ...
            },
        },
    }

If the LLM call fails or returns unusable output, callers fall back to a
deterministic allocation derived from the spending-history average ratios
(or an equal split when no history is available).

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
FALLBACK_MODEL_VERSION = "HISTORY_RATIO_FALLBACK"

# Cold-start prior used when there is no spending history and the LLM call
# fails. Weights essentials over discretionary; sums to 1.0 across the 9
# categories in CATEGORY_COLS. Runs through the same normalize → round →
# fix-sum math as every other path.
_BASE_PRIOR: dict[str, float] = {
    "FOOD": 0.22,
    "UTILITIES": 0.20,
    "TRANSPORTATION": 0.10,
    "EDUCATION": 0.10,
    "HEALTH": 0.08,
    "ENTERTAINMENT": 0.08,
    "SHOPPING": 0.08,
    "OTHER": 0.08,
    "CLOTHING": 0.06,
}

# Discretionary categories the cold-start prior trims/boosts by focusMode.
_DISCRETIONARY_CATEGORIES = ("ENTERTAINMENT", "SHOPPING", "CLOTHING")
# Category that absorbs whatever is freed (or borrowed) by the adjustment;
# acts as the savings/repayment bucket.
_SLACK_CATEGORY = "OTHER"

# Multiplier applied to each discretionary category per focusMode. < 1 trims
# discretionary and pushes the slack into _SLACK_CATEGORY (more saving).
# Unknown modes fall through to the neutral base prior (factor 1.0) via
# dict.get default.
#   SAVE_MODE       - prioritize saving: gently trim discretionary.
#   REDUCE_SPENDING - aggressively cut discretionary spend.
#   TRACK_ONLY      - no intervention: keep the neutral base prior.
# SAVE_MORE is aliased to SAVE_MODE because the sample payload used that
# spelling; confirm which the backend actually emits.
_FOCUS_DISCRETIONARY_SCALE: dict[str, float] = {
    "SAVE_MODE": 0.6,
    "SAVE_MORE": 0.6,
    "REDUCE_SPENDING": 0.4,
    "TRACK_ONLY": 1.0,
}


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
    financial_setup: dict[str, Any],
    spending_history: dict[str, Any] | None,
    month: str | None,
) -> str:
    categories = ", ".join(CATEGORY_COLS)
    setup_json = json.dumps(financial_setup, ensure_ascii=False)
    history_json = (
        json.dumps(spending_history, ensure_ascii=False)
        if spending_history
        else "null"
    )
    month_label = month or "tháng này"

    return (
        "Bạn là cố vấn tài chính cá nhân của SmartMoney.\n"
        f"Nhiệm vụ: phân bổ ngân sách chi tiêu cho {month_label} theo từng danh mục.\n\n"
        f"Tổng ngân sách an toàn (safe_spending): {total_budget} {currency}\n"
        f"Thiết lập tài chính: {setup_json}\n"
        "  - savingPace: nhịp độ tiết kiệm mong muốn.\n"
        "  - interventionLevel: mức độ can thiệp/điều chỉnh so với thói quen cũ.\n"
        "  - focusMode: mục tiêu trọng tâm "
        "(SAVE_MODE = ưu tiên tiết kiệm, REDUCE_SPENDING = cắt giảm chi tiêu, "
        "TRACK_ONLY = chỉ theo dõi, giữ nguyên thói quen).\n"
        f"Lịch sử chi tiêu (có thể null): {history_json}\n"
        "  - categories[*].lastMonthRatio: tỷ lệ chi của tháng trước.\n"
        "  - categories[*].averageRatio: tỷ lệ chi trung bình nhiều tháng.\n\n"
        f"Danh mục cho phép: {categories}\n\n"
        "Quy tắc:\n"
        "- Trả về DUY NHẤT một đối tượng JSON, khóa là tên danh mục, giá trị là TỶ LỆ (số thực 0..1).\n"
        "- Phải có ĐỦ tất cả danh mục ở trên; tổng các tỷ lệ xấp xỉ 1.0.\n"
        "- Lấy averageRatio làm nền, rồi điều chỉnh theo focusMode, savingPace và interventionLevel.\n"
        "- SAVE_MODE hoặc REDUCE_SPENDING: giảm các danh mục không thiết yếu "
        "(ENTERTAINMENT, SHOPPING, CLOTHING) và dồn phần dư vào tiết kiệm/OTHER; "
        "REDUCE_SPENDING cắt mạnh hơn SAVE_MODE.\n"
        "- TRACK_ONLY: giữ nguyên thói quen, bám sát averageRatio, không cắt giảm.\n"
        "- interventionLevel càng mạnh thì càng được điều chỉnh xa thói quen cũ; GENTLE thì bám sát lịch sử.\n"
        "- Nếu không có lịch sử, dựa vào thiết lập tài chính để phân bổ hợp lý.\n"
        "- Không thêm chú thích hay markdown.\n\n"
        "Ví dụ định dạng:\n"
        '{"FOOD": 0.3, "TRANSPORTATION": 0.1, "CLOTHING": 0.05, "UTILITIES": 0.15, '
        '"ENTERTAINMENT": 0.1, "HEALTH": 0.05, "EDUCATION": 0.1, "SHOPPING": 0.1, "OTHER": 0.05}'
    )


class LLMBudgetAllocator:
    """LLM-first budget allocator with a deterministic history fallback."""

    def __init__(
        self,
        llm_service: LLMService,
        fallback_predictor: BudgetPredictor,
    ) -> None:
        self._llm = llm_service
        # BudgetPredictor is used only for its deterministic money math
        # helpers (normalize_ratios / round_money / fix_total_sum).
        self._fallback = fallback_predictor

    async def allocate(
        self,
        financial_setup: dict[str, Any],
        spending_history: dict[str, Any] | None = None,
        *,
        currency: str = "VND",
        month: str | None = None,
    ) -> dict[str, Any]:
        
        if not isinstance(financial_setup, dict) or not financial_setup:
            raise ValueError("financial_setup must be a non-empty dictionary")

        total_budget = self._resolve_total_budget(financial_setup)

        try:
            ratios = await self._llm_ratios(
                total_budget,
                currency,
                financial_setup,
                spending_history,
                month,
            )
            return self._finalize(total_budget, ratios, MODEL_VERSION)
        except Exception as e:
            logger.warning(
                "LLM budget allocation failed (%s); "
                "falling back to spending-history ratios",
                e,
            )
            return self._fallback_allocation(
                total_budget,
                financial_setup,
                spending_history,
            )

    def _resolve_total_budget(self, financial_setup: dict[str, Any]) -> int:
        raw = financial_setup.get("safe_spending")
        try:
            total_budget = int(round(float(raw)))
        except (TypeError, ValueError):
            raise ValueError(
                f"financial_setup.safe_spending must be numeric, got {raw!r}"
            )
        if total_budget <= 0:
            raise ValueError("financial_setup.safe_spending must be greater than 0")
        return total_budget

    async def _llm_ratios(
        self,
        total_budget: int,
        currency: str,
        financial_setup: dict[str, Any],
        spending_history: dict[str, Any] | None,
        month: str | None,
    ) -> dict[str, float]:
        prompt = _build_prompt(
            total_budget,
            currency,
            financial_setup,
            spending_history,
            month,
        )

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

    def _history_ratios(
        self,
        spending_history: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Pull per-category ratios from spendingHistory for the fallback.

        Prefers ``averageRatio`` (multi-month average) and falls back to
        ``lastMonthRatio`` when the average is missing.
        """
        ratios: dict[str, float] = {}
        if not isinstance(spending_history, dict):
            return ratios

        categories = spending_history.get("categories")
        if not isinstance(categories, dict):
            return ratios

        for category in CATEGORY_COLS:
            entry = categories.get(category)
            if not isinstance(entry, dict):
                continue
            value = entry.get("averageRatio")
            if value is None:
                value = entry.get("lastMonthRatio")
            try:
                ratios[category] = max(float(value), 0.0)
            except (TypeError, ValueError):
                continue

        return ratios

    def _cold_start_prior(
        self,
        financial_setup: dict[str, Any],
    ) -> dict[str, float]:
        """Focus-mode-aware default used when there is no spending history.

        Starts from the essentials-weighted base prior, then scales the
        discretionary categories by focusMode and moves whatever is freed
        (or borrowed) into the savings/repayment bucket. Unrecognized
        focusMode values fall through to the neutral base prior.
        """
        focus_mode = str(financial_setup.get("focusMode") or "").strip().upper()
        scale = _FOCUS_DISCRETIONARY_SCALE.get(focus_mode, 1.0)

        prior = dict(_BASE_PRIOR)
        if scale != 1.0:
            freed = 0.0
            for category in _DISCRETIONARY_CATEGORIES:
                adjusted = prior[category] * scale
                freed += prior[category] - adjusted
                prior[category] = adjusted
            # freed > 0 when trimming (goes to savings); < 0 when boosting
            # (borrowed from savings). Clamp so the bucket never goes negative;
            # normalize_ratios rescales the whole vector to sum to 1 afterwards.
            prior[_SLACK_CATEGORY] = max(prior[_SLACK_CATEGORY] + freed, 0.0)

        return prior

    def _fallback_allocation(
        self,
        total_budget: int,
        financial_setup: dict[str, Any],
        spending_history: dict[str, Any] | None,
    ) -> dict[str, Any]:
        ratios = self._history_ratios(spending_history)

        if not ratios or sum(ratios.values()) <= 0:
            # No usable history: use the focusMode-aware cold-start prior.
            ratios = self._cold_start_prior(financial_setup)

        return self._finalize(total_budget, ratios, FALLBACK_MODEL_VERSION)

    def _finalize(
        self,
        total_budget: int,
        ratios_map: dict[str, float],
        model_version: str,
    ) -> dict[str, Any]:
        # Reuse the predictor's math so LLM and fallback paths behave identically.
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
            "modelVersion": model_version,
            "totalBudget": total_budget,
            "categories": categories,
        }
