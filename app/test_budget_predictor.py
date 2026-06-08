from app.services.budget.budget_predictor import BudgetPredictor


def main():
    predictor = BudgetPredictor()

    profile = {
        "role": "student",
        "living_status": "rent_room",
        "income_level": "medium",
        "transport_mode": "motorbike",
        "spending_style": "balanced",
        "work_style": "part_time",
        "family_status": "single",
        "study_intensity": "course_heavy",
        "health_need": "normal",
    }

    result = predictor.predict(
        total_budget=6_000_000,
        profile=profile,
    )

    print(result)

    total = sum(item["amount"] for item in result["categories"])
    print("Total amount:", total)
    print("Expected:", result["totalBudget"])


if __name__ == "__main__":
    main()