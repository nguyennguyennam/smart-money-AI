from enum import Enum


class Category(str, Enum):
    FOOD = "FOOD"
    TRANSPORTATION = "TRANSPORTATION"
    CLOTHING = "CLOTHING"
    UTILITIES = "UTILITIES"
    ENTERTAINMENT = "ENTERTAINMENT"
    HEALTH = "HEALTH"
    EDUCATION = "EDUCATION"
    OTHER = "OTHER"


# Backward-compatible constant list of string category names
CATEGORIES = [c.value for c in Category]
