from __future__ import annotations

import pickle
import re
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple

from app.services.classifer.enums import Category


_ALLOWED_VIET_CHARS = "A-Za-zÀ-ỹĐđ"


def clean_text(text: str) -> str:
	"""Clean input text using regex.

	- Removes numbers and special characters
	- Keeps Vietnamese characters (Unicode Latin Extended range)
	- Collapses whitespace
	"""
	if text is None:
		return ""

	text = str(text)
	text = re.sub(fr"[^{_ALLOWED_VIET_CHARS}\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def _sigmoid(x: float) -> float:
	# Numerically stable sigmoid
	if x >= 0:
		z = math.exp(-x)
		return 1.0 / (1.0 + z)
	z = math.exp(x)
	return z / (1.0 + z)


def _softmax(values: list[float]) -> list[float]:
	if not values:
		return []
	m = max(values)
	exps = [math.exp(v - m) for v in values]
	s = sum(exps)
	if s == 0:
		return [0.0 for _ in values]
	return [e / s for e in exps]


@dataclass(frozen=True)
class ClassificationResult:
	category: Category
	confidence: float


class ClassifierService:
	def __init__(self, model_path: Path | None = None):
		if model_path is None:
			model_path = Path(__file__).resolve().parents[2] / "models" / "classifier_model.pkl"

		if not model_path.exists():
			raise FileNotFoundError(f"Classifier model not found at: {model_path}")

		self._model: Any
		with model_path.open("rb") as f:
			self._model = pickle.load(f)

	def classify(self, raw_text: str) -> ClassificationResult:
		cleaned = clean_text(raw_text)
		if not cleaned:
			raise ValueError("Text is empty after cleaning")

		model = self._model

		# model.pkl in this repo is a dict bundle: vectorizer + classifier + label_encoder
		if isinstance(model, dict) and "model" in model and "vectorizer" in model:
			clf = model["model"]
			vectorizer = model["vectorizer"]
			label_encoder = model.get("label_encoder")

			if not hasattr(vectorizer, "transform"):
				raise ValueError("Vectorizer does not support transform()")
			x_vec = vectorizer.transform([cleaned])

			if hasattr(clf, "predict_proba"):
				probs = [float(p) for p in clf.predict_proba(x_vec)[0]]
				best_idx = max(range(len(probs)), key=probs.__getitem__)
				confidence = float(probs[best_idx])
				raw_label = clf.classes_[best_idx] if hasattr(clf, "classes_") else best_idx
				if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
					raw_label = label_encoder.inverse_transform([raw_label])[0]
				category = self._coerce_category(raw_label)
				return ClassificationResult(category=category, confidence=confidence)

			if hasattr(clf, "decision_function") and hasattr(clf, "classes_"):
				scores_any = clf.decision_function(x_vec)
				classes = list(clf.classes_)
				if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
					classes = [label_encoder.inverse_transform([c])[0] for c in classes]

				if len(classes) == 2 and not isinstance(scores_any[0], (list, tuple)):
					s = float(scores_any[0])
					p1 = _sigmoid(s)
					probs = [1.0 - p1, p1]
				else:
					row = scores_any[0]
					s = [float(v) for v in row]
					probs = _softmax(s)

				best_idx = max(range(len(probs)), key=probs.__getitem__)
				confidence = float(probs[best_idx])
				category = self._coerce_category(classes[best_idx])
				return ClassificationResult(category=category, confidence=confidence)

			raise ValueError("Loaded classifier does not support confidence scoring")

		x = [cleaned]

		# Preferred: sklearn-style predict_proba
		if hasattr(model, "predict_proba"):
			probs_any = model.predict_proba(x)[0]
			probs = [float(p) for p in probs_any]
			best_idx = max(range(len(probs)), key=probs.__getitem__)
			confidence = float(probs[best_idx])
			label = model.classes_[best_idx] if hasattr(model, "classes_") else None
			category = self._coerce_category(label)
			return ClassificationResult(category=category, confidence=confidence)

		# Fallback: sklearn-style decision_function -> convert to probabilities
		if hasattr(model, "decision_function") and hasattr(model, "classes_"):
			classes = list(model.classes_)

			scores_any = model.decision_function(x)

			# Binary: scores shape (n_samples,)
			if len(classes) == 2 and not isinstance(scores_any[0], (list, tuple)):
				s = float(scores_any[0])
				p1 = _sigmoid(s)
				probs = [1.0 - p1, p1]
			else:
				row = scores_any[0]
				s = [float(v) for v in row]
				probs = _softmax(s)

			best_idx = max(range(len(probs)), key=probs.__getitem__)
			confidence = float(probs[best_idx])
			category = self._coerce_category(classes[best_idx])
			return ClassificationResult(category=category, confidence=confidence)

		raise ValueError("Loaded model does not support confidence scoring")

	@staticmethod
	def _coerce_category(label: Any) -> Category:
		if isinstance(label, Category):
			return label

		if label is None:
			raise ValueError("Model returned no class label")

		label_str = str(label).strip().upper()
		try:
			return Category(label_str)
		except Exception as e:
			raise ValueError(f"Model predicted unknown category: {label_str}") from e


@lru_cache(maxsize=1)
def get_classifier_service() -> ClassifierService:
	"""Singleton service so the model loads only once per process."""
	return ClassifierService()



