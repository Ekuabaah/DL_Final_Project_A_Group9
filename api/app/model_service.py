from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import pickle
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class InferenceResult:
	label: str
	confidence: float
	is_unknown: bool
	top_probabilities: list[dict[str, float | str]]


class ModelService:
	def __init__(
		self,
		checkpoint_path: str,
		img_size: int = 224,
		threshold: float = 0.5,
		top_k: int = 3,
	) -> None:
		self.checkpoint_path = Path(checkpoint_path)
		self.img_size = img_size
		self.threshold = threshold
		self.top_k = top_k
		self.device = torch.device("cpu")

		self.class_names: list[str] = []
		self.model: torch.nn.Module | None = None
		self.eval_transform = transforms.Compose(
			[
				transforms.Resize(256),
				transforms.CenterCrop(self.img_size),
				transforms.ToTensor(),
				transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
			]
		)

	def load(self) -> None:
		if not self.checkpoint_path.exists():
			raise FileNotFoundError(
				f"Checkpoint not found: {self.checkpoint_path.resolve()}"
			)

		ckpt = self._load_checkpoint()
		state_dict, class_names, num_classes = self._parse_checkpoint(ckpt)
		self.model = self._build_and_load_model(state_dict, num_classes)
		self.class_names = class_names

	def _load_checkpoint(self) -> Any:
		# PyTorch >=2.6 defaults torch.load(weights_only=True), which can fail
		# for checkpoints that include Python objects (e.g. numpy scalars).
		try:
			return torch.load(self.checkpoint_path, map_location=self.device)
		except pickle.UnpicklingError as exc:
			if "Weights only load failed" not in str(exc):
				raise
			# Safe here because checkpoint is local and user-generated.
			return torch.load(
				self.checkpoint_path,
				map_location=self.device,
				weights_only=False,
			)

	def _parse_checkpoint(self, ckpt: Any) -> tuple[dict[str, Any], list[str], int]:
		if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
			state_dict = ckpt["model_state_dict"]
			class_names = ckpt.get("class_names")
			num_classes = ckpt.get("num_classes")
		elif isinstance(ckpt, dict):
			# Fallback for plain state_dict checkpoints
			state_dict = ckpt
			class_names = None
			num_classes = None
		else:
			raise ValueError("Unsupported checkpoint format")

		if num_classes is None:
			# Try to infer class count from the final FC layer weights
			fc_key = "fc.3.weight"
			if fc_key in state_dict:
				num_classes = int(state_dict[fc_key].shape[0])
			elif "fc.weight" in state_dict:
				num_classes = int(state_dict["fc.weight"].shape[0])
			else:
				raise ValueError(
					"Could not infer number of classes from checkpoint. "
					"Please include 'num_classes' in checkpoint metadata."
				)

		if class_names is None:
			class_names = [f"class_{i}" for i in range(num_classes)]

		if len(class_names) != num_classes:
			raise ValueError(
				"Checkpoint metadata mismatch: class_names length does not match num_classes"
			)

		return state_dict, class_names, int(num_classes)

	def _build_and_load_model(
		self, state_dict: dict[str, Any], num_classes: int
	) -> torch.nn.Module:
		model = models.resnet50(weights=None)
		in_features = model.fc.in_features

		def _clean_parallel_prefix(sd: dict[str, Any]) -> dict[str, Any]:
			return {key.replace("module.", "", 1): value for key, value in sd.items()}

		def _configure_head(sd: dict[str, Any]) -> None:
			# Support multiple head layouts depending on how the checkpoint was trained:
			# 1) fc.weight / fc.bias                         -> plain Linear
			# 2) fc.1.weight (+ fc.1.bias), out=num_classes -> Dropout + Linear
			# 3) fc.3.weight (+ fc.3.bias)                  -> Dropout + Linear(512) + ReLU + Linear
			if "fc.weight" in sd:
				model.fc = torch.nn.Linear(in_features, num_classes)
				return

			if "fc.3.weight" in sd and "fc.3.bias" in sd:
				hidden = int(sd["fc.1.weight"].shape[0]) if "fc.1.weight" in sd else 512
				model.fc = torch.nn.Sequential(
					torch.nn.Dropout(p=0.4),
					torch.nn.Linear(in_features, hidden),
					torch.nn.ReLU(inplace=True),
					torch.nn.Linear(hidden, num_classes),
				)
				return

			if "fc.1.weight" in sd and "fc.1.bias" in sd:
				model.fc = torch.nn.Sequential(
					torch.nn.Dropout(p=0.4),
					torch.nn.Linear(in_features, num_classes),
				)
				return

			raise ValueError(
				"Unsupported classifier head in checkpoint. "
				"Expected one of: fc.weight, fc.1.weight, or fc.3.weight"
			)

		# Try original keys first, then DataParallel-cleaned keys.
		try_state_dicts = [state_dict, _clean_parallel_prefix(state_dict)]
		last_error: Exception | None = None
		for current_sd in try_state_dicts:
			try:
				_configure_head(current_sd)
				model.load_state_dict(current_sd, strict=True)
				last_error = None
				break
			except Exception as exc:
				last_error = exc

		if last_error is not None:
			raise RuntimeError(
				f"Unable to load checkpoint into ResNet-50 model: {last_error}"
			) from last_error

		model.to(self.device)
		model.eval()
		return model

	def predict(self, image_bytes: bytes) -> InferenceResult:
		if self.model is None:
			raise RuntimeError("Model is not loaded")

		image = Image.open(BytesIO(image_bytes)).convert("RGB")
		tensor = self.eval_transform(image).unsqueeze(0).to(self.device)

		with torch.no_grad():
			logits = self.model(tensor)
			probs = torch.softmax(logits, dim=1).squeeze(0)

		top_k = min(self.top_k, len(self.class_names))
		top_values, top_indices = torch.topk(probs, k=top_k)

		confidence = float(top_values[0].item())
		label = self.class_names[int(top_indices[0].item())]
		is_unknown = confidence < self.threshold

		top_probabilities: list[dict[str, float | str]] = []
		for value, idx in zip(top_values.tolist(), top_indices.tolist()):
			top_probabilities.append(
				{
					"label": self.class_names[int(idx)],
					"value": float(value),
				}
			)

		return InferenceResult(
			label=label,
			confidence=confidence,
			is_unknown=is_unknown,
			top_probabilities=top_probabilities,
		)
