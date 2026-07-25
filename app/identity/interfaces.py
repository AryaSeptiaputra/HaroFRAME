from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Protocol

import numpy as np
from PIL import Image


class FusionStrategy(str, Enum):
	"""How multiple reference-image face embeddings are combined into one."""

	MEAN = "mean"
	WEIGHTED_BY_DET_SCORE = "weighted_by_det_score"
	BEST_QUALITY = "best_quality"


@dataclass(slots=True, frozen=True)
class FaceEmbedding:
	"""A single detected face's identity signal, extracted from one image."""

	vector: np.ndarray
	det_score: float
	bbox: tuple[float, float, float, float]
	landmarks_5pt: np.ndarray | None = None
	source_image_id: str = ""


@dataclass(slots=True)
class IdentityReference:
	"""One or more reference photos of the same person plus their derived embeddings."""

	images: list[Image.Image]
	embeddings: list[FaceEmbedding] = field(default_factory=list)
	fused_embedding: FaceEmbedding | None = None


@dataclass(slots=True, frozen=True)
class StructureHint:
	"""Pose/depth structure signal, deliberately separate from :class:`IdentityReference`.

	The images here may come from the identity reference photo, or later from an
	unrelated driving video frame — identity conditioning must not assume the two
	are the same source.
	"""

	# Raw driving photos, not precomputed control maps — a StructureConditioner's
	# preprocess() turns these into the actual pose/depth control image.
	pose_image: Image.Image | None = None
	depth_image: Image.Image | None = None
	source: Literal["reference", "driving_frame", "explicit"] = "explicit"


@dataclass(slots=True)
class IdentityConditioning:
	"""Bundled, pipeline-ready conditioning produced by one or more providers."""

	adapter_kwargs: dict[str, Any] = field(default_factory=dict)
	applied_adapters: list[str] = field(default_factory=list)
	used_structure_conditioning: bool = False
	restorer_hook: Callable[[Image.Image], Image.Image] | None = None


class FaceConditioningProvider(Protocol):
	"""Shared shape of any provider that turns a face identity into pipeline kwargs.

	Both the IP-Adapter-family providers (``ipadapter/``) and InstantID
	(``instantid/``) satisfy this Protocol, which lets :class:`IdentityEngine`
	hold a single ``face_adapter`` slot regardless of which one is configured.
	"""

	def load(self, pipeline: Any) -> None:
		"""Attach/register this provider's weights onto a diffusers pipeline."""
		...

	def build_conditioning(
		self,
		reference: IdentityReference,
		*,
		structure: StructureHint | None = None,
		scale: float = 1.0,
	) -> IdentityConditioning:
		"""Build the kwargs needed to condition a pipeline call on ``reference``."""
		...

	def unload(self, pipeline: Any) -> None:
		"""Detach this provider's weights from a diffusers pipeline."""
		...
