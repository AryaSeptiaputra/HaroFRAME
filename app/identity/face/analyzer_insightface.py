from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.config import FaceConfig
from app.identity.exceptions import ModelLoadError, NoFaceDetectedError
from app.identity.interfaces import FaceEmbedding, IdentityReference


class InsightFaceAnalyzer:
	"""FaceAnalyzer implementation backed by InsightFace's FaceAnalysis app.

	Model loading is deferred to the first call so constructing this class stays
	cheap even when no identity conditioning ends up being requested.
	"""

	def __init__(self, config: FaceConfig, providers: list[str] | None = None) -> None:
		self._config = config
		self._providers = providers or ["CPUExecutionProvider"]
		self._app = None

	def _ensure_loaded(self):
		if self._app is not None:
			return self._app
		try:
			from insightface.app import FaceAnalysis
		except ImportError as exc:
			raise ModelLoadError(
				"insightface is not installed; install the project's core "
				"dependencies to use InsightFaceAnalyzer"
			) from exc
		app = FaceAnalysis(name=self._config.model_pack, providers=self._providers)
		app.prepare(ctx_id=0, det_size=tuple(self._config.det_size))
		self._app = app
		return self._app

	def analyze(self, image: Image.Image) -> list[FaceEmbedding]:
		app = self._ensure_loaded()
		image_bgr = np.array(image.convert("RGB"))[:, :, ::-1]
		faces = app.get(image_bgr)

		embeddings: list[FaceEmbedding] = []
		for face in faces:
			if face.det_score < self._config.min_det_score:
				continue
			vector = getattr(face, "normed_embedding", None)
			if vector is None:
				vector = face.embedding
			vector = np.asarray(vector, dtype=np.float32)
			norm = np.linalg.norm(vector)
			if norm > 0:
				vector = vector / norm
			embeddings.append(
				FaceEmbedding(
					vector=vector,
					det_score=float(face.det_score),
					bbox=tuple(float(v) for v in face.bbox),
					landmarks_5pt=np.asarray(face.kps, dtype=np.float32) if face.kps is not None else None,
				)
			)
		return embeddings

	def analyze_reference(self, reference: IdentityReference) -> IdentityReference:
		embeddings: list[FaceEmbedding] = []
		for idx, image in enumerate(reference.images):
			faces = self.analyze(image)
			if not faces:
				continue
			best = max(faces, key=lambda f: f.det_score)
			embeddings.append(
				FaceEmbedding(
					vector=best.vector,
					det_score=best.det_score,
					bbox=best.bbox,
					landmarks_5pt=best.landmarks_5pt,
					source_image_id=str(idx),
				)
			)
		if not embeddings:
			raise NoFaceDetectedError("no face detected in any of the provided reference images")
		reference.embeddings = embeddings
		return reference
