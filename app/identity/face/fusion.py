from __future__ import annotations

import numpy as np

from app.identity.exceptions import NoFaceDetectedError
from app.identity.interfaces import FaceEmbedding, FusionStrategy


def _normalize(vector: np.ndarray) -> np.ndarray:
	norm = np.linalg.norm(vector)
	if norm == 0:
		return vector
	return vector / norm


def _fused_metadata(embeddings: list[FaceEmbedding], fused_vector: np.ndarray) -> FaceEmbedding:
	best = max(embeddings, key=lambda e: e.det_score)
	return FaceEmbedding(
		vector=fused_vector,
		det_score=float(np.mean([e.det_score for e in embeddings])),
		bbox=best.bbox,
		landmarks_5pt=best.landmarks_5pt,
		source_image_id="fused(" + ",".join(e.source_image_id for e in embeddings) + ")",
	)


def fuse_mean(embeddings: list[FaceEmbedding]) -> FaceEmbedding:
	stacked = np.stack([e.vector for e in embeddings])
	return _fused_metadata(embeddings, _normalize(stacked.mean(axis=0)))


def fuse_weighted_by_det_score(embeddings: list[FaceEmbedding]) -> FaceEmbedding:
	weights = np.array([e.det_score for e in embeddings], dtype=np.float64)
	if weights.sum() <= 0:
		weights = np.ones_like(weights)
	stacked = np.stack([e.vector for e in embeddings])
	fused_vector = _normalize(np.average(stacked, axis=0, weights=weights))
	return _fused_metadata(embeddings, fused_vector)


def fuse_best_quality(embeddings: list[FaceEmbedding]) -> FaceEmbedding:
	return max(embeddings, key=lambda e: e.det_score)


_STRATEGY_FUNCS = {
	FusionStrategy.MEAN: fuse_mean,
	FusionStrategy.WEIGHTED_BY_DET_SCORE: fuse_weighted_by_det_score,
	FusionStrategy.BEST_QUALITY: fuse_best_quality,
}


def fuse_embeddings(
	embeddings: list[FaceEmbedding],
	strategy: FusionStrategy = FusionStrategy.MEAN,
) -> FaceEmbedding:
	"""Combine multiple reference-image face embeddings into one robust identity vector."""
	if not embeddings:
		raise NoFaceDetectedError("cannot fuse an empty list of face embeddings")
	if len(embeddings) == 1:
		return embeddings[0]
	return _STRATEGY_FUNCS[strategy](embeddings)
