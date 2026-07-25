from __future__ import annotations

import numpy as np
import pytest

from app.identity.exceptions import NoFaceDetectedError
from app.identity.face.fusion import (
	fuse_best_quality,
	fuse_embeddings,
	fuse_mean,
	fuse_weighted_by_det_score,
)
from app.identity.interfaces import FaceEmbedding, FusionStrategy


def _embedding(vector, det_score, source_image_id="0"):
	return FaceEmbedding(
		vector=np.asarray(vector, dtype=np.float32),
		det_score=det_score,
		bbox=(0.0, 0.0, 10.0, 10.0),
		source_image_id=source_image_id,
	)


def test_fuse_embeddings_empty_raises():
	with pytest.raises(NoFaceDetectedError):
		fuse_embeddings([])


def test_fuse_embeddings_single_returns_as_is():
	emb = _embedding([1.0, 0.0], 0.9)
	assert fuse_embeddings([emb]) is emb


def test_fuse_mean_normalizes_average():
	a = _embedding([1.0, 0.0], 0.9, "0")
	b = _embedding([0.0, 1.0], 0.8, "1")

	fused = fuse_mean([a, b])

	expected = np.array([1.0, 1.0]) / np.linalg.norm([1.0, 1.0])
	np.testing.assert_allclose(fused.vector, expected, atol=1e-6)
	assert fused.det_score == pytest.approx((0.9 + 0.8) / 2)
	assert fused.source_image_id == "fused(0,1)"


def test_fuse_weighted_by_det_score_favors_higher_score():
	a = _embedding([1.0, 0.0], 1.0, "0")
	b = _embedding([0.0, 1.0], 0.0, "1")

	fused = fuse_weighted_by_det_score([a, b])

	np.testing.assert_allclose(fused.vector, [1.0, 0.0], atol=1e-6)


def test_fuse_weighted_by_det_score_falls_back_to_uniform_when_all_zero():
	a = _embedding([1.0, 0.0], 0.0, "0")
	b = _embedding([0.0, 1.0], 0.0, "1")

	fused = fuse_weighted_by_det_score([a, b])

	expected = np.array([1.0, 1.0]) / np.linalg.norm([1.0, 1.0])
	np.testing.assert_allclose(fused.vector, expected, atol=1e-6)


def test_fuse_best_quality_returns_highest_scoring_embedding_unchanged():
	a = _embedding([1.0, 0.0], 0.4, "0")
	b = _embedding([0.0, 1.0], 0.95, "1")

	assert fuse_best_quality([a, b]) is b


@pytest.mark.parametrize(
	"strategy,expected_func",
	[
		(FusionStrategy.MEAN, fuse_mean),
		(FusionStrategy.WEIGHTED_BY_DET_SCORE, fuse_weighted_by_det_score),
		(FusionStrategy.BEST_QUALITY, fuse_best_quality),
	],
)
def test_fuse_embeddings_dispatches_to_strategy(strategy, expected_func):
	a = _embedding([1.0, 0.0], 0.9, "0")
	b = _embedding([0.3, 0.7], 0.5, "1")

	result = fuse_embeddings([a, b], strategy=strategy)
	expected = expected_func([a, b])

	np.testing.assert_allclose(result.vector, expected.vector, atol=1e-6)
