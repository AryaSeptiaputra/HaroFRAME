from __future__ import annotations

from app.generation.exceptions import MotionPlanError
from app.generation.interfaces import CameraMotionSpec, FrameTransform, MotionPlan

_MAX_SCALE_CAP = 3.0
_FACE_PADDING_FRACTION = 0.25

_PAN_AXES = {
	"left": ("x", -1.0),
	"right": ("x", 1.0),
	"up": ("y", -1.0),
	"down": ("y", 1.0),
}


class KenBurns2DPlanner:
	"""Synthetic 2D pan/zoom motion planner (the classic "Ken Burns effect").

	Used when there is no driving video: motion is entirely synthesized from a
	CameraMotionSpec rather than derived from an external pose/depth signal. When
	``spec.face_bbox`` is known, both zoom and pan are clamped per-frame so the
	face never leaves the cropped/zoomed view.
	"""

	def plan(self, source_size: tuple[int, int], spec: CameraMotionSpec) -> MotionPlan:
		width, height = source_size
		if width <= 0 or height <= 0:
			raise MotionPlanError(f"invalid source_size: {source_size!r}")
		if spec.num_frames < 1:
			raise MotionPlanError(f"num_frames must be >= 1, got {spec.num_frames}")

		zoom_lo, zoom_hi = spec.zoom_range
		zoom_lo, zoom_hi = max(1.0, zoom_lo), max(1.0, zoom_hi)
		if zoom_lo > zoom_hi:
			zoom_lo, zoom_hi = zoom_hi, zoom_lo

		max_safe_scale = self._max_safe_scale(width, height, spec.face_bbox)
		zoom_hi = min(zoom_hi, max_safe_scale)
		zoom_lo = min(zoom_lo, zoom_hi)

		zoom_start, zoom_end = (zoom_hi, zoom_lo) if spec.direction == "out" else (zoom_lo, zoom_hi)
		pan_axis, pan_sign = _PAN_AXES.get(spec.direction, (None, 0.0))
		pan_lo, pan_hi = spec.pan_fraction

		transforms: list[FrameTransform] = []
		for index in range(spec.num_frames):
			t = index / (spec.num_frames - 1) if spec.num_frames > 1 else 0.0
			eased = self._ease(t, spec.easing)
			scale = zoom_start + (zoom_end - zoom_start) * eased

			pan_fraction = pan_lo + (pan_hi - pan_lo) * eased
			desired_tx = pan_sign * pan_fraction * width if pan_axis == "x" else 0.0
			desired_ty = pan_sign * pan_fraction * height if pan_axis == "y" else 0.0

			tx, ty = self._clamp_to_face_safe(width, height, scale, desired_tx, desired_ty, spec.face_bbox)
			transforms.append(FrameTransform(scale=scale, translate_x=tx, translate_y=ty, frame_index=index))

		return MotionPlan(transforms=transforms, num_frames=spec.num_frames, fps=spec.fps)

	@staticmethod
	def _ease(t: float, easing: str) -> float:
		if easing == "linear":
			return t
		return t * t * (3.0 - 2.0 * t)

	@staticmethod
	def _padded_face_box(face_bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
		x1, y1, x2, y2 = face_bbox
		pad_x, pad_y = (x2 - x1) * _FACE_PADDING_FRACTION, (y2 - y1) * _FACE_PADDING_FRACTION
		return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)

	def _max_safe_scale(
		self,
		width: int,
		height: int,
		face_bbox: tuple[float, float, float, float] | None,
	) -> float:
		if face_bbox is None:
			return _MAX_SCALE_CAP
		fx1, fy1, fx2, fy2 = self._padded_face_box(face_bbox)
		fx1, fy1 = max(fx1, 0.0), max(fy1, 0.0)
		fx2, fy2 = min(fx2, float(width)), min(fy2, float(height))
		cx, cy = width / 2.0, height / 2.0
		half_w_needed = max(cx - fx1, fx2 - cx)
		half_h_needed = max(cy - fy1, fy2 - cy)
		if half_w_needed <= 0 or half_h_needed <= 0:
			return _MAX_SCALE_CAP
		scale_w = (width / 2.0) / half_w_needed
		scale_h = (height / 2.0) / half_h_needed
		return max(1.0, min(scale_w, scale_h, _MAX_SCALE_CAP))

	def _clamp_to_face_safe(
		self,
		width: int,
		height: int,
		scale: float,
		desired_tx: float,
		desired_ty: float,
		face_bbox: tuple[float, float, float, float] | None,
	) -> tuple[float, float]:
		crop_w, crop_h = width / scale, height / scale
		max_tx, max_ty = max(0.0, (width - crop_w) / 2.0), max(0.0, (height - crop_h) / 2.0)
		tx = max(-max_tx, min(max_tx, desired_tx))
		ty = max(-max_ty, min(max_ty, desired_ty))
		if face_bbox is None:
			return tx, ty

		fx1, fy1, fx2, fy2 = self._padded_face_box(face_bbox)

		lo_tx, hi_tx = (fx2 - crop_w / 2.0) - width / 2.0, (fx1 + crop_w / 2.0) - width / 2.0
		if lo_tx <= hi_tx:
			tx = max(lo_tx, min(hi_tx, tx))
		lo_ty, hi_ty = (fy2 - crop_h / 2.0) - height / 2.0, (fy1 + crop_h / 2.0) - height / 2.0
		if lo_ty <= hi_ty:
			ty = max(lo_ty, min(hi_ty, ty))

		tx = max(-max_tx, min(max_tx, tx))
		ty = max(-max_ty, min(max_ty, ty))
		return tx, ty
