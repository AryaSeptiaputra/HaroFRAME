from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True, frozen=True)
class LoraSpec:
	"""Resolved, ready-to-load LoRA weights -- the runtime counterpart of LoraEntryConfig."""

	adapter_name: str
	source: str
	weight_name: str | None
	subfolder: str | None
	scale: float


class LoraManager(Protocol):
	"""Attaches/detaches a stack of named LoRA adapters onto an SDXL pipeline.

	Loaded once per pipeline construction (not per frame) -- LoRA-induced style
	is therefore constant across an entire generated clip by construction.
	"""

	def load(self, pipeline: Any) -> None:
		...

	def unload(self, pipeline: Any) -> None:
		...

	def active_adapter_names(self, pipeline: Any) -> list[str]:
		...
