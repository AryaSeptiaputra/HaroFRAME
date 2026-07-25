from __future__ import annotations


class IdentityModuleError(Exception):
	"""Base class for all errors raised by the identity module."""


class NoFaceDetectedError(IdentityModuleError):
	"""Raised when face analysis finds no usable face in a reference image."""


class ModelLoadError(IdentityModuleError):
	"""Raised when a provider fails to load its underlying model weights."""


class UnsupportedPipelineError(IdentityModuleError):
	"""Raised when a provider is attached to an incompatible diffusers pipeline."""


class ConflictingAdapterConfigError(IdentityModuleError):
	"""Raised when more than one primary face adapter is enabled in config."""
