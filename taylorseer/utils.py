from collections.abc import Callable
import torch

Feature = torch.Tensor | tuple[torch.Tensor, ...]


def map_feature(fn: Callable, *features: Feature) -> Feature:
    if isinstance(features[0], tuple):
        return tuple(fn(*items) for items in zip(*features, strict=True))
    return fn(*features)


def validate_nonnegative_int(name: str, value: int) -> None:
    """Validate loop indices and counts, including rejection of bools."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
