
from typing import Any, Type, TypeVar, cast
import torch


def float_to_int16(x: torch.Tensor) -> torch.Tensor:
    """Converts a floating point tensor to float16, then reinterprets as int16."""
    downcast = x.type(torch.float16)
    if not downcast.isfinite().all():
        raise ValueError("Cannot convert to 16 bit: values are not finite")

    return downcast.view(torch.int16)


def int16_to_float(x: torch.Tensor) -> torch.Tensor:
    """Converts int16 to float16, then reinterprets as float32."""
    return x.view(torch.float16).type(torch.float32)
