"""
Optional AOT compilation helpers mirroring the Hugging Face space implementation.
Falls back to a no-op when the `spaces` runtime is unavailable.
"""

from typing import Any, Callable, ParamSpec

try:
    import spaces  # type: ignore
except ImportError:  # pragma: no cover - optimisation is optional
    spaces = None  # type: ignore

import torch
from torch.utils._pytree import tree_map

P = ParamSpec("P")

TRANSFORMER_IMAGE_SEQ_LENGTH_DIM = torch.export.Dim("image_seq_length")
TRANSFORMER_TEXT_SEQ_LENGTH_DIM = torch.export.Dim("text_seq_length")

TRANSFORMER_DYNAMIC_SHAPES = {
    "hidden_states": {
        1: TRANSFORMER_IMAGE_SEQ_LENGTH_DIM,
    },
    "encoder_hidden_states": {
        1: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    },
    "encoder_hidden_states_mask": {
        1: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
    },
    "image_rotary_emb": (
        {
            0: TRANSFORMER_IMAGE_SEQ_LENGTH_DIM,
        },
        {
            0: TRANSFORMER_TEXT_SEQ_LENGTH_DIM,
        },
    ),
}

INDUCTOR_CONFIGS = {
    "conv_1x1_as_mm": True,
    "epilogue_fusion": False,
    "coordinate_descent_tuning": True,
    "coordinate_descent_check_all_directions": True,
    "max_autotune": True,
    "triton.cudagraphs": True,
}


def optimize_pipeline_(
    pipeline: Callable[P, Any], *args: P.args, **kwargs: P.kwargs
) -> None:
    """Compile the Qwen transformer with AOT autograd when the `spaces` runtime is available."""

    if spaces is None:  # pragma: no cover - optimisation is optional
        return

    @spaces.GPU(duration=1500)  # type: ignore[misc]
    def compile_transformer():
        with spaces.aoti_capture(pipeline.transformer) as call:  # type: ignore[attr-defined]
            pipeline(*args, **kwargs)

        dynamic_shapes = tree_map(lambda t: None, call.kwargs)
        dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

        exported = torch.export.export(
            mod=pipeline.transformer,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )

        return spaces.aoti_compile(exported, INDUCTOR_CONFIGS)  # type: ignore[attr-defined]

    spaces.aoti_apply(compile_transformer(), pipeline.transformer)  # type: ignore[attr-defined]
