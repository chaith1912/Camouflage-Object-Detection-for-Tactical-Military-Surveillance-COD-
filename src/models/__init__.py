"""
Source Models Package for COD Framework (TensorFlow 2.x)
Exposes DGNet architecture, loss functions, and gradient extraction tools.
"""

from .dgnet import (
    DGNet,
    DGNetLoss,
    TFGradientExtractor,
    ContextBranch,
    TextureBranch,
    PartialDecoder,
    build_dgnet_model,
)

__all__ = [
    "DGNet",
    "DGNetLoss",
    "TFGradientExtractor",
    "ContextBranch",
    "TextureBranch",
    "PartialDecoder",
    "build_dgnet_model",
]
