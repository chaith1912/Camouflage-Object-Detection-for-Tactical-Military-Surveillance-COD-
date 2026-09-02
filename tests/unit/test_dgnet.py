"""
Unit tests for DGNet model architecture and loss functions (TensorFlow 2.x / Keras).
"""

import sys
from pathlib import Path
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dgnet import (
    DGNet,
    DGNetLoss,
    TFGradientExtractor,
    build_dgnet_model,
)


def test_tf_gradient_extractor():
    extractor = TFGradientExtractor(kernel_type="sobel")
    dummy_img = tf.random.normal((2, 128, 128, 3))
    grad_map = extractor(dummy_img)

    assert grad_map.shape == (2, 128, 128, 1)
    assert tf.reduce_min(grad_map) >= 0.0 and tf.reduce_max(grad_map) <= 1.0


def test_dgnet_mobilenet_v3_large_forward():
    model = build_dgnet_model(
        backbone_name="mobilenet_v3_large",
        input_shape=(256, 256, 3),
        gradient_kernel="sobel",
        pretrained=False
    )

    dummy_input = tf.random.normal((2, 256, 256, 3))
    output_mask = model(dummy_input, training=False)

    assert output_mask.shape == (2, 256, 256, 1)
    assert tf.reduce_min(output_mask) >= 0.0 and tf.reduce_max(output_mask) <= 1.0


def test_dgnet_mobilenet_v3_small_forward():
    model = build_dgnet_model(
        backbone_name="mobilenet_v3_small",
        input_shape=(192, 192, 3),
        gradient_kernel="sobel",
        pretrained=False
    )

    dummy_input = tf.random.normal((1, 192, 192, 3))
    output_mask = model(dummy_input, training=False)

    assert output_mask.shape == (1, 192, 192, 1)


def test_dgnet_training_multi_task_outputs_and_loss():
    model = build_dgnet_model(
        backbone_name="mobilenet_v3_large",
        input_shape=(128, 128, 3),
        pretrained=False
    )
    loss_fn = DGNetLoss(lambda_grad_loss=1.5)

    dummy_input = tf.random.normal((2, 128, 128, 3))
    dummy_gt_mask = tf.cast(tf.random.uniform((2, 128, 128, 1), minval=0, maxval=2, dtype=tf.int32), tf.float32)

    outputs = model(dummy_input, training=True)
    assert isinstance(outputs, dict)
    assert "mask_pred" in outputs and "grad_pred" in outputs and "gt_grad" in outputs

    total_loss = loss_fn(dummy_gt_mask, outputs)
    assert float(total_loss) > 0.0


if __name__ == "__main__":
    pytest.main([__file__])
