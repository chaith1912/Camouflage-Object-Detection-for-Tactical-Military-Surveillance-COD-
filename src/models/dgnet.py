"""
DGNet: Deep Gradient Network for Realtime Camouflaged Object Detection (COD)
Pure TensorFlow 2.x / Keras Implementation
Tactical Military Surveillance Framework

Architecture Overview:
1. Context Branch ("The Searcher"): Multi-scale feature extractor using Keras MobileNet (MobileNetV3/MobileNetV2),
   EfficientNet, or ResNet backbones.
2. Texture Branch ("The Identifier" / GIT): Gradient-Induced Transition module operating on image
   gradient magnitude maps (Sobel/Scharr/Laplacian) to discover subtle structural anomalies.
3. Partial Decoder Component (PDC): Aggregates multi-scale context and texture features via
   neighbor connections to output camouflage mask predictions and predicted gradient maps.
4. DGNetLoss: Multi-task supervision combining BCE + IoU segmentation loss and MAE gradient loss.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, initializers


class TFGradientExtractor(layers.Layer):
    """
    Differentiable TensorFlow layer for extracting image gradient magnitude maps
    using standard gradient kernels (Sobel, Scharr, or Laplacian).
    """

    def __init__(self, kernel_type: str = "sobel", **kwargs):
        super().__init__(**kwargs)
        self.kernel_type = kernel_type.lower()

        if self.kernel_type == "sobel":
            kx = np.array([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=np.float32)
            ky = np.array([[-1.0, -2.0, -1.0],
                           [ 0.0,  0.0,  0.0],
                           [ 1.0,  2.0,  1.0]], dtype=np.float32)
        elif self.kernel_type == "scharr":
            kx = np.array([[-3.0, 0.0, 3.0],
                           [-10.0, 0.0, 10.0],
                           [-3.0, 0.0, 3.0]], dtype=np.float32) / 16.0
            ky = np.array([[-3.0, -10.0, -3.0],
                           [ 0.0,   0.0,  0.0],
                           [ 3.0,  10.0,  3.0]], dtype=np.float32) / 16.0
        elif self.kernel_type == "laplacian":
            kl = np.array([[0.0,  1.0, 0.0],
                           [1.0, -4.0, 1.0],
                           [0.0,  1.0, 0.0]], dtype=np.float32)
            self.kl = tf.constant(kl[:, :, np.newaxis, np.newaxis], dtype=tf.float32)
        else:
            raise ValueError(f"Unsupported gradient kernel_type: {kernel_type}. Choose 'sobel', 'scharr', or 'laplacian'.")

        if self.kernel_type in ["sobel", "scharr"]:
            self.kx = tf.constant(kx[:, :, np.newaxis, np.newaxis], dtype=tf.float32)
            self.ky = tf.constant(ky[:, :, np.newaxis, np.newaxis], dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Input: inputs of shape [B, H, W, 3] or [B, H, W, 1]
        Output: Gradient magnitude map of shape [B, H, W, 1] normalized in range [0, 1]
        """
        if inputs.shape[-1] == 3:
            # Convert RGB to Grayscale: 0.299 R + 0.587 G + 0.114 B
            gray = 0.299 * inputs[..., 0:1] + 0.587 * inputs[..., 1:2] + 0.114 * inputs[..., 2:3]
        else:
            gray = inputs

        if self.kernel_type in ["sobel", "scharr"]:
            gx = tf.nn.conv2d(gray, self.kx, strides=[1, 1, 1, 1], padding="SAME")
            gy = tf.nn.conv2d(gray, self.ky, strides=[1, 1, 1, 1], padding="SAME")
            magnitude = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-8)
        else:  # laplacian
            magnitude = tf.abs(tf.nn.conv2d(gray, self.kl, strides=[1, 1, 1, 1], padding="SAME"))

        # Batch-wise normalization to [0, 1]
        max_val = tf.reduce_max(magnitude, axis=[1, 2, 3], keepdims=True) + 1e-8
        magnitude = magnitude / max_val
        return magnitude


class ChannelSpatialAttention(layers.Layer):
    """
    Lightweight Spatial and Channel Attention Module (CBAM-style) in Keras
    to focus on subtle camouflaged boundaries.
    """

    def __init__(self, channels: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        reduced_ch = max(channels // reduction, 8)
        self.fc1 = layers.Dense(reduced_ch, activation="relu", use_bias=False)
        self.fc2 = layers.Dense(channels, use_bias=False)
        self.spatial_conv = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid", use_bias=False)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Channel Attention
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        ca = self.fc2(self.fc1(avg_pool)) + self.fc2(self.fc1(max_pool))
        ca_weight = tf.nn.sigmoid(ca)
        x_ca = inputs * ca_weight

        # Spatial Attention
        avg_spatial = tf.reduce_mean(x_ca, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(x_ca, axis=-1, keepdims=True)
        spatial_cat = tf.concat([avg_spatial, max_spatial], axis=-1)
        sa_weight = self.spatial_conv(spatial_cat)

        return x_ca * sa_weight


class ContextBranch(layers.Layer):
    """
    Context Branch ("The Searcher"):
    Uses MobileNet (MobileNetV3 / MobileNetV2), EfficientNet, or ResNet as backbone
    to extract multi-scale semantic context features across 4 stages (strides 4, 8, 16, 32).
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        input_shape: Tuple[int, int, int] = (384, 384, 3),
        feature_channels: int = 64,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name.lower()
        self.feature_channels = feature_channels
        weights = "imagenet" if pretrained else None

        if "mobilenet_v3_large" in self.backbone_name:
            base_model = tf.keras.applications.MobileNetV3Large(
                include_top=False, weights=weights, input_shape=input_shape
            )
            # Extracted layers corresponding to strides 4, 8, 16, 32
            stage_names = [
                "expanded_conv/Add",             # Stride 4 (40 ch)
                "expanded_conv_3/Add",           # Stride 8 (40 ch)
                "expanded_conv_11/Add",          # Stride 16 (112 ch)
                "conv_1"                         # Stride 32 (960 ch)
            ]
        elif "mobilenet_v3_small" in self.backbone_name:
            base_model = tf.keras.applications.MobileNetV3Small(
                include_top=False, weights=weights, input_shape=input_shape
            )
            stage_names = [
                "expanded_conv/project/BatchNorm", # Stride 4 (16 ch)
                "expanded_conv_1/Add",            # Stride 8 (24 ch)
                "expanded_conv_7/Add",            # Stride 16 (48 ch)
                "conv_1"                          # Stride 32 (576 ch)
            ]
        elif "mobilenet_v2" in self.backbone_name:
            base_model = tf.keras.applications.MobileNetV2(
                include_top=False, weights=weights, input_shape=input_shape
            )
            stage_names = [
                "block_2_add",   # Stride 4 (24 ch)
                "block_5_add",   # Stride 8 (32 ch)
                "block_12_add",  # Stride 16 (96 ch)
                "out_relu"       # Stride 32 (1280 ch)
            ]
        elif "efficientnet" in self.backbone_name:
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=weights, input_shape=input_shape
            )
            stage_names = [
                "block2b_add",  # Stride 4
                "block3b_add",  # Stride 8
                "block5c_add",  # Stride 16
                "top_activation"# Stride 32
            ]
        elif "resnet" in self.backbone_name:
            base_model = tf.keras.applications.ResNet50(
                include_top=False, weights=weights, input_shape=input_shape
            )
            stage_names = [
                "conv2_block3_out", # Stride 4
                "conv3_block4_out", # Stride 8
                "conv4_block6_out", # Stride 16
                "conv5_block3_out"  # Stride 32
            ]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}.")

        # Multi-output extractor sub-model
        outputs = []
        for name in stage_names:
            try:
                outputs.append(base_model.get_layer(name).output)
            except ValueError:
                # Fallback to nearest available layer with compatible stride
                fallback_layer = [l for l in base_model.layers if l.name.startswith(name.split("/")[0])][-1]
                outputs.append(fallback_layer.output)

        self.extractor = models.Model(inputs=base_model.input, outputs=outputs, name="context_extractor")

        # 1x1 Adapters to project all multi-scale feature maps to standardized channel dimensions
        self.adapter1 = models.Sequential([layers.Conv2D(feature_channels, 1, use_bias=False), layers.BatchNormalization(), layers.ReLU()])
        self.adapter2 = models.Sequential([layers.Conv2D(feature_channels, 1, use_bias=False), layers.BatchNormalization(), layers.ReLU()])
        self.adapter3 = models.Sequential([layers.Conv2D(feature_channels, 1, use_bias=False), layers.BatchNormalization(), layers.ReLU()])
        self.adapter4 = models.Sequential([layers.Conv2D(feature_channels, 1, use_bias=False), layers.BatchNormalization(), layers.ReLU()])

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        feats = self.extractor(inputs)
        c1 = self.adapter1(feats[0])
        c2 = self.adapter2(feats[1])
        c3 = self.adapter3(feats[2])
        c4 = self.adapter4(feats[3])
        return c1, c2, c3, c4


class TextureBranch(layers.Layer):
    """
    Texture Branch ("The Identifier" / GIT):
    Processes image gradient magnitude maps and shallow context features
    to highlight high-frequency structural boundaries and intensity transitions.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.grad_conv = models.Sequential([
            layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.fusion = models.Sequential([
            layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            ChannelSpatialAttention(out_channels)
        ])

    def call(self, grad_map: tf.Tensor, shallow_feat: tf.Tensor) -> tf.Tensor:
        # Resize gradient map to match shallow feature resolution (Stride 4)
        target_shape = (tf.shape(shallow_feat)[1], tf.shape(shallow_feat)[2])
        grad_map_down = tf.image.resize(grad_map, size=target_shape, method="bilinear")

        grad_feat = self.grad_conv(grad_map_down)
        combined = tf.concat([shallow_feat, grad_feat], axis=-1)
        texture_feat = self.fusion(combined)
        return texture_feat


class NeighborConnectionBlock(layers.Layer):
    """
    Neighbor Connection Block (NCB):
    Fuses high-level semantic context with lower-level spatial detail from adjacent stages.
    """

    def __init__(self, channels: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.conv_high = models.Sequential([
            layers.Conv2D(channels, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.conv_low = models.Sequential([
            layers.Conv2D(channels, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.fusion = models.Sequential([
            layers.Conv2D(channels, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            ChannelSpatialAttention(channels)
        ])

    def call(self, high_feat: tf.Tensor, low_feat: tf.Tensor) -> tf.Tensor:
        target_shape = (tf.shape(low_feat)[1], tf.shape(low_feat)[2])
        high_up = tf.image.resize(high_feat, size=target_shape, method="bilinear")
        high_proc = self.conv_high(high_up)
        low_proc = self.conv_low(low_feat)
        merged = tf.concat([high_proc, low_proc], axis=-1)
        return self.fusion(merged)


class PartialDecoder(layers.Layer):
    """
    Partial Decoder Component (PDC):
    Progressively aggregates multi-scale context features (C1, C2, C3, C4)
    and texture features (T) to predict camouflage mask and edge gradient maps.
    """

    def __init__(self, channels: int = 64, num_classes: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.ncb3 = NeighborConnectionBlock(channels)
        self.ncb2 = NeighborConnectionBlock(channels)
        self.ncb1 = NeighborConnectionBlock(channels)
        self.texture_ncb = NeighborConnectionBlock(channels)

        # Output prediction heads
        self.mask_head = models.Sequential([
            layers.Conv2D(channels // 2, kernel_size=3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(num_classes, kernel_size=1)
        ])

        self.grad_head = models.Sequential([
            layers.Conv2D(channels // 2, kernel_size=3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(1, kernel_size=1)
        ])

    def call(self, c1: tf.Tensor, c2: tf.Tensor, c3: tf.Tensor, c4: tf.Tensor, texture_feat: tf.Tensor, target_shape: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
        f3 = self.ncb3(c4, c3)
        f2 = self.ncb2(f3, c2)
        f1 = self.ncb1(f2, c1)
        final_feat = self.texture_ncb(f1, texture_feat)

        mask_logits = tf.image.resize(self.mask_head(final_feat), size=target_shape, method="bilinear")
        grad_logits = tf.image.resize(self.grad_head(final_feat), size=target_shape, method="bilinear")

        return mask_logits, grad_logits


class DGNet(models.Model):
    """
    Deep Gradient Network (DGNet) for Camouflaged Object Detection in TensorFlow 2.x.
    Optimized for Edge Deployment on NVIDIA Jetson / ARM / TFLite devices.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        input_shape: Tuple[int, int, int] = (384, 384, 3),
        gradient_kernel: str = "sobel",
        num_classes: int = 1,
        feature_channels: int = 64,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.gradient_kernel = gradient_kernel
        self.num_classes = num_classes

        self.gradient_extractor = TFGradientExtractor(kernel_type=gradient_kernel)
        self.context_branch = ContextBranch(backbone_name=backbone_name, input_shape=input_shape, feature_channels=feature_channels, pretrained=pretrained)
        self.texture_branch = TextureBranch(in_channels=feature_channels, out_channels=feature_channels)
        self.decoder = PartialDecoder(channels=feature_channels, num_classes=num_classes)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, return_dict: bool = False) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        target_shape = (tf.shape(inputs)[1], tf.shape(inputs)[2])

        # 1. Image Gradient Map Extraction
        grad_map = self.gradient_extractor(inputs)

        # 2. Context Branch Multi-Scale Feature Extraction
        c1, c2, c3, c4 = self.context_branch(inputs)

        # 3. Texture Branch Feature Transition Extraction
        texture_feat = self.texture_branch(grad_map, c1)

        # 4. Partial Decoder Component Fusion
        mask_logits, grad_logits = self.decoder(c1, c2, c3, c4, texture_feat, target_shape=target_shape)

        if training or return_dict:
            return {
                "mask_pred": mask_logits,
                "grad_pred": grad_logits,
                "gt_grad": grad_map
            }

        return tf.nn.sigmoid(mask_logits)


class DGNetLoss(losses.Loss):
    """
    Multi-Task Supervision Loss Function for DGNet in TensorFlow:
    Loss_total = Loss_segmentation + lambda_grad * Loss_gradient
    """

    def __init__(self, lambda_grad_loss: float = 1.5, name: str = "dgnet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_grad_loss = lambda_grad_loss
        self.bce = losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.mae = losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def _weighted_bce_iou_loss(self, y_true: tf.Tensor, y_pred_logits: tf.Tensor) -> tf.Tensor:
        bce_loss = self.bce(y_true, y_pred_logits)
        pred_sig = tf.nn.sigmoid(y_pred_logits)

        intersection = tf.reduce_sum(pred_sig * y_true, axis=[1, 2, 3])
        union = tf.reduce_sum(pred_sig + y_true, axis=[1, 2, 3]) - intersection
        iou_loss = 1.0 - (intersection + 1.0) / (union + 1.0)
        return bce_loss + tf.reduce_mean(iou_loss)

    def call(self, y_true: tf.Tensor, y_pred_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        mask_logits = y_pred_dict["mask_pred"]
        grad_logits = y_pred_dict["grad_pred"]
        gt_grad = y_pred_dict["gt_grad"]

        seg_loss = self._weighted_bce_iou_loss(y_true, mask_logits)
        grad_loss = self.mae(gt_grad, tf.nn.sigmoid(grad_logits))

        total_loss = seg_loss + self.lambda_grad_loss * grad_loss
        return total_loss


def build_dgnet_model(
    backbone_name: str = "mobilenet_v3_large",
    input_shape: Tuple[int, int, int] = (384, 384, 3),
    gradient_kernel: str = "sobel",
    num_classes: int = 1,
    pretrained: bool = True
) -> DGNet:
    """Factory helper function to instantiate DGNet in TensorFlow."""
    return DGNet(
        backbone_name=backbone_name,
        input_shape=input_shape,
        gradient_kernel=gradient_kernel,
        num_classes=num_classes,
        pretrained=pretrained
    )
