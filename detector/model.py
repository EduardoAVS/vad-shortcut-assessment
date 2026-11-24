"""MobileNetV3-Small accident classifiers.

`AccidentMobileNetV3SmallGRU` is GRUMobileNet, the detector evaluated in the paper (frozen ImageNet backbone per
frame -> GRU -> linear head on the last step). `AccidentMobileNetV3Small` (per-frame head on the pooled backbone
features) is not used in the paper; it is kept so that `score_test_videos.load_any_checkpoint` can still load
per-frame checkpoints.
"""
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def get_mobilenet_v3_small_weights():
    return MobileNet_V3_Small_Weights.DEFAULT


def create_mobilenet_v3_small(pretrained: bool = True):
    if not pretrained:
        return mobilenet_v3_small(weights=None)

    weights = get_mobilenet_v3_small_weights()
    try:
        return mobilenet_v3_small(weights=weights)
    except Exception:
        print("Warning: ImageNet weights unavailable; using weights=None.")
        return mobilenet_v3_small(weights=None)


class AccidentMobileNetV3Small(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_size: int = 128,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = create_mobilenet_v3_small(pretrained=pretrained)

        # Use MobileNetV3-Small only as a feature extractor.
        self.backbone.classifier = nn.Identity()

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        feature_dim = self._infer_feature_dim(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    @property
    def features(self):
        return self.backbone.features

    @property
    def avgpool(self):
        return self.backbone.avgpool

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.features.eval()
            self.avgpool.eval()
        return self

    def _extract_features_4d(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def _infer_feature_dim(self, input_size: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = self._extract_features_4d(dummy)
        self.train(was_training)
        return int(features.shape[1])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return self._extract_features_4d(x)

        if x.dim() == 5:
            batch_size, num_frames, channels, height, width = x.shape
            x = x.reshape(batch_size * num_frames, channels, height, width)
            features = self._extract_features_4d(x)
            return features.reshape(batch_size, num_frames, -1)

        raise ValueError(f"expected input with 4 or 5 dimensions, got shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.classifier(features)


class AccidentMobileNetV3SmallGRU(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_size: int = 128,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.0,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.backbone = create_mobilenet_v3_small(pretrained=pretrained)

        # Use MobileNetV3-Small only as a per-frame feature extractor.
        self.backbone.classifier = nn.Identity()

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        feature_dim = self._infer_feature_dim(input_size)
        effective_dropout = 0.0 if gru_num_layers == 1 else gru_dropout

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=effective_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(gru_hidden_size, num_classes)

    @property
    def features(self):
        return self.backbone.features

    @property
    def avgpool(self):
        return self.backbone.avgpool

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.features.eval()
            self.avgpool.eval()
        return self

    def _extract_features_4d(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def _infer_feature_dim(self, input_size: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = self._extract_features_4d(dummy)
        self.train(was_training)
        return int(features.shape[1])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected input with shape [B, T, C, H, W], got {tuple(x.shape)}")

        batch_size, num_frames, channels, height, width = x.shape
        x = x.reshape(batch_size * num_frames, channels, height, width)
        features = self._extract_features_4d(x)
        return features.reshape(batch_size, num_frames, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        gru_output, _ = self.gru(features)
        last_step = gru_output[:, -1, :]
        return self.classifier(last_step)
