import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder 
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder

class JointLoss(nn.Module):
    def __init__(self, first_loss, second_loss, first_weight=0.5, second_weight=0.5):
        super(JointLoss, self).__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.first_weight = first_weight
        self.second_weight = second_weight

    def forward(self, y_pred, y_true):
        loss1 = self.first_loss(y_pred, y_true)
        loss2 = self.second_loss(y_pred, y_true)
        return self.first_weight * loss1 + self.second_weight * loss2


class MultiGroupEncoderUNetPlusPlus(nn.Module):
    def __init__(self, encoder_name="resnet34", total_in_channels=6, num_classes=5, encoder_weights="imagenet"):
        super().__init__()

        assert total_in_channels % 3 == 0, "Only supports input channels divisible by 3 (e.g., 6, 9, 12)"
        self.num_groups = total_in_channels // 3

        # Create encoders, one per 3-channel group
        self.encoders = nn.ModuleList([
            smp.encoders.get_encoder(encoder_name, in_channels=3, weights=encoder_weights)
            for _ in range(self.num_groups)
        ])

        # Assume all encoders have same output channels
        encoder_channels = [e.out_channels for e in self.encoders]
        base_channels = encoder_channels[0]
        merged_channels = [c * self.num_groups for c in base_channels]

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=merged_channels,
            decoder_channels=[256, 128, 64, 32, 16],
            n_blocks=5,
            use_norm=True,
            center=False,
            attention_type=None
        )

        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        group_feats = []
        for i in range(self.num_groups):
            xi = x[:, i*3:(i+1)*3, :, :]
            feats = self.encoders[i](xi)
            group_feats.append(feats)

        # fuse across groups at each level (e.g., concat)
        fused_feats = []
        for level_feats in zip(*group_feats):
            fused = torch.cat(level_feats, dim=1)
            fused_feats.append(fused)

        decoder_out = self.decoder(fused_feats)
        return self.segmentation_head(decoder_out)

class MultiGroupDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with multiple 3-channel encoders and late feature fusion.
    Works with backbones such as 'resnet50'.
    """
    def __init__(self, encoder_name="resnet50", total_in_channels=6, num_classes=5, encoder_weights="imagenet"):
        super().__init__()

        assert total_in_channels % 3 == 0, "Only supports input channels divisible by 3 (e.g., 6, 9, 12)"
        self.num_groups = total_in_channels // 3

        # Create encoders, one per 3-channel group
        self.encoders = nn.ModuleList([
            smp.encoders.get_encoder(encoder_name, in_channels=3, weights=encoder_weights)
            for _ in range(self.num_groups)
        ])

        # Assume all encoders have same output channels
        encoder_channels = [e.out_channels for e in self.encoders]
        base_channels = encoder_channels[0]
        merged_channels = [c * self.num_groups for c in base_channels]

        # Custom DeepLabV3Plus-like decoder with adaptive upsampling
        from segmentation_models_pytorch.decoders.deeplabv3.decoder import ASPP, SeparableConv2d
        
        self.aspp = nn.Sequential(
            ASPP(
                merged_channels[-1],  # Last (deepest) feature channels
                256,
                (12, 24, 36),
                separable=True,
                dropout=0.5,
            ),
            SeparableConv2d(
                256, 256, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # High-resolution feature processing (from level 2)
        highres_in_channels = merged_channels[2]  # Level 2 features
        highres_out_channels = 48
        self.block1 = nn.Sequential(
            nn.Conv2d(
                highres_in_channels, highres_out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        
        # Final fusion block
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + 256,  # 48 + 256 = 304
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.segmentation_head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        group_feats = []
        for i in range(self.num_groups):
            xi = x[:, i*3:(i+1)*3, :, :]
            feats = self.encoders[i](xi)
            group_feats.append(feats)

        # Fuse across groups at each level (concatenate features)
        fused_feats = []
        for level_feats in zip(*group_feats):
            fused = torch.cat(level_feats, dim=1)
            fused_feats.append(fused)

        # Apply ASPP to deepest features
        aspp_features = self.aspp(fused_feats[-1])
        
        # Process high-resolution features (level 2)
        high_res_features = self.block1(fused_feats[2])
        
        # Adaptive upsampling to match high-res feature size
        if aspp_features.shape[-2:] != high_res_features.shape[-2:]:
            aspp_features = F.interpolate(
                aspp_features,
                size=high_res_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate and fuse
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        
        # Upsample to original input size
        if fused_features.shape[-2:] != x.shape[-2:]:
            fused_features = F.interpolate(
                fused_features,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        return self.segmentation_head(fused_features)


class MultiGroupSegformer(nn.Module):
    """
    SegFormer with multiple 3-channel encoders.
    Works for encoder_name = 'mit_b0' … 'mit_b5'.

    Parameters
    ----------
    encoder_name        backbone in SMP ('mit_b3' etc.)
    total_in_channels   total input channels (must be 3 × N)
    num_classes         segmentation classes
    encoder_weights     'imagenet' or None
    seg_channels        decoder's internal channel width (defaults to 256)
    """

    def __init__(
        self,
        encoder_name: str = "mit_b3",
        total_in_channels: int = 6,
        num_classes: int = 5,
        encoder_weights: str = "imagenet",
        seg_channels: int = 256,
    ):
        super().__init__()

        if total_in_channels % 3 != 0 or total_in_channels < 3:
            raise ValueError("total_in_channels must be 3 × N (N ≥ 1).")
        self.num_groups = total_in_channels // 3

        # ========== encoders (one per 3-channel group) ==========
        self.encoders = nn.ModuleList(
            [
                smp.encoders.get_encoder(
                    encoder_name, in_channels=3, weights=encoder_weights
                )
                for _ in range(self.num_groups)
            ]
        )

        # channel topo of one encoder: e.g. [64, 128, 320, 512]
        base_ch = self.encoders[0].out_channels
        encoder_depth = len(base_ch)                      # SegFormer uses depth 4
        fused_ch = [c * self.num_groups for c in base_ch] # concat across groups

        # ========== decoder ==========
        # SegformerDecoder signature in SMP:
        #   SegformerDecoder(encoder_channels, encoder_depth, segmentation_channels)
        self.decoder = SegformerDecoder(
            encoder_channels=fused_ch,
            encoder_depth=encoder_depth,
            segmentation_channels=seg_channels,
        )

        self.segmentation_head = nn.Conv2d(seg_channels, num_classes, kernel_size=1)

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats_by_group = [enc(x[:, g*3:(g+1)*3]) for g, enc in enumerate(self.encoders)]
        fused_feats    = [torch.cat([fg[i] for fg in feats_by_group], dim=1)
                          for i in range(len(feats_by_group[0]))]

        dec   = self.decoder(fused_feats)            # (B, seg_channels, H/4, W/4)
        logits = self.segmentation_head(dec)         # (B, num_classes, H/4, W/4)

        # --- upsample to full resolution ---
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits,
                                   size=x.shape[2:],
                                   mode="bilinear",
                                   align_corners=False)
        return logits
    

def build_model_for_multi_channels(model_name, encoder_name='resnet34', in_channels=3, num_classes=5):

    encoder_name_list = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
        'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d',
        'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        'efficientnet-b6', 'efficientnet-b7', 
        'mit_b1', 'mit_b2', 'mit_b3', 
        'timm-efficientnet-b5'
    ]
    if encoder_name not in encoder_name_list:
        raise ValueError(f"Unknown encoder name: {encoder_name}. Supported encoders are: {encoder_name_list}")

    if model_name == 'UnetPlusPlus':
        if in_channels ==3:
            return smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=num_classes
            )
        elif in_channels % 3 == 0 and in_channels > 3:
            return MultiGroupEncoderUNetPlusPlus(
                encoder_name=encoder_name,
                total_in_channels=in_channels,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"UnetPlusPlus only supports input channels of 3, 6, 9, etc. (divisible by 3), got {in_channels}.")
        return model_class
    
    elif model_name == 'DeepLabV3Plus':
        if in_channels == 3:
            return smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=num_classes
            )
        elif in_channels % 3 == 0 and in_channels > 3:
            return MultiGroupDeepLabV3Plus(
                encoder_name=encoder_name,
                total_in_channels=in_channels,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"DeepLabV3Plus only supports input channels divisible by 3 (e.g., 6, 9), got {in_channels}.")

    elif model_name == 'Segformer':
        if in_channels == 3:
            return smp.Segformer(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=num_classes
            )
        elif in_channels % 3 == 0 and in_channels > 3:
            return MultiGroupSegformer(
                encoder_name=encoder_name,
                total_in_channels=in_channels,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Segformer only supports input channels divisible by 3 (e.g., 6, 9), got {in_channels}.")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
