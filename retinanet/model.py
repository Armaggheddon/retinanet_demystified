import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import nms

from .utils import BoxCoder, AnchorGenerator
from .losses import RetinaNetLoss



class MultiResNetBackbone(nn.Module):

    _supported_backbones = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}

    def __init__(
            self,
            backbone_name: str = "resnet50",
            pretrained: bool = True,
            freeze_backbone: bool = False,
    ):
        super(MultiResNetBackbone, self).__init__()

        if backbone_name not in self._supported_backbones:
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Supported backbones are: {self._supported_backbones}")

        rn = models.get_model(
            backbone_name,
            weights="DEFAULT" if pretrained else None
        )

        # ### 4. RetinaNet Detector [Feature Pyramid Network Backbone]
        # RetinaNet uses feature pyramid levels P3 to P7, where P3 to P5 are
        # computed from the output of the corresponding ResNet residual stage 
        # (C3 through C5) ...
        # ###
        # P3 - P7 refer to the features in the FPN that is added after the 
        # resnet backbone. While C3 - C5 refer to the output of the resnet 
        # stages.
        self.layer0 = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        # This backbone implementation supports different ResNet sizes, 
        # therefore the number of output channels is not hardcoded but obtained
        # from the model itself.
        # For ResNet50+, the output channels are obtained from the last
        # bn3.num_features, while ResNet18/34 use bn2.num_features.
        # A trick is to get the layer type from the first block of layer1 where:
        # ResNet18/34 use BasicBlock while ResNet50+ use Bottleneck.
        is_bottleneck = isinstance(rn.layer1[0], models.resnet.Bottleneck)
        if is_bottleneck:
            # ResNet50, ResNet101, ResNet152
            self.out_channels = [
                self.layer2[-1].bn3.num_features,  # C3 
                self.layer3[-1].bn3.num_features,  # C4
                self.layer4[-1].bn3.num_features,  # C5
            ]
            # ResNet50 has [512, 1024, 2048] channels in C3, C4, C5
            # corresponding to [1/8, 1/16, 1/32] of the input image size.
        else:
            # ResNet18, ResNet34
            self.out_channels = [
                self.layer2[-1].bn2.num_features,  # C3
                self.layer3[-1].bn2.num_features,  # C4
                self.layer4[-1].bn2.num_features,  # C5
            ]

        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.layer0(x)
        # ### 4. RetinaNet Detector [Feature Pyramid Network Backbone]
        # .. we don’t use the high-resolution pyramid level P2 for computational 
        # reasons, ...
        # ###
        c2 = self.layer1(x) 
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # For ResNet50 and an input image of size [1, 3, 224, 224]:
        # c3: [1, 512, 28, 28] (224/8=28)
        # c4: [1, 1024, 14, 14] (224/16=14)
        # c5: [1, 2048, 7, 7] (224/32=7)
        return {"c3": c3, "c4": c4, "c5": c5}
    

class FPN(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int):
        super(FPN, self).__init__()
        self.out_channels = out_channels

        # ### 4. RetinaNet Detector [Feature Pyramid Network Backbone]
        # Following [20], we build FPN on top of the ResNet architecture [16].
        # ###
        # [20] refers to the (FPN paper)[https://arxiv.org/pdf/1612.03144] that 
        # states:
        # ### 3. Feature Pyramid Networks [Top-down pathway and lateral connections]
        # To start the iteration, we simply attach a 1×1 convolutional
        # layer on C5 to produce the coarsest resolution map. Finally, we 
        # append a 3×3 convolution on each merged map to generate the final 
        # feature map, which is to reduce the aliasing effect of upsampling.
        # ###
        # The mentioned upsampling is done via nearest neighbor upsampling by
        # a factor of 2. 
        self.lateral_convs = nn.ModuleList() # 1x1 convs to reduce channels
        self.fpn_convs = nn.ModuleList() # 3x3 convs after merging to smooth feature maps

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            
        # ### 4. RetinaNet Detector [Feature Pyramid Network Backbone]
        # P6 is obtained via a 3×3 stride-2 conv on C5, and P7 is computed by 
        # applying ReLU followed by a 3×3 stride-2 conv on P6.
        # ...
        # we include P7 to improve large object detection.
        # ###
        self.p6_conv = nn.Conv2d(
            in_channels_list[-1], 
            out_channels, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.p7_conv = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )

        # The paper states that the new convolutional layers are initialized as:
        # ### 4.1 Inference and Training [Initialization]
        # New layers added for FPN are initialized as in [20].
        # ### 
        # Where [20] refers to the FPN paper that states:
        # ### 4.2 Feature Pyramid Networks for Fast R-CNN
        # These layers are randomly initialized, as there are no pre-trained 
        # fc layers available in ResNets.
        # ###
        # The following uses the Kaiming initialization, as per the PyTorch
        # implementation at
        # https://docs.pytorch.org/vision/main/_modules/torchvision/ops/feature_pyramid_network.html#FeaturePyramidNetwork
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        # While to remain consistent with the original paper definition of the 
        # FPN the upsampling operation should use scale_factor=2, this would limit
        # the model to work only with input sizes that can be used. Here we use
        # `size` to upsample to the exact size of the lateral feature map. This 
        # allows the model to work with rectangular images and with sizes that
        # are not multiples of 2^n. Additionally, when images that match the 
        # multiple of 2^n are used, the results are identical to using
        # `scale_factor=2`. This also aligns with the torchvision implementation
        # https://docs.pytorch.org/vision/main/_modules/torchvision/ops/feature_pyramid_network.html#FeaturePyramidNetwork
        c3, c4, c5 = x["c3"], x["c4"], x["c5"]

        # Top-down pathway
        p5_lateral = self.lateral_convs[2](c5) # C5_channels -> FPN_out_channels
        p5 = self.fpn_convs[2](p5_lateral) # smoothing

        p4_lateral = self.lateral_convs[1](c4) # C4_channels -> FPN_out_channels
        p5_upsampled = nn.functional.interpolate(
            p5, size=p4_lateral.shape[-2:], mode="nearest")
        p4 = self.fpn_convs[1](p4_lateral + p5_upsampled) # merging + smoothing

        p3_lateral = self.lateral_convs[0](c3) # C3_channels -> FPN_out_channels
        p4_upsampled = nn.functional.interpolate(
            p4, size=p3_lateral.shape[-2:], mode="nearest")
        p3 = self.fpn_convs[0](p3_lateral + p4_upsampled) # merging + smoothing

        p6 = self.p6_conv(c5) # stride-2 conv on C5
        p7 = self.p7_conv(nn.functional.relu(p6)) # stride-2 conv on ReLU(P6)

        # The FPN outputs a list of feature maps that has the same number of 
        # channels (out_channels) but different spatial resolutions.
        # C3 [1/8], C4 [1/16], C5 [1/32] of the input image size.
        # P3 [1/8], P4 [1/16], P5 [1/32], P6 [1/64], P7 [1/128] of the input image size.
        # For example for a 224x224 input image and out_channels=256 with a 
        # ResNet50 backbone:
        # C3 [X, 512, 28, 28] -> P3 [X, 256, 28, 28] (224/8=28)
        # C4 [X, 1024, 14, 14] -> P4 [X, 256, 14, 14] (224/16=14)
        # C5 [X, 2048, 7, 7] -> P5 [X, 256, 7, 7] (224/32=7)
        # P6 [X, 256, 4, 4] (224/64=3.5 -> 4 due to padding and stride)
        # P7 [X, 256, 2, 2] (224/128=1.75 -> 2 due to padding and stride)
        return {"p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7}


class ClassificationSubnet(nn.Module):
    def __init__(
            self,
            fpn_out_channels: list[int],
            num_anchors: int,
            num_classes: int,
    ):
        super(ClassificationSubnet, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.fpn_out_channels = fpn_out_channels

        # ### 4. RetinaNet Detector [Classification Subnet]
        # This subnet is a small FCN attached to each FPN level; parameters
        # of this subnet are shared across all pyramid levels. 
        # ...
        # Taking an input feature map with C channels from a given pyramid
        # level, the subnet applies four 3x3 conv layer with K*A filters.
        # Finally sigmoid activations are attached to output the K*A binary
        # predictions per spatial location. 
        # ###
        # Where K is the number of classes and A is the number of anchors
        # per spatial location (9 in our case). In the paper C is the depth
        # of the features in the FPN (256 in our case).
        cls_layers = []
        for _ in range(4):
            cls_layers.append(
                nn.Conv2d(
                    in_channels=fpn_out_channels,
                    out_channels=fpn_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            cls_layers.append(nn.ReLU())
        self.cls_layers = nn.Sequential(*cls_layers)

        # Final conv layer has K*A filters to output the required predictions
        # per spatial location.
        # cls_output has the raw logits, sigmoid will be applied later
        self.cls_output = nn.Conv2d(
            fpn_out_channels, 
            num_anchors * num_classes, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )

        # ### 4.1 Inference and Training [Initialization]
        # All new conv layers except the final one in the RetinaNet subnets 
        # are initialized with bias b=0 and a Gaussian weight fill with σ=0.01.
        # ###
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # ### 4.1 Inference and Training [Initialization]
        # For the final conv layer of the classification subnet, we set the bias
        # initialization to b = -log((1 - π) / π), where π specifies that
        # at the start of training every anchor should be labeled as foreground
        # with confidence ~π. We use π = 0.01 in all experiments, although
        # results are robust to the exact values. This initialization prevents
        # the large number of background anchors from generating a large,
        # destabilizing loss at the start of training.
        # ###
        pi = 0.01 # prior probability for 
        self.cls_output.bias.data.fill_(-math.log((1 - pi) / pi))

    def forward(self, fpn_features: list[torch.Tensor]) -> torch.Tensor:
        all_cls_logits = []

        for feat in fpn_features.values(): # P3 to P7
            cls_feat = self.cls_layers(feat)
            cls_logits = self.cls_output(cls_feat) # raw logits

            # ### 4. RetinaNet Detector [Classification Subnet]
            # Finally sigmoid activations are attached to output the K*A binary
            # predictions per spatial location.
            # ###
            # Instead of applying sigmoid here, we will apply it later
            # when calculating the focal loss. This is numerically
            # more stable than applying sigmoid here and then
            # using a binary cross-entropy loss.

            # reshape outputs to:
            # [B, A * K, H, W] -> [B, H * W * A, K]
            B, _, H, W = feat.shape
            cls_pred = cls_logits.permute(0, 2, 3, 1)
            cls_pred = cls_pred.reshape( 
                B, 
                H * W * self.num_anchors,
                self.num_classes
            ) # has now size (B, H*W*A, K)
            all_cls_logits.append(cls_pred)

        return torch.cat(all_cls_logits, dim=1) # (B, sum(H*W*A), K)

class BoxRegressionSubnet(nn.Module):
    def __init__(
            self, 
            fpn_out_channels: list[int],
            num_anchors: int,
    ):
        super(BoxRegressionSubnet, self).__init__()

        self.num_anchors = num_anchors
        self.fpn_out_channels = fpn_out_channels

        # ### 4. RetinaNet Detector [Box Regression Subnet]
        # The design of the box regression subnet is identical to the 
        # classification subnet except that it terminates in 4A linear 
        # outputs per spatial location.
        # ###
        reg_layers = []
        for _ in range(4):
            reg_layers.append(
                nn.Conv2d(
                    in_channels=fpn_out_channels,
                    out_channels=fpn_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            reg_layers.append(nn.ReLU())
        self.reg_layers = nn.Sequential(*reg_layers)

        # ### 4. Inference and Training [Box Regression Subnet]
        # ... terminates in 4*A linear outputs per spatial location.
        # ###
        # Where A is the number of anchors per spatial location (9 in our case).
        # And the 4 values per anchor are the box offsets (dx, dy, dw, dh).
        self.reg_output = nn.Conv2d(
            fpn_out_channels, 
            num_anchors * 4, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )

        # ### 4.1 Inference and Training [Box Regression Subnet]
        # The object classification subnet and the box regression subnet, though
        # sharing a common structure, use separate parameters.
        # ###

        # ### 4.1
        # ...
        # ###
        # Initialization is the same as for the classification subnet. The only
        # difference is that for the box regression subnet we do not need to
        # set a different bias for the final layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fpn_features: list[torch.Tensor]) -> torch.Tensor:
        all_bbox_regressions = []

        for feat in fpn_features.values(): # P3 to P7
            reg_feat = self.reg_layers(feat)
            bbox_regression = self.reg_output(reg_feat) # raw box offsets

            # reshape outputs to:
            # (B, A * 4, H, W) -> (B, H * W * A, 4)
            B, _, H, W = feat.shape
            bbox_pred = bbox_regression.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(B, H * W * self.num_anchors, 4)
            all_bbox_regressions.append(bbox_pred)

        return torch.cat(all_bbox_regressions, dim=1) # (B, sum(H*W*A), 4)


class RetinaNetHead(nn.Module):
    def __init__(
            self,
            fpn_out_channels: list[int],
            num_anchors: int,
            num_classes: int,
    ):
        super(RetinaNetHead, self).__init__()

        # This module just combines the classification and box regression 
        # subnets.

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.fpn_out_channels = fpn_out_channels

        self.classification_subnet = ClassificationSubnet(
            fpn_out_channels=fpn_out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        self.box_regression_subnet = BoxRegressionSubnet(
            fpn_out_channels=fpn_out_channels,
            num_anchors=num_anchors,
        )

    def forward(self, fpn_features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        cls_logits = self.classification_subnet(fpn_features)
        bbox_regression = self.box_regression_subnet(fpn_features)

        return cls_logits, bbox_regression


class RetinaNet(nn.Module):
    def __init__(
            self,
            backbone_name: str = "resnet50",
            pretrained_backbone: bool = True,
            freeze_backbone: bool = False,
            num_classes: int = 80,
            fpn_out_channels: int = 256,
            anchor_scales: tuple[float, float, float] = (2**0, 2**(1/3), 2**(2/3)),
            anchor_aspect_ratios: tuple[float, float, float] = (0.5, 1.0, 2.0),
            anchor_strides: tuple[int, int, int, int, int] = (8, 16, 32, 64, 128),
            pos_iou_threshold: float = 0.5,
            neg_iou_threshold: float = 0.4,
            focal_loss_alpha: float = 0.25,
            focal_loss_gamma: float = 2.0,
            smooth_l1_loss_beta: float = 0.11,
            box_coder_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
            box_coder_xform_clip: float = math.log(1000.0 / 16.0),
            score_threshold: float = 0.05,
            nms_iou_threshold: float = 0.5,
            detections_per_img: int = 100,
    ):
        super(RetinaNet, self).__init__()

        self.num_classes = num_classes
        self.fpn_out_channels = fpn_out_channels
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.detections_per_img = detections_per_img

        # Backbone
        self.backbone = MultiResNetBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )

        # FPN
        self.fpn = FPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=fpn_out_channels,
        )

        # RetinaNet head (classification and box regression subnets)
        num_anchors_per_location = len(anchor_scales) * len(anchor_aspect_ratios)
        self.head = RetinaNetHead(
            fpn_out_channels=fpn_out_channels,
            num_anchors=num_anchors_per_location,
            num_classes=num_classes,
        )

        # Anchor generator
        self.anchor_generator = AnchorGenerator(
            strides=anchor_strides,
            scales=anchor_scales,
            aspect_ratios=anchor_aspect_ratios,
        )

        self.box_coder = BoxCoder(
            weights=box_coder_weights,
            bbox_xform_clip=box_coder_xform_clip,
        )

        # Loss function (Focal Loss for classification and Smooth L1 for boxes)
        self.loss_evaluator = RetinaNetLoss(
            num_classes=num_classes,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            smooth_l1_beta=smooth_l1_loss_beta,
            pos_iou_threshold=pos_iou_threshold,
            neg_iou_threshold=neg_iou_threshold,
            box_coder=self.box_coder,
        )

    def forward(
            self, 
            images: torch.Tensor, 
            targets: list[dict[str, torch.Tensor]] = None
    ) -> dict[str, torch.Tensor]:
        
        torch._assert(
            images.dim() == 4,
            f"Expected images to be a 4D tensor, got {images.dim()}D tensor instead."
        )

        torch._assert(
            images.dtype == torch.float32,
            f"Expected images to be a float32 tensor, got {images.dtype} tensor instead"
        )

        if self.training:
            torch._assert(
                targets is not None,
                "In training mode, targets should be passed"
            )

            torch._assert(
                len(images) == len(targets),
                f"Expected {len(images)} targets, got {len(targets)} instead"
            )

        
        backbone_features = self.backbone(images) # {"c3", "c4", "c5"}

        fpn_features = self.fpn(backbone_features) # {"p3", "p4", "p5", "p6", "p7"}

        # get the feature map sizes for anchor generation
        feature_map_sizes = []
        for feature_level in fpn_features.values():
            _, _, H, W = feature_level.shape
            feature_map_sizes.append((H, W))

        # get image size, assuming all images in the batch have the same size
        image_size = images.shape[2:] # (H, W)

        # anchor generation
        # all anchors will be (Total_anchors, 4) for a single image, then used across batch
        all_anchors_list = self.anchor_generator(
            feature_map_sizes=feature_map_sizes,
            device=images.device
        )
        all_anchors = torch.cat(all_anchors_list, dim=0) # (Total_anchors, 4)

        # prediction heads
        # cls_preds_logits: (B, Total_anchors, Num_classes)
        # bbox_regressions: (B, Total_anchors, 4)
        cls_preds_logits, reg_preds = self.head(fpn_features)

        if self.training or targets is not None:
            # When training we extract the ground truth boxes and labels
            # from the targets and compute the loss.
            # assert targets are on same device as inputs and have the 
            # required fields
            torch._assert(
                all("boxes" in t and "labels" in t for t in targets),
                "Each target should have 'boxes' and 'labels' fields"
            )
            torch._assert(
                all(t["boxes"].device == images.device and t["labels"].device == images.device for t in targets),
                "All target boxes should be on the same device as the input images"
            )

            gt_boxes_batch = [t["boxes"] for t in targets]
            gt_labels_batch = [t["labels"] for t in targets]

            loss, class_loss, box_loss = self.loss_evaluator(
                cls_preds_logits, reg_preds, all_anchors,
                gt_boxes_batch, gt_labels_batch
            )

            return {
                "loss": loss,
                "class_loss": class_loss,
                "box_loss": box_loss,
            }
        
        else:
            # During inference we post-process the raw outputs to
            # return the final detections.
            # This also checks if the forward is being exported
            # to ONNX format, in which case it returns the raw
            # predictions without post-processing.
            # if torch.onnx.is_in_onnx_export():
            #     return cls_preds_logits, reg_preds, all_anchors
            # else:
            detections = self._postprocess_detections(
                cls_preds_logits, reg_preds, all_anchors, image_size
            )
            return detections
    
    def _postprocess_detections(
            self,
            cls_preds_logits: torch.Tensor,
            reg_preds: torch.Tensor,
            anchors: torch.Tensor,
            image_size: tuple[int, int]
    ) -> list[dict[str, torch.Tensor]]:
        
        batch_size = cls_preds_logits.shape[0]
        final_detections = []

        # Apply sigmoid to get class probabilities
        cls_probs = torch.sigmoid(cls_preds_logits)

        # Iterate over batch
        for i in range(batch_size):
            image_cls_probs = cls_probs[i] # (Total_anchors, Num_classes)
            img_reg_preds = reg_preds[i] # (Total_anchors, 4)

            # 1. Filter by confidence score
            max_scores, class_idxs = image_cls_probs.max(dim=1) # (Total_anchors, )
            keep_idxs = max_scores >= self.score_threshold

            filtered_cls_probs = image_cls_probs[keep_idxs] # (Num_kept_anchors, Num_classes)
            filtered_reg_preds = img_reg_preds[keep_idxs] # (Num_kept_anchors, 4)
            filtered_anchors = anchors[keep_idxs] # (Num_kept_anchors, 4)
            
            # If no anchors remain after filtering, return empty detections
            if filtered_anchors.numel() == 0:
                final_detections.append({
                    "boxes": torch.zeros((0, 4), device=cls_preds_logits.device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=cls_preds_logits.device),
                    "scores": torch.zeros((0,), device=cls_preds_logits.device),
                })
                continue

            # 2. Decode bounding box predictions
            # Convert regression deltas relative to anchors into absolute
            # box coordinates
            decoded_boxes = self.box_coder.decode(
                filtered_anchors,
                filtered_reg_preds, 
            )

            # 3. Clip boxes to image size
            # The paper references the technique used in
            # Faster R-CNN to clip boxes to image boundaries
            # https://arxiv.org/pdf/1506.01497
            # ### 3.3 Implementation Details
            # This may generate crossboundary proposal boxes, which we clip to the image
            # boundary.
            # ###
            H, W = image_size
            decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], min=0, max=W) # x1
            decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], min=0, max=H) # y1
            decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], min=0, max=W) # x2
            decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], min=0, max=H) # y2

            # 4. Remove boxes with invalid dimensions, e.g. x2 <= x1 or y2 <= y1
            # or height or width <= 0
            # Again the paper references the technique used in
            # Faster R-CNN to remove such boxes
            # https://arxiv.org/pdf/1506.01497
            # ### 3.3 Implementation Details
            # During training, we ignore all cross-boundary anchors so they do 
            # not contribute to the loss.
            # ###
            # Here we remove them again during inference to avoid issues
            # during NMS.
            widths = decoded_boxes[:, 2] - decoded_boxes[:, 0]
            heights = decoded_boxes[:, 3] - decoded_boxes[:, 1]
            # Use a small epsilon (1e-2) to avoid floating point precision issues
            valid_boxes_mask = (widths > 1e-2) & (heights > 1e-2) # (Num_kept_anchors, )

            decoded_boxes = decoded_boxes[valid_boxes_mask]
            filtered_cls_probs = filtered_cls_probs[valid_boxes_mask]

            if decoded_boxes.numel() == 0:
                final_detections.append({
                    "boxes": torch.zeros((0, 4), device=cls_preds_logits.device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=cls_preds_logits.device),
                    "scores": torch.zeros((0,), device=cls_preds_logits.device),
                })
                continue 

            # 5. Apply Non-Maximum Suppression (NMS) and Top-K filtering
            # For NMS we need to process per-class to avoid suppressing 
            # boxes of different classes. The output of filtered_cls_probs
            # has shape (Num_valid_boxes, Num_classes)
            # We want to get the final detection for each class.
            # PyTorch implementation uses batched NMS to run this step faster,
            # but here we implement the per-class NMS as described in the paper
            # for clarity.
            image_boxes = []
            image_labels = []
            image_scores = []

            for class_idx in range(self.num_classes):
                # Get the class scores for this class
                scores_per_class = filtered_cls_probs[:, class_idx] # (Num_valid_boxes, )

                # Apply NMS. We need to select the boxes and scores for the 
                # current class only. We apply thresholding again in case
                # some boxes have low scoring classes for a box which overall
                # had a high max score across classes.
                # This ensures we only NMS boxes relevant to this specific class
                # at its score level.
                class_specific_keep_idx = scores_per_class > self.score_threshold
                if not class_specific_keep_idx.any():
                    continue

                boxes_for_nms = decoded_boxes[class_specific_keep_idx]
                scores_for_nms = scores_per_class[class_specific_keep_idx]

                # nms_indices are the indices of the boxes that are kept
                nms_indices = nms(
                    boxes_for_nms,
                    scores_for_nms,
                    self.nms_iou_threshold
                )

                image_boxes.append(boxes_for_nms[nms_indices])
                image_labels.append(
                    torch.full(
                        (len(nms_indices),), 
                        class_idx + 1, # +1 because class 0 is background  
                        dtype=torch.long,
                        device=cls_preds_logits.device
                    )
                )
                image_scores.append(scores_for_nms[nms_indices])

            if len(image_boxes) == 0:
                final_detections.append({
                    "boxes": torch.zeros((0, 4), device=cls_preds_logits.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=cls_preds_logits.device),
                    "scores": torch.zeros((0,), device=cls_preds_logits.device),
                })
                continue

            # Concatenate results from all classes
            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)

            # Sort by score and select Top-K detections
            if image_scores.numel() > self.detections_per_img:
                topk = torch.topk(
                    image_scores, self.detections_per_img, sorted=True)
                image_boxes = image_boxes[topk.indices]
                image_labels = image_labels[topk.indices]
                image_scores = image_scores[topk.indices]

            final_detections.append({
                "boxes": image_boxes,
                "labels": image_labels,
                "scores": image_scores,
            })

        

        return final_detections