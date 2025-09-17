import torch
import torch.nn as nn

from .utils import BoxCoder, AnchorMatcher

class RetinaNetLoss(nn.Module):
    def __init__(
            self, 
            num_classes: int,
            focal_loss_alpha: float = 0.25,
            focal_loss_gamma: float = 2.0,
            smooth_l1_beta: float = 0.11, 
            pos_iou_threshold: float = 0.5,
            neg_iou_threshold: float = 0.4,
            box_coder: BoxCoder = None
    ):
        super(RetinaNetLoss, self).__init__()
        self.num_classes = num_classes

        # ### 5.1 Training Dense Detection
        # We use γ = 2.0 with α = .25 for all experiments but α = .5 works 
        # nearly as well (.4 AP lower).
        # ###
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        # ### 4.1 Inference and Training [Optimization]
        # ... standard smooth L1 loss used for box regression [10].
        # ###
        # Where [10] introduces the smooth L1 loss without a beta value,
        # which in the PyTorch implementation defaults to 1.0.
        # 2. Fast R-CNN architecture and training [Multi-task loss]
        #   -> https://arxiv.org/pdf/1504.08083
        # Since the paper does not specify a beta value, we will use 1.0
        # as a reasonable default.
        # Note that later implementations, e.g. Detectron and Detectron2, use
        # a beta value of 0.11 which gives slightly better results.
        self.smooth_l1_beta = smooth_l1_beta
        self.box_coder = (
            box_coder 
            if box_coder is not None 
            else BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        )

        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold

    def forward(
            self,
            cls_predictions: torch.Tensor, # (B, Total_anchors, num_classes) raw logits
            reg_predictions: torch.Tensor, # (B, Total_anchors, 4) box regression deltas
            anchors: torch.Tensor, # (Total_anchors, 4) in (x1, y1, x2, y2) format
            gt_boxes_batch: list[torch.Tensor], # list of (num_gt_boxes, 4) in (x1, y1, x2, y2) format per image in batch
            gt_labels_batch: list[torch.Tensor] # list of (num_gt_boxes, ) per image in batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = cls_predictions.shape[0]
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_positive_anchors = 0 # to normalize regression loss

        for b in range(batch_size):
            # Get the predictions for the current image
            cls_pred_i = cls_predictions[b] # (Total_anchors, num_classes)
            reg_pred_i = reg_predictions[b] # (Total_anchors, 4)
            gt_boxes_i = gt_boxes_batch[b] # (num_gt_boxes, 4)
            gt_labels_i = gt_labels_batch[b] # (num_gt_boxes, )

            # Create anchor matcher for this image
            matcher = AnchorMatcher(
                num_classes=self.num_classes,
                pos_iou_threshold=self.pos_iou_threshold,
                neg_iou_threshold=self.neg_iou_threshold,
                box_coder=self.box_coder
            ).to(cls_pred_i.device)

            # Match anchors to ground truth, where:
            # cls_targets_i: (Total_anchors, num_classes) with -1 (ignore), 0 (bg), or 1-k (fg)
            # reg_targets_i: (Total_anchors, 4) box regression targets with 0 for bg/ignored
            # positive_mask_i: (Total_anchors, ) boolean mask for positive anchors
            cls_targets_i, reg_targets_i, positive_mask_i = matcher(
                anchors, gt_boxes_i, gt_labels_i)
            
            # Classification loss (Focal Loss)
            # Convert cls_targets_i to one-hot encoding for focal loss computation
            cls_loss_mask = (cls_targets_i >= 0) # ignore anchors with -1

            # Filter predictions and targets for relevant anchors
            cls_pred_masked = cls_pred_i[cls_loss_mask] # (num_relevant_anchors, num_classes)
            cls_targets_masked = cls_targets_i[cls_loss_mask] # (num_relevant_anchors, 4)

            if cls_pred_masked.numel() > 0:
                # If at least one valid anchor ...
                # Convert 1-indexed class labels to one-hot (0-indexed) if
                # background class is 0. If ground truth labels are 1-indexed
                # then, 0 for cls_targets_i is background, so we need 
                # num_classes + 1 for one-hot encoding, or, as per paper, use
                # target where for each anchor, the "true" class index ranges
                # from 0 to num_classes-1 and use binary cross entropy for each
                # class separately, treating it as a k-binary classification problem.
                # The target for a class "c" if 1 if the anchor is of class "c",
                # else 0 (including background and other classes). 
                binary_cls_targets = torch.zeros_like(
                    cls_pred_masked, dtype=torch.float32) # (num_relevant_anchors, num_classes)
                
                # For positive anchors (where cls_targets_masked > 0), set the
                # corresponding class index to 1.
                # cls_targets_masked is 1-indexes, so subtract 1 for 0-indexing
                foreground_indices = cls_targets_masked > 0
                if foreground_indices.any():
                    binary_cls_targets[
                        foreground_indices,
                        cls_targets_masked[foreground_indices] - 1] = 1.0
                    
                # Compute BCE loss with logits unreduced
                # ### 3. Focal Loss
                # We introduce the focal loss starting from the cross entropy
                # (CE) loss for binary classification:
                # CE (p, y) = -log(p) if y=1
                #            -log(1-p) otherwise (y=0)
                # where y ∈ {-1, 1} is the ground-truth class and p ∈ [0, 1]
                # is the model's estimated probability for the class with label 
                # y=1.
                # ###
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    cls_pred_masked, binary_cls_targets, reduction="none"
                ) # (num_relevant_anchors, num_classes)

                # Compute p_t for focal loss modulating factor
                # ### 3.2 Focal Loss Definition
                # ... we note that the implementation of the loss layer combines
                # the sigmoid operation for computing p with the loss 
                # computation, resulting in greater numerical stability.
                # ###
                prob = torch.sigmoid(cls_pred_masked)

                # ### 3. Focal Loss
                # For notational convinience, we define p_t as:
                # p_t = p if y=1
                #     = 1-p otherwise
                # and rewrite CE(p, y) = CE(p_t) = -log(p_t).
                # ###
                p_t = (prob * binary_cls_targets) + (1 - binary_cls_targets) * (1 - prob)

                # Apply alpha (weighting factor) and gamma (focusing factor)
                # ### 3.1 Balanced Cross Entropy
                # A common method for addressing class imbalance is to
                # introduce a weighting factor α ∈ [0, 1] for class 1 and 1-α 
                # for class -1. In practice α may be set by the incerse class 
                # frequency or set by cross-validation.
                # ...
                # CE(p_t) = -α_t * log(p_t)
                # ###
                # Where α_t is α if y=1, and 1-α if y=0 and p_t is the model's
                # estimated probability for the class with label y=1.
                # This implementation is vectorized for all anchors and classes
                # simultaneously for both positive and negative anchors.
                alpha_factor = self.focal_loss_alpha * binary_cls_targets + \
                               (1 - self.focal_loss_alpha) * (1 - binary_cls_targets)
                
                # ### 3.2 Focal Loss Definition
                # While α balances the importance of positive/negative examples,
                # it does not differentiate between easy/hard examples. Instead,
                # we propose to reshape the loss function to down-weight easy
                # examples and thus focus training on hard negatives.
                # More formally, we propose to add a modulating factor 
                # (1 - p_t)^γ to the cross entropy loss, with tunable focusing
                # parameter γ ≥ 0.
                # When an example is misclassified and p_t is small, the
                # modulating factor is near 1 and the loss is unaffected. As p_t
                # → 1, the factor goes to 0 and the loss for well-classified
                # examples is down-weighted. The focusing parameter γ smoothly
                # adjusts the rate at which easy examples are down-weighted.
                # When γ = 0, FL is equivalent to CE, and as γ is increased, the
                # effect of the modulating factor is likewise increased (
                # we found γ = 2 to work best in our experiments). ù
                # ###
                modulating_factor = (1.0 - p_t) ** self.focal_loss_gamma

                focal_loss = alpha_factor * modulating_factor * bce_loss
                total_cls_loss += focal_loss.sum()

            # Box regression loss (Smooth L1 Loss)
            # Only for positive anchors
            if positive_mask_i.any():
                reg_pred_positive = reg_pred_i[positive_mask_i] # (num_positive_anchors, 4)
                reg_targets_positive = reg_targets_i[positive_mask_i] # (num_positive_anchors

                # Smooth L1 loss with 
                # reduction = sum (common, then normalize by num positive anchors)
                # or reduction = mean (might normalize per-element, then
                # scale by the number of pos anchors to get "sum" effect)
                # Here we use "sum" and normalize as the paper states:
                # ### 4.1 
                # The training loss is the sum the focal loss and the standard
                #  smooth L1 loss used for box regression [10].
                # ###
                # Where [10] is the Fast R-CNN paper which uses the smooth L1
                # loss with reduction = sum and then normalizes by the number
                # of positive anchors. 
                # 2.3 Fine-tuning for detection [Multi-task loss]
                # https://arxiv.org/pdf/1504.08083
                smooth_l1_loss = nn.functional.smooth_l1_loss(
                    reg_pred_positive, reg_targets_positive, 
                    beta=self.smooth_l1_beta, reduction="sum")
                
                total_reg_loss += smooth_l1_loss
                num_positive_anchors += positive_mask_i.sum().item()

        # Normalize losses by number of positive anchors across batch
        # ### 4.1 Inference and Training
        # The total focal loss of an image is computed as the sum
        # of the focal loss over all ∼100k anchors, normalized by the
        # number of anchors assigned to a ground-truth box. We perform the normalization by the number of assigned anchors,
        # not total anchors, since the vast majority of anchors are easy
        # negatives and receive negligible loss values under the focal
        # loss.
        # ###
        if num_positive_anchors == 0:
            # No anchors matched in this batch so no regression loss and
            # classification loss would only be background class. In this case
            # regression loss is 0 and classification loss will only come from
            # background and thus ignored. However, since focal_loss.sum() 
            # already handles this case, we can then just set the total 
            # regression loss to 0.
            total_reg_loss = torch.tensor(0.0, device=cls_predictions.device)
        else:
            # At least one positive anchor exists, here we normalize the total
            # loss by the total number of positive anchors in the batch. 
            total_reg_loss /= num_positive_anchors

        # Classification loss is often normalized by the number of positive 
        # anchors as well, but in RetinaNet's original implementation, it is
        # summed over all anchors and not normalized by number of positives.
        # The focal loss terms handle the weighting of positive vs negative 
        # anchors. However, it is also common to divide by the number of 
        # positive anchors (similar in what Faster R-CNN does) for both losses
        # for stability. We will do that here.
        # The paper states:
        # ### 4.1 Inference and Training
        # we normalize the focal loss by the number of anchors assigned to a 
        # ground-truth box (i.e. positive anchors).
        # ...
        # the regression loss is the number of positive anchors
        # ###
        # This implies dividing both losses by the number of positive anchors.
        if num_positive_anchors == 0:
            # Just sum over non-ignored anchors if there are no positive anchors
            total_cls_loss = (
                total_cls_loss / cls_loss_mask.sum().item()
                if cls_loss_mask.sum.item() > 0 
                else total_cls_loss
            )
        else:
            total_cls_loss /= num_positive_anchors

        # Total loss is the sum of classification and regression losses. The
        # paper does not states any weighting between the two losses, we can
        # assume that they are effectively balanced by the normalization.
        # ### 4.1 Inference and Training [Optimization]
        #  The training loss is the sum the focal loss and the standard smooth 
        # L1 loss used for box regression.
        # ###
        total_loss = total_cls_loss + total_reg_loss

        return total_loss, total_cls_loss, total_reg_loss
                


            
        

