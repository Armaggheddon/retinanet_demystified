import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class BoxCoder:
	""" This class encodes and decodes a set of bounding boxes into
	the representation used for training the regressors.
	It is based on the torchvision implementation in
	https://github.com/pytorch/vision/blob/e9e0ee24fb9b9ec08e8ebcddd363b1bee289ffa6/torchvision/models/detection/_utils.py#L122
	and the Fast R-CNN paper: https://arxiv.org/pdf/1506.01497
	"""
	def __init__(
			self,
			weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
			bbox_xform_clip: float = math.log(1000.0 / 16)
	):
		# The paper uses the same weights for all parameters of the 
		# bounding boxes as per the Fast R-CNN paper.
		# 
		# ### 3.1.2 Loss Function
		# For bounding box regression, we adopt the parameterizations of the 
		# 4 coordinates following [5]:
		# tx = (x − xa)/wa, ty = (y − ya)/ha,
		# tw = log(w/wa), th = log(h/ha),
		# t_x* = (x* − xa)/wa, t_y* = (y* − ya)/ha,
		# t_w* = log(w*/wa), t_h* = log(h*/ha)
		# where x, y, w, and h denote the box's center coordinates,
		# and its width and height. Variables x, x_a and x* are for the 
		# predicted box, anchor box and ground-truth box respectively.
		# ###
		# Weights are scaling factors for (tx, ty, tw, th) used in 
		# encoding/decoding to control the magnitude of the regression
		# targets.
		# In this implmentation, we reference tx, ty, tw, th as dx, dy, dw, dh
		# as they are effectively deltas from the anchor boxes.
		self.weights = torch.tensor(weights, dtype=torch.float32)
		
		# Clips the predicted tw and th to prevent too large/small predicted
		# dimensions, defailt is log(1000/16) as in Detectron2
		self.bbox_xform_clip = bbox_xform_clip

	@staticmethod
	def _box_to_center_form(boxes: torch.Tensor) -> torch.Tensor:
		""" Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.
		Args:
			boxes (Tensor): boxes in (x1, y1, x2, y2) format, shape (N, 4)
		Returns:
			boxes (Tensor): boxes in (cx, cy, w, h) format, shape (N, 4)
		"""
		x1, y1, x2, y2 = boxes.unbind(-1)
		cx = (x1 + x2) / 2
		cy = (y1 + y2) / 2
		w = x2 - x1
		h = y2 - y1
		return torch.stack((cx, cy, w, h), dim=-1)
	
	@staticmethod
	def _box_to_corner_form(boxes: torch.Tensor) -> torch.Tensor:
		""" Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.
		Args:
			boxes (Tensor): boxes in (cx, cy, w, h) format, shape (N, 4)
		Returns:
			boxes (Tensor): boxes in (x1, y1, x2, y2) format, shape (N, 4)
		"""
		cx, cy, w, h = boxes.unbind(-1)
		x1 = cx - 0.5 * w
		y1 = cy - 0.5 * h
		x2 = cx + 0.5 * w
		y2 = cy + 0.5 * h
		return torch.stack((x1, y1, x2, y2), dim=-1)

	def encode(
			self,
			anchors: torch.Tensor,
			gt_boxes: torch.Tensor
	) -> torch.Tensor:
		""" Encode a set of boxes with respect to anchors.
		Args:
			anchors (Tensor): anchors in (x1, y1, x2, y2) format, shape (N, 4)
			boxes (Tensor): target boxes in (x1, y1, x2, y2) format, shape (N, 4)
		Returns:
			encoded_boxes (Tensor): encoded boxes in (dx, dy, dw, dh) format,
				shape (N, 4)
		"""
		if anchors.shape != gt_boxes.shape:
			raise ValueError("Anchors and boxes must have the same shape.")
		
		anchors_cxywh = self._box_to_center_form(anchors)
		gt_boxes_cxywh = self._box_to_center_form(gt_boxes)

		# handle 0-div for anchors with zero w/h using epsilon
		eps = torch.finfo(anchors_cxywh.dtype).eps
		Aw = anchors_cxywh[:, 2].clamp(min=eps)
		Ah = anchors_cxywh[:, 3].clamp(min=eps)

		dx = (gt_boxes_cxywh[:, 0] - anchors_cxywh[:, 0]) / Aw
		dy = (gt_boxes_cxywh[:, 1] - anchors_cxywh[:, 1]) / Ah
		dw = torch.log(gt_boxes_cxywh[:, 2] / Aw)
		dh = torch.log(gt_boxes_cxywh[:, 3] / Ah)

		encoded_boxes = torch.stack((dx, dy, dw, dh), dim=-1) * self.weights.to(anchors.device)
		return encoded_boxes
	
	def decode(
			self,
			anchors: torch.Tensor,
			box_deltas: torch.Tensor
	) -> torch.Tensor:
		""" Decode a set of box deltas with respect to anchors.
		Args:
			anchors (Tensor): anchors in (x1, y1, x2, y2) format, shape (N, 4)
			box_deltas (Tensor): encoded boxes in (dx, dy, dw, dh) format,
				shape (N, 4)
		Returns:
			decoded_boxes (Tensor): decoded boxes in (x1, y1, x2, y2) format,
				shape (N, 4)
		"""
		if anchors.shape != box_deltas.shape:
			raise ValueError("Anchors and box_deltas must have the same shape.")
		
		anchors_cxywh = self._box_to_center_form(anchors)
		
		# Handle 0-div for anchors with zero w/h using epsilon
		eps = torch.finfo(anchors_cxywh.dtype).eps
		box_deltas = box_deltas / self.weights.to(anchors.device)
		box_deltas[:, 2:] = torch.clamp(
			box_deltas[:, 2:], max=self.bbox_xform_clip) # clip dw, dh

		# Extract anchor coordinates and deltas
		Ax, Ay, Aw, Ah = anchors_cxywh.unbind(-1)
		dx, dy, dw, dh = box_deltas.unbind(-1)

		# Decode to predict center form
		Px = Ax + dx * Aw
		Py = Ay + dy * Ah
		Pw = Aw * torch.exp(dw)
		Ph = Ah * torch.exp(dh)

		decoded_boxes = torch.stack((Px, Py, Pw, Ph), dim=-1)
		return self._box_to_corner_form(decoded_boxes)


class AnchorGenerator(nn.Module):
	""" Generate anchors for feature maps at different levels of the FPN."""
	def __init__(
			self,
			strides: tuple[int, int, int, int, int] = (8, 16, 32, 64, 128),
			scales: tuple[float, float, float] = (2**0, 2**(1/3), 2**(2/3)),
			aspect_ratios: tuple[float, float, float] = (0.5, 1.0, 2.0)
	):
		""" Initialize the anchor generator.
		Args:
			strides (tuple): strides for each feature map level,
				e.g. (8, 16, 32, 64, 128) for FPN levels P3 to P7
			scales (tuple): scales for anchors at each level,
				e.g. (2^0, 2^(1/3), 2^(2/3)) as per RetinaNet paper, H/W
			aspect_ratios (tuple): aspect ratios for anchors,
				e.g. (0.5, 1.0, 2.0) as per RetinaNet paper
		"""
		super(AnchorGenerator, self).__init__()

		# Strides is the cumulative stride of the feature map relative to the 
		# input image. Therefore are mostly backbone dependent, while being 
		# input agnostic. For ResNet18 + FPN are calculated as follows:
		# rn.layer0 conv(stride=2) + maxpool(stride=2); spatial size = 1/4
		#	-> stride = 2*2 = 4
		# rn.layer1 conv(stride=1); stride = 4*1 = 4 (no further downsampling)
		# (C3 128*28*28) rn.layer2 conv(stride=2); stride = 4*2 = 8
		# (C4 256*14*14) rn.layer3 conv(stride=2); stride = 8*2 = 16
		# (C5 512*7*7) rn.layer4 conv(stride=2); stride = 16*2 = 32
		# These correspond to FPN channels P3, P4, P5, P6, P7 respectively.
		# FPN adds P6 and P7 so it becomes:
		# fpn.p6 conv(stride=2) on C5; stride = 32*2 = 64
		# fpn.p7 conv(stride=2) on P6; stride = 64*2 = 128
		# Therefore the final strides are (4, 8, 16, 32, 64, 128)
		self.strides = strides
		# Scales are multiplicative factors to the base anchor size
		# ### 4. RetinaNed Detector [Anchors]
		# For denser scale coverage than in [20], at each level we add anchors 
		# of sizes {2^0, 2^(1/3), 2^(2/3)} of the original set of 3 aspect
		# ratios.
		# ###
		self.scales = scales
		# ### 4. RetinaNed Detector [Anchors]
		# As in [20], at each pyramid level we use anchors at
		# three aspect ratios {1:2, 1:1, 2:1}
		# ###
		self.aspect_ratios = aspect_ratios

		# Cache cell anchors for different feature map sizes
		self._cell_anchors = {}

	def _generate_anchors_for_cell(
			self,
			stride: int,
			scales: tuple[float, ...],
			aspect_ratios: tuple[float, ...],
			device: torch.device
	) -> torch.Tensor:
		# generates a set of anchors for a single 1x1 feature map cell
        # at a given stride, scales and aspect ratios
        # return anchors in (x_center, y_center, width, height) format
		if (stride, tuple(scales), tuple(aspect_ratios)) in self._cell_anchors:
			return self._cell_anchors[(
				stride, tuple(scales), tuple(aspect_ratios))]
		
		anchors = []

		# The paper uses base_size = stride * 4, howver recent implementations
        # tend to use base_size = stride and then apply scales on top of that.
        # anchor size increases with stride, usijng explicit base_sizes for each FPN
        # level directly based on paper sizes P3: 32, P4: 64, P5: 128, P6: 256, 
		# P7: 512
        # the FPN strides are 8, 16, 32, 64, 128 respectively
        # so a mapping could be base_size = stride * 4
        # so for P3 smallest anchor might be 32*32, for P4 64*64, etc...
		# ### 4. RetinaNed Detector [Anchors]
		# The anchors have areas of 32^2, to 512^2 pixels on pyramid levels 
		# P3 to P7, respectively.
		# ### 
		effective_base_size = stride * 4
		for scale in scales:
			for ratio in aspect_ratios:
				# area = (effective_base_size * scale) ** 2
				# w = sqrt(area * aspect_ratio)
				# h = sqrt(area / aspect_ratio)
				w = effective_base_size * scale * (ratio ** 0.5)
				h = effective_base_size * scale / (ratio ** 0.5)

				# Anchors are defined as relative to the cell center (0, 0)
				# so center is (0, 0), later they will be shifted to the
				# actual cell locations (x_center, y_center, width, height)
				# (x1, y1, x2, y2) w.r.t. cell center (0, 0)
				anchors.append([-w / 2, -h / 2, w / 2, h / 2])
		
		anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
		self._cell_anchors[(stride, tuple(scales), tuple(aspect_ratios))] = anchors
		return anchors
	
	def forward(
			self,
			feature_map_sizes: list[tuple[int, int]],
			device: torch.device
	) -> list[torch.Tensor]:
		""" Generate anchors for all feature maps.
		Args:
			feature_map_sizes (list[tuple[int, int]]): list of (height, width)
				for each feature map from P3 to P7. For a 224x224 input image
				it would be [(28, 28), (14, 14), (7, 7), (4, 4), (2, 2)]
			device (torch.device): device to put the anchors on
		Returns:
			anchors (Tensor): list of anchors for each feature map,
				each of shape (H*W*num_anchors_per_cell, 4) in 
				(x1, y1, x2, y2) format
		"""
		all_anchors = []
		for i, (H, W) in enumerate(feature_map_sizes):
			stride = self.strides[i]
			cell_anchors = self._generate_anchors_for_cell(
				stride, self.scales, self.aspect_ratios, device)
			
			# Generate grid offsets for the center of each cell in the feature 
			# map grid. Each grid center should be at (stride/2, stride/2),
			# (3*stride/2, stride/2), ...
			shift_x = torch.arange(0, W, device=device) * stride + stride // 2
			shift_y = torch.arange(0, H, device=device) * stride + stride // 2

			# Create a grid of (x, y) coordinates
			shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
			# shifts has shape (H*W, 4) in (cx, cy, cx, cy) format
			shifts = torch.stack(
				(shift_x.flatten(), shift_y.flatten(),
				shift_x.flatten(), shift_y.flatten()), dim=1)
			
			# Add base anchors to each shift to get all anchors for this feature
			# map
			# cell_anchors.shape: (num_anchors_per_cell, 4)
			# shifts.shape: (H*W, 4)
			# all_anchors.shape: (H*W, num_anchors_per_cell, 4)
			# then reshape to (H*W*num_anchors_per_cell, 4) + (H*W, 1, 4)
			# broadcasting adds cell_anchors to each shift
			anchors_per_level = (
				cell_anchors.reshape(1, -1, 4) + shifts.reshape(-1, 1, 4))
			# Flatten to (H*W*num_anchors_per_cell, 4)
			anchors_per_level = anchors_per_level.reshape(-1, 4) 
			all_anchors.append(anchors_per_level)

		return all_anchors


class AnchorMatcher(nn.Module):
	def __init__(
			self,
			num_classes: int = 80,
			pos_iou_threshold: float = 0.5,
			neg_iou_threshold: float = 0.4,
			box_coder: BoxCoder = None
	):
		super(AnchorMatcher, self).__init__()
		self.num_classes = num_classes
		# ### 4. RetinaNed Detector [Anchors]
		# Specifically, anchors are assigned to ground-truth object boxes using
		# an intersection-over-union (IoU) threshold of 0.5; and to background 
		# if their IoU is in [0, 0.4)
		# ###
		self.pos_iou_threshold = pos_iou_threshold
		self.neg_iou_threshold = neg_iou_threshold
		self.box_coder = (
			box_coder
			if box_coder is not None 
			else BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
		)

	def forward(
			self,
			anchors: torch.Tensor,
			gt_boxes: torch.Tensor,
			gt_labels: torch.Tensor
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		""" Match anchors to ground truth boxes and encode the targets.
		Args:
			anchors (Tensor): anchors in (x1, y1, x2, y2) format, shape (N, 4)
			gt_boxes (Tensor): ground truth boxes in (x1, y1, x2, y2) format,
				shape (M, 4)
			gt_labels (Tensor): ground truth labels, shape (M, )
		Returns:
			cls_targets (Tensor): classification targets, shape (N, num_classes)
				-1 for ignore, 0 for background, 1-K for K classes (foreground)
			reg_targets (Tensor): regression targets, shape (N, 4), 0 for
				background/ignored anchors
			positive_masks (Tensor): boolean mask for positive anchors, 
				shape (N, ), 1 for positive anchors, 0 otherwise
		"""
		num_anchors = anchors.shape[0]
		num_gt = gt_boxes.shape[0]

		# cls_targets initialized to -1 (ignore), 0 for background, 1-K for 
		# K classes (foreground)
		cls_targets = torch.full(
			(num_anchors, ), -1, dtype=torch.long, device=anchors.device)
		# reg_targets initialized to 0, will be populated for positive anchors
		reg_targets = torch.zeros(
			(num_anchors, 4), dtype=torch.float32, device=anchors.device)
		
		if num_gt == 0:
			# All anchors are negative (background)
			cls_targets[:] = 0
			return (
				cls_targets, 
				reg_targets, 
				cls_targets.new_zeros(num_anchors, dtype=torch.bool)
			)
		
		# Calculate IoU between all anchors and all gt_boxes
		iou_matrix = box_iou(anchors, gt_boxes) # (N_anchors, N_gt)

		# Find best gt box for each anchor (max iou)
        # max_iou_poer_anchor: (N_anchors, ) max iou an anchor has with ANY GT
        # matched_gt_idx_per_anchor: (N_anchors, ) index of the gt box that 
		# 		yields the max_iou_per anchor
		max_iou_per_anchor, matched_gt_idx_per_anchor = iou_matrix.max(dim=1)

		# Find best anchor for each gt box (max iou)
        # max_iou_per_gt: (Num_gt, ) max iou a gt box has with ANY anchor
        # matched_anchor_idx_per_gt: (Num_gt, ) index of the anchor that yields
		# 		the max_iou_per_gt box 
		max_iou_per_gt, matched_anchor_idx_per_gt = iou_matrix.max(dim=0)

		# Determine anchor assignments
		# 1. Anchors with IoU >= pos_iou_threshold are positive
		pos_mask = max_iou_per_anchor >= self.pos_iou_threshold

		# 2. For each gt box, ensure its best matching anchor is positive.
		# This addresses the cases where a gt box might not have an anchor
		# with IoU >= pos_iou_threshold but it still needs to be assigned
		# to at least one anchor (=detected).
		# Find the anchors that are the best match for ANY gt box, ensure that
		# max_iou_per_gt has a threshold check to avoid promoting very low
		# IoU anchors (bad matches). This is often done by setting 
		# matches_anchor_idx_per_gt as positive if their max_iou_per_gt is
		# sufficiently high. For simplicity and common retinanet 
		# implementations, we will promote the ABSOLUTE BEST for each gt box

		# Ensure that each gt box has at least one positive anchor assigneg.
		# The specific gt box for these promoted anchors will be 
		# matched_gt_idx_per_anchor[matched_anchor_idx_per_gt].
		# The matched_gt_idx_per_anchor already contains the gt box index that
		# yields the max IoU for each anchor so we just need a positive mask for 
		# these "best-of-best" anchors.
		pos_mask[matched_anchor_idx_per_gt] = True

		# 3. Anchors with IoU < neg_iou_threshold are negative (background)
		# only if not already positive. &(~pos_mask) excludes anchors
		# that are already marked as positive.
		neg_mask = (max_iou_per_anchor < self.neg_iou_threshold) & (~pos_mask)

		# 4. Fill target tensors for 
		# Classification targets, 0 for bg (neg_mask)
		cls_targets[neg_mask] = 0
		# gt label for fg (pos_mask) (with the 1-indexing in mind)
		cls_targets[pos_mask] = gt_labels[matched_gt_idx_per_anchor[pos_mask]]

		# Regression targets only for positive anchors
		if pos_mask.any():
			matched_gt_boxes_for_pos_anchors = gt_boxes[
				matched_gt_idx_per_anchor[pos_mask]]
			pos_anchors = anchors[pos_mask]

			# Encode gt boxes relative to the positive anchors
			reg_targets[pos_mask] = self.box_coder.encode(
				pos_anchors, matched_gt_boxes_for_pos_anchors)

		return cls_targets, reg_targets, pos_mask

