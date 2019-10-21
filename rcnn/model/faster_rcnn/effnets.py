from efficientnet_pytorch.model import EfficientNet
from efficientnet_pytorch.utils import round_filters

from rcnn.model.utils.config import cfg
from rcnn.model.faster_rcnn.faster_rcnn import _fasterRCNN
from rcnn.model.rpn.rpn import _RPN
from rcnn.model.roi_pooling.modules.roi_pool import _RoIPooling
from rcnn.model.roi_crop.modules.roi_crop import _RoICrop
from rcnn.model.roi_align.modules.roi_align import RoIAlignAvg
from rcnn.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from rcnn.model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


class E(EfficientNet):
    def extract_base(self, inputs):

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        strided = 0
        for idx, block in enumerate(self._blocks):
            if block.stride == 2 or block.stride == [2]:
                strided += 1
            s = x.size()
            if s[1] > 2000:
                pass
            if strided >= 4:
                return x
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        return x

    def extract_top(self, x):
        strided = 0
        for idx, block in enumerate(self._blocks):
            if block.stride == 2 or block.stride == [2]:
                strided += 1
            s = x.size()
            if not strided >= 4:
                continue
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x


class effnet(nn.Module):
    def __init__(self, classes, b=7, pretrained=False, class_agnostic=False):
        super().__init__()
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.b = b
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        if self.pretrained:
            self.effnet = E.from_pretrained(f'efficientnet-b{self.b}')
        else:
            self.effnet = E.from_name(f'efficientnet-b{self.b}')
        self.dout_base_model = round_filters(112, self.effnet._global_params)

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.effnet.extract_base(im_data)

        # feed base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _head_to_tail(self, pool5):
        fc7 = self.effnet.extract_top(pool5).mean(3).mean(2)
        return fc7

    def _init_module(self):
        self.RCNN_cls_score = nn.Linear(round_filters(1280, self.effnet._global_params), self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(round_filters(1280, self.effnet._global_params), 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(round_filters(1280, self.effnet._global_params), 4 * self.n_classes)

        # Fix blocks
        for p in self.effnet._conv_stem.parameters():
            p.requires_grad = False
        for p in self.effnet._bn0.parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.effnet.apply(set_bn_fix)

    def create_architecture(self):
        self._init_module()
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)


if __name__ == '__main__':
    e = E.from_name('efficientnet-b2')
    p = round_filters(112, e._global_params)
    x = torch.zeros((1, 3, 224, 224))
    x1 = e.extract_base(x)
    x2 = e.extract_top(x1)
    print(x2)
