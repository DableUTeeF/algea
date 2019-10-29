import os
import cv2
import sys

# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "rcnn"

import torch
import numpy as np
from .model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from .model.rpn.bbox_transform import clip_boxes
from .model.nms.nms_wrapper import nms
from .model.rpn.bbox_transform import bbox_transform_inv
from .model.utils.net_utils import save_net, load_net, vis_detections
from .model.utils.blob import im_list_to_blob
from .model.faster_rcnn.vgg16 import vgg16
from .model.faster_rcnn.resnet import resnet
from scipy.misc import imread
import time
from boxutils import add_bbox
from evaluate_util import evaluate, all_annotation_from_instance


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':
    checkpoint = torch.load('/home/palm/PycharmProjects/algea/snapshots/rcnn/1/faster_rcnn_1_25_4307.pth')
    pascal_classes = ['__background__', 'ov', 'mif']

    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()

    folder = '/media/palm/data/MicroAlgae/16_8_62/images'
    val_set = [s.split(',')[0] for s in
               open('/home/palm/PycharmProjects/algea/dataset/test_annotations').read().split('\n')]
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    device = 'cuda'
    im_data = im_data.to(device)
    im_info = im_info.to(device)
    num_boxes = num_boxes.to(device)
    gt_boxes = gt_boxes.to(device)
    fasterRCNN.to(device)
    cfg.CUDA = device == 'cuda'
    fasterRCNN.eval()

    for file in os.listdir(folder):
        im_file = os.path.join(folder, file)
        if im_file not in val_set:
            continue
        im = np.array(imread(im_file))
        im = im[:, :, ::-1]
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
            cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
        box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        im2show = np.copy(im)
        all_detections = []
        for j in range(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > 0.05).view(-1)
            # print(inds.numel())
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                # print(cls_dets.cpu().numpy().astype('uint16'))
                # print(cls_scores.cpu().numpy())
                # print(cls_boxes.cpu().numpy().astype('uint16'))
                # im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)
                dets = cls_dets.cpu().numpy().astype('uint16')
                all_detection = [[], []]
                for det in dets:
                    all_detection[det[4]].append(np.array((*det[:4], )))
                    im2show = add_bbox(im2show, det[:4], det[4], pascal_classes)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        result_path = os.path.join('/home/palm/PycharmProjects/algea/dataset/rcnn_plotted/0', file + "_det.jpg")
        cv2.imwrite(result_path, im2show)
