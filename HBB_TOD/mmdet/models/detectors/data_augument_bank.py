import copy
import math
import torch

from torch.nn.functional import grid_sample
from torchvision import transforms

import os
import cv2
import numpy as np

import torch.nn.functional as F
import random
from scipy.ndimage import rotate

from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import mmcv
from matplotlib.collections import PatchCollection
import torchvision.transforms.functional as TF

EPS = 1e-2

def bbox_flip(bboxes, img_shape, direction='horizontal', version='oc'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 5*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    # version = 'oc'
    assert bboxes.shape[-1] % 5 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
    elif direction == 'vertical':
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    else:
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    if version == 'oc':
        rotated_flag = (bboxes[:, 4] != np.pi / 2)
        flipped[rotated_flag, 4] = np.pi / 2 - bboxes[rotated_flag, 4]
        flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
        flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
    else:
        flipped[:, 4] = norm_angle(np.pi - bboxes[:, 4], version)
    return flipped

def point_flip(bboxes, img_shape, direction='horizontal', version='oc'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 2*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 2 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
    elif direction == 'vertical':
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    else:
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    return flipped


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes[:, :4] = new_bboxes[:, :4] / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def rbbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): shape (n, 6)
        labels (torch.Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def poly2obb(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_oc(polys)
    elif version == 'le135':
        results = poly2obb_le135(polys)
    elif version == 'le90':
        results = poly2obb_le90(polys)
    else:
        raise NotImplementedError
    return results


def poly2obb_np(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_np_oc(polys)
    elif version == 'le135':
        results = poly2obb_np_le135(polys)
    elif version == 'le90':
        results = poly2obb_np_le90(polys)
    else:
        raise NotImplementedError
    return results


def obb2hbb(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    if version == 'oc':
        results = obb2hbb_oc(rbboxes)
    elif version == 'le135':
        results = obb2hbb_le135(rbboxes)
    elif version == 'le90':
        results = obb2hbb_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly_np(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_np_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_np_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_np_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2xyxy(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    if version == 'oc':
        results = obb2xyxy_oc(rbboxes)
    elif version == 'le135':
        results = obb2xyxy_le135(rbboxes)
    elif version == 'le90':
        results = obb2xyxy_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def hbb2obb(hbboxes, version='oc'):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
        version (Str): angle representations.

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = hbb2obb_oc(hbboxes)
    elif version == 'le135':
        results = hbb2obb_le135(hbboxes)
    elif version == 'le90':
        results = hbb2obb_le90(hbboxes)
    else:
        raise NotImplementedError
    return results


def poly2obb_oc(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    points = torch.reshape(polys, [-1, 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
    _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
                    points[:, 1, 1] - points[:, 0, 1]),
        axis=1)
    odd = torch.eq(torch.remainder((_thetas / (np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
    return rbboxes


def poly2obb_le135(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le135')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_le90(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le90')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a


def poly2obb_np_le135(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    if edge1 < 2 or edge2 < 2:
        return
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, 'le135')
    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return x_ctr, y_ctr, width, height, angle


def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


def obb2poly_oc(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


def obb2poly_le135(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2hbb_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,pi/2]
    """
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    hbboxes = rbboxes.clone().detach()
    hbboxes[:, 2::5] = hbbox_h
    hbboxes[:, 3::5] = hbbox_w
    hbboxes[:, 4::5] = np.pi / 2
    return hbboxes


def obb2hbb_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
    edges1 = torch.abs(bboxes[:, 2] - bboxes[:, 0])
    edges2 = torch.abs(bboxes[:, 3] - bboxes[:, 1])
    angles = bboxes.new_zeros(bboxes.size(0))
    inds = edges1 < edges2
    rotated_boxes = torch.stack((x_ctr, y_ctr, edges1, edges2, angles), dim=1)
    rotated_boxes[inds, 2] = edges2[inds]
    rotated_boxes[inds, 3] = edges1[inds]
    rotated_boxes[inds, 4] = np.pi / 2.0
    return rotated_boxes


def obb2hbb_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    hbboxes = torch.cat([center - bias, center + bias], dim=-1)
    _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    _w = hbboxes[..., 2] - hbboxes[..., 0]
    _h = hbboxes[..., 3] - hbboxes[..., 1]
    _theta = theta.new_zeros(theta.size(0))
    obboxes1 = torch.stack([_x, _y, _w, _h, _theta], dim=-1)
    obboxes2 = torch.stack([_x, _y, _h, _w, _theta - np.pi / 2], dim=-1)
    obboxes = torch.where((_w >= _h)[..., None], obboxes1, obboxes2)
    return obboxes


def hbb2obb_oc(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    rbboxes = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    return rbboxes


def hbb2obb_le135(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def hbb2obb_le90(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta - np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def obb2xyxy_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    # pi/2 >= a > 0, so cos(a)>0, sin(a)>0
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


def obb2xyxy_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    N = rotatex_boxes.shape[0]
    if N == 0:
        return rotatex_boxes.new_zeros((rotatex_boxes.size(0), 4))
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def obb2xyxy_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    # N = obboxes.shape[0]
    # if N == 0:
    #     return obboxes.new_zeros((obboxes.size(0), 4))
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center - bias, center + bias], dim=-1)


def obb2poly_np_oc(rbboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    score = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le135(rrects):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle, score = rrect[:6]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3, score],
                        dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')


def dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).

    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)


def gaussian2bbox(gmm):
    """Convert Gaussian distribution to polygons by SVD.

    Args:
        gmm (dict[str, torch.Tensor]): Dict of Gaussian distribution.

    Returns:
        torch.Tensor: Polygons.
    """
    try:
        from torch_batch_svd import svd
    except ImportError:
        svd = None
    L = 3
    var = gmm.var
    mu = gmm.mu
    assert mu.size()[1:] == (1, 2)
    assert var.size()[1:] == (1, 2, 2)
    T = mu.size()[0]
    var = var.squeeze(1)
    if svd is None:
        raise ImportError('Please install torch_batch_svd first.')
    U, s, Vt = svd(var)
    size_half = L * s.sqrt().unsqueeze(1).repeat(1, 4, 1)
    mu = mu.repeat(1, 4, 1)
    dx_dy = size_half * torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]],
                                     dtype=torch.float32,
                                     device=size_half.device)
    bboxes = (mu + dx_dy.matmul(Vt.transpose(1, 2))).reshape(T, 8)

    return bboxes


def gt2gaussian(target):
    """Convert polygons to Gaussian distributions.

    Args:
        target (torch.Tensor): Polygons with shape (N, 8).

    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    L = 3
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))




# def random_rescale_image(image_tensor, coordinates, gt_bboxes, scale_factor_uniform=[0.5,1.5], version='le90'):
#     """
#     Rescales the image tensor randomly with scale factor in (0.5, 1.5).
#     The final image size remains C*H*W.
    
#     image_tensor: tensor of shape (C, H, W)
#     coordinates: tensor of shape (M, 2) [cx, cy]
    
#     Returns the rescaled image tensor of shape (C, H, W).
#     coordinates_rescaled shape (M, 2)
#     ignore_gt (M, ) 1 to ignore the gt, 0 to not ignore
#     scale_factor: float
#     """
#     # Get the original dimensions
#     C, H, W = image_tensor.shape
#     num_bboxes, _ = coordinates.shape
#     coordinates_rescale = coordinates.clone()
#     ignore_gt = torch.zeros(num_bboxes).long().to(coordinates.device)

#     if gt_bboxes == None:
#         gt_bboxes_rescale = None
#     else:
#         gt_bboxes_rescale = gt_bboxes.clone()
#         gt_bboxes_rescale_poly = obb2poly(gt_bboxes_rescale, version=version)
    
#     # Generate a random scale factor between 0.5 and 1.5
#     index = np.around(np.arange(scale_factor_uniform[0], scale_factor_uniform[1]+0.1, 0.1), 1)
#     index = np.delete(index, np.where(index == 1.0))
#     prop = np.array([1/len(index) for i in range(len(index))])
#     if sum(prop) != 1.0:
#         prop[-1] += 1 - sum(prop)
#     scale_factor_H = np.random.choice(index,p=prop.ravel())
#     scale_factor_W = np.random.choice(index,p=prop.ravel())

#     # scale_factor = np.around(np.random.uniform(scale_factor_uniform[0], scale_factor_uniform[1]), 1)
#     # Calculate new dimensions
#     new_H = int(H * scale_factor_H)
#     new_W = int(W * scale_factor_W)

#     # Calculate the new coordinates
#     if scale_factor_H < 1.0:
#         blank_h = int((H - new_H) / 2)
#     else:
#         blank_h = int((new_H - H) / 2)
#     if scale_factor_W < 1.0:
#         blank_w = int((W - new_W) / 2)
#     else:
#         blank_w = int((new_W - W) / 2)

#     coordinates_rescale[:, 0] = (coordinates[:, 0] * scale_factor_W).long()
#     coordinates_rescale[:, 1] = (coordinates[:, 1] * scale_factor_H).long()
#     if gt_bboxes != None:
#         gt_bboxes_rescale_poly[:,[0,2,4,6]] *= scale_factor_W
#         gt_bboxes_rescale_poly[:,[1,3,5,7]] *= scale_factor_H

#     if scale_factor_W >= 1.0 and scale_factor_H >= 1.0:
#         outsider = (((coordinates_rescale[:, 0] >= 0) & (coordinates_rescale[:, 0] < blank_w)) |
#                     ((coordinates_rescale[:, 1] >= 0) & (coordinates_rescale[:, 1] < blank_h)) |
#                     ((coordinates_rescale[:, 0] >= (blank_w + W)) & (coordinates_rescale[:, 0] < new_W)) |
#                     ((coordinates_rescale[:, 1] >= (blank_h + H)) & (coordinates_rescale[:, 1] < new_H))).nonzero().reshape(-1)
#         ignore_gt[outsider] = 1
#         coordinates_rescale[:, 0] -= blank_w
#         coordinates_rescale[:, 1] -= blank_h
#         if gt_bboxes != None:
#             gt_bboxes_rescale_poly[:,[0,2,4,6]] -= blank_w
#             gt_bboxes_rescale_poly[:,[1,3,5,7]] -= blank_h
#     elif scale_factor_W >= 1.0 and scale_factor_H < 1.0:
#         outsider = (((coordinates_rescale[:, 0] >= 0) & (coordinates_rescale[:, 0] < blank_w)) |
#                     ((coordinates_rescale[:, 0] >= (blank_w + W)) & (coordinates_rescale[:, 0] < new_W))).nonzero().reshape(-1)
#         ignore_gt[outsider] = 1
#         coordinates_rescale[:, 0] -= blank_w
#         coordinates_rescale[:, 1] += blank_h
#         if gt_bboxes != None:
#             gt_bboxes_rescale_poly[:,[0,2,4,6]] -= blank_w
#             gt_bboxes_rescale_poly[:,[1,3,5,7]] += blank_h
#     elif scale_factor_W < 1.0 and scale_factor_H >= 1.0:
#         outsider = (((coordinates_rescale[:, 1] >= 0) & (coordinates_rescale[:, 1] < blank_h)) |
#                     ((coordinates_rescale[:, 1] >= (blank_h + H)) & (coordinates_rescale[:, 1] < new_H))).nonzero().reshape(-1)
#         ignore_gt[outsider] = 1
#         coordinates_rescale[:, 1] -= blank_h
#         coordinates_rescale[:, 0] += blank_w
#         if gt_bboxes != None:
#             gt_bboxes_rescale_poly[:,[0,2,4,6]] += blank_w
#             gt_bboxes_rescale_poly[:,[1,3,5,7]] -= blank_h
#     else:
#         coordinates_rescale[:, 0] += blank_w
#         coordinates_rescale[:, 1] += blank_h
#         if gt_bboxes != None:
#             gt_bboxes_rescale_poly[:,[0,2,4,6]] += blank_w
#             gt_bboxes_rescale_poly[:,[1,3,5,7]] += blank_h

#     # Resize the image
#     rescaled_image = F.interpolate(image_tensor.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
#     # rescaled_image = F.interpolate(image_tensor.unsqueeze(0), size=(new_H, new_W), mode='nearest').squeeze(0)

#     # Create an empty canvas
#     output_image = torch.zeros_like(image_tensor)
#     # Calculate the offsets for center alignment
#     if scale_factor_H < 1.0:
#         start_y = (H - new_H) // 2
#         end_y = start_y + new_H
#     else:
#         start_y = (new_H - H) // 2
#         end_y = start_y + H

#     if scale_factor_W < 1.0:
#         start_x = (W - new_W) // 2
#         end_x = start_x + new_W
#     else:
#         start_x = (new_W - W) // 2
#         end_x = start_x + W

#     if scale_factor_H < 1.0 and scale_factor_W < 1.0:
#         output_image[:, start_y:end_y, start_x:end_x] = rescaled_image
#     elif scale_factor_H >= 1.0 and scale_factor_W >= 1.0:
#         output_image = rescaled_image[:, start_y:end_y, start_x:end_x]
#     elif scale_factor_H >= 1.0 and scale_factor_W < 1.0:
#         output_image[:, :, start_x:end_x] = rescaled_image[:, start_y:end_y, :]
#     elif scale_factor_H < 1.0 and scale_factor_W >= 1.0:
#         output_image[:, start_y:end_y, :] = rescaled_image[:, :, start_x:end_x] 

#     # output_image = torch.round(output_image)

#     if gt_bboxes != None:
#         gt_bboxes_rescale = poly2obb(gt_bboxes_rescale_poly, version=version)

#     return tuple([output_image, coordinates_rescale, ignore_gt, [scale_factor_H, scale_factor_W], gt_bboxes_rescale])
    # return output_image, coordinates_rescale, ignore_gt
def random_rescale_image(image_tensor, coordinates, gt_bboxes, scale_factor_uniform=[0.5,1.5], version='le90'):
    """
    Rescales the image tensor randomly with scale factor in (0.5, 1.5).
    The final image size remains C*H*W.
    
    image_tensor: tensor of shape (C, H, W)
    coordinates: tensor of shape (M, 2) [cx, cy]
    
    Returns the rescaled image tensor of shape (C, H, W).
    coordinates_rescaled shape (M, 2)
    ignore_gt (M, ) 1 to ignore the gt, 0 to not ignore
    scale_factor: float
    """
    # Get the original dimensions
    C, H, W = image_tensor.shape
    num_bboxes, _ = coordinates.shape
    coordinates_rescale = coordinates.clone()
    ignore_gt = torch.zeros(num_bboxes).long().to(coordinates.device)

    # Generate a random scale factor between 0.5 and 1.5
    index = np.around(np.arange(scale_factor_uniform[0], scale_factor_uniform[1]+0.1, 0.1), 1)
    index = np.delete(index, np.where(index == 1.0))
    prop = np.array([1/len(index) for i in range(len(index))])
    if sum(prop) != 1.0:
        prop[-1] += 1 - sum(prop)
    scale_factor_H = np.random.choice(index,p=prop.ravel())
    scale_factor_W = scale_factor_H

    # scale_factor = np.around(np.random.uniform(scale_factor_uniform[0], scale_factor_uniform[1]), 1)
    # Calculate new dimensions
    new_H = int(H * scale_factor_H)
    new_W = int(W * scale_factor_W)

    # Calculate the new coordinates
    if scale_factor_H < 1.0:
        blank_h = int((H - new_H) / 2)
        blank_w = int((W - new_W) / 2)
    else:
        blank_h = int((new_H - H) / 2)
        blank_w = int((new_W - W) / 2)


    coordinates_rescale[:, 0] = (coordinates[:, 0] * scale_factor_W).long()
    coordinates_rescale[:, 1] = (coordinates[:, 1] * scale_factor_H).long()

    if scale_factor_W >= 1.0 and scale_factor_H >= 1.0:
        outsider = (((coordinates_rescale[:, 0] >= 0) & (coordinates_rescale[:, 0] < blank_w)) |
                    ((coordinates_rescale[:, 1] >= 0) & (coordinates_rescale[:, 1] < blank_h)) |
                    ((coordinates_rescale[:, 0] >= (blank_w + W)) & (coordinates_rescale[:, 0] < new_W)) |
                    ((coordinates_rescale[:, 1] >= (blank_h + H)) & (coordinates_rescale[:, 1] < new_H))).nonzero().reshape(-1)
        ignore_gt[outsider] = 1
        coordinates_rescale[:, 0] -= blank_w
        coordinates_rescale[:, 1] -= blank_h
    else:
        coordinates_rescale[:, 0] += blank_w
        coordinates_rescale[:, 1] += blank_h


    # Resize the image
    rescaled_image = F.interpolate(image_tensor.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)

    # Create an empty canvas
    output_image = torch.zeros_like(image_tensor)
    # Calculate the offsets for center alignment
    if scale_factor_H < 1.0:
        start_y = (H - new_H) // 2
        end_y = start_y + new_H
        start_x = (W - new_W) // 2
        end_x = start_x + new_W
    else:
        start_y = (new_H - H) // 2
        end_y = start_y + H
        start_x = (new_W - W) // 2
        end_x = start_x + W

    if scale_factor_H < 1.0 and scale_factor_W < 1.0:
        output_image[:, start_y:end_y, start_x:end_x] = rescaled_image
    elif scale_factor_H >= 1.0 and scale_factor_W >= 1.0:
        output_image = rescaled_image[:, start_y:end_y, start_x:end_x]

    # output_image = torch.round(output_image)

    if gt_bboxes == None:
        gt_bboxes_rescale = None
    else:
        gt_bboxes_rescale = gt_bboxes.clone()
        gt_bboxes_rescale[:,:2] = coordinates_rescale
        gt_bboxes_rescale[:,2:4] *= scale_factor_H

    output_image = torch.round(output_image)
    
    return tuple([output_image, coordinates_rescale, ignore_gt, scale_factor_H, gt_bboxes_rescale])

def fix_rescale_rbbox(image_tensor, gt_bboxes, insider, scale_factor_uniform, version='le90'):
    """
    Rescales the image tensor randomly with scale factor in (0.5, 1.5).
    The final image size remains C*H*W.
    
    image_tensor: tensor of shape (C, H, W)
    coordinates: tensor of shape (M, 2) [cx, cy]
    
    Returns the rescaled image tensor of shape (C, H, W).
    coordinates_rescaled shape (M, 2)
    ignore_gt (M, ) 1 to ignore the gt, 0 to not ignore
    scale_factor: float
    """
    # Get the original dimensions
    if gt_bboxes == None:
        return tuple([None])
    C, H, W = image_tensor.shape
    num_bboxes, _ = gt_bboxes.shape
    gt_bboxes_rescale = gt_bboxes.clone()

    scale_factor_H = scale_factor_uniform  
    scale_factor_W = scale_factor_uniform

    # Calculate new dimensions
    new_H = int(H * scale_factor_H)
    new_W = int(W * scale_factor_W)

    if scale_factor_H < 1.0:
        blank_h = int((H - new_H) / 2)
        blank_w = int((W - new_W) / 2)
    else:
        blank_h = int((new_H - H) / 2)
        blank_w = int((new_W - W) / 2)

    gt_bboxes_rescale[:, 0] = (gt_bboxes_rescale[:, 0] * scale_factor_W).long()
    gt_bboxes_rescale[:, 1] = (gt_bboxes_rescale[:, 1] * scale_factor_H).long()

    if scale_factor_W >= 1.0 and scale_factor_H >= 1.0:
        gt_bboxes_rescale[:, 0] -= blank_w
        gt_bboxes_rescale[:, 1] -= blank_h
    else:
        gt_bboxes_rescale[:, 0] += blank_w
        gt_bboxes_rescale[:, 1] += blank_h

    gt_bboxes_rescale[:,2:4] *= scale_factor_H
    pos_inds = (insider == 0).nonzero().reshape(-1)
    
    return tuple([gt_bboxes_rescale[pos_inds]])

def random_flip_image(image_tensor, coordinates, gt_bboxes, version='le90'):
    """
    Apply a random flip ('horizontal', 'vertical', or 'diagonal') to the image tensor.
    
    image_tensor: tensor of shape (C, H, W)
    
    Returns a tensor of the same shape (C, H, W) after the flip.
    """
    C, H, W = image_tensor.shape
    num_bboxes, _ = coordinates.shape
    ignore_gt = torch.zeros(num_bboxes).long().to(coordinates.device)
    

    flip_type = random.choice(['horizontal', 'vertical'])
    
    if flip_type == 'horizontal':
        flipped_tensor = torch.flip(image_tensor, dims=[2])
    elif flip_type == 'vertical':
        flipped_tensor = torch.flip(image_tensor, dims=[1])

    coordinates_flip = point_flip(coordinates, image_tensor.shape[1:3], direction=flip_type, version=version)
    
    # flipped_tensor = torch.round(flipped_tensor)
    
    if gt_bboxes == None:
        gt_bboxes_flip = None
    else:
        gt_bboxes_flip = bbox_flip(gt_bboxes, image_tensor.shape[1:3], direction=flip_type, version=version)

    flipped_tensor = torch.round(flipped_tensor)

    return tuple([flipped_tensor, coordinates_flip, ignore_gt, flip_type, gt_bboxes_flip])
    # return flipped_tensor, coordinates_flip, ignore_gt

def fix_flip_rbbox(image_tensor, gt_bboxes, insider, flip_type, version='le90'):
    """
    Apply a random flip ('horizontal', 'vertical', or 'diagonal') to the image tensor.
    
    image_tensor: tensor of shape (C, H, W)
    
    Returns a tensor of the same shape (C, H, W) after the flip.
    """
    if gt_bboxes == None:
        return tuple([None])
    pos_inds = (insider == 0).nonzero().reshape(-1)
    
    gt_bboxes_flip = bbox_flip(gt_bboxes, image_tensor.shape[1:3], direction=flip_type, version=version)

    return tuple([gt_bboxes_flip[pos_inds]])


def rotate_point(px, py, angle, cx, cy):
    """
    Rotate a point (px, py) around another point (cx, cy) by the given angle.
    Angle is in degrees and counter-clockwise.
    """
    radians = np.deg2rad(angle)
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)
    x_new = cos_a * (px - cx) - sin_a * (py - cy) + cx
    y_new = sin_a * (px - cx) + cos_a * (py - cy) + cy
    # import pdb; pdb.set_trace()
    return x_new.reshape(-1,1), y_new.reshape(-1,1)


def random_rotate_image(image_tensor, coordinates, gt_bboxes, angle_range=[1,360], version='le90'):
    """
    Rotate the image tensor by a random angle between 0 and 360 degrees.
    
    image_tensor: tensor of shape (C, H, W)
    
    Returns a tensor of the same shape (C, H, W) after the rotation.
    """
    # Randomly select an angle
    angle = np.random.randint(angle_range[0], angle_range[1])

    C, H, W = image_tensor.shape
    cx, cy = W / 2, H / 2  # center of the image
    num_bboxes, _ = coordinates.shape
    ignore_gt = torch.ones(num_bboxes).long().to(coordinates.device)
    
    # Initialize the rotated tensor
    rotated_image_tensor = TF.rotate(image_tensor, angle, fill=0)
    
    # rotated_image_tensor = torch.round(rotated_image_tensor)

    coordinates_poly = coordinates
    coordinates_poly_rotated = torch.zeros_like(coordinates_poly)

    if gt_bboxes == None:
        gt_bboxes_rotate = None
    else:
        bboxes_poly = obb2poly(gt_bboxes, version=version)
        bboxes_poly_rotated = torch.zeros_like(bboxes_poly)

    if gt_bboxes != None:
        x1, y1, x2, y2, x3, y3, x4, y4 = torch.chunk(bboxes_poly, 8, dim=1)
        # Rotate each corner
        x1_new, y1_new = rotate_point(x1, y1, -angle, cx, cy)
        x2_new, y2_new = rotate_point(x2, y2, -angle, cx, cy)
        x3_new, y3_new = rotate_point(x3, y3, -angle, cx, cy)
        x4_new, y4_new = rotate_point(x4, y4, -angle, cx, cy)

        bboxes_poly_rotated = torch.cat([x1_new, y1_new, x2_new, y2_new, x3_new, y3_new, x4_new, y4_new],dim=1)

    x1, y1 = torch.chunk(coordinates_poly, 2, dim=1)
    # Rotate each corner
    x1_new, y1_new = rotate_point(x1, y1, -angle, cx, cy)
    coordinates_poly_rotated = torch.cat([x1_new, y1_new],dim=1)
    x1_new = x1_new.reshape(-1)
    y1_new = y1_new.reshape(-1)
    insider = (((0 <= x1_new) & (x1_new < W)) & ((0 <= y1_new) & (y1_new < H))).nonzero().reshape(-1)
    ignore_gt[insider] = 0

    if gt_bboxes != None:
        gt_bboxes_rotate = poly2obb(bboxes_poly_rotated, version=version)

    rotated_image_tensor = torch.round(rotated_image_tensor)

    return tuple([rotated_image_tensor, coordinates_poly_rotated, ignore_gt, angle, gt_bboxes_rotate])
    # coordinates_obb_rotated = poly2obb(coordinates_poly_rotated, version=version)
    # return rotated_image_tensor, coordinates_obb_rotated, ignore_gt

def fix_rotate_rbbox(image_tensor, gt_bboxes, insider, angle, version='le90'):
    if gt_bboxes == None:
        return tuple([None])
    C, H, W = image_tensor.shape
    cx, cy = W / 2, H / 2  # center of the image
    num_bboxes, _ = gt_bboxes.shape
    
    pos_inds = (insider == 0).nonzero().reshape(-1)

    bboxes_poly = obb2poly(gt_bboxes, version=version)

    x1, y1, x2, y2, x3, y3, x4, y4 = torch.chunk(bboxes_poly, 8, dim=1)
    # Rotate each corner
    x1_new, y1_new = rotate_point(x1, y1, -angle, cx, cy)
    x2_new, y2_new = rotate_point(x2, y2, -angle, cx, cy)
    x3_new, y3_new = rotate_point(x3, y3, -angle, cx, cy)
    x4_new, y4_new = rotate_point(x4, y4, -angle, cx, cy)

    bboxes_poly_rotated = torch.cat([x1_new, y1_new, x2_new, y2_new, x3_new, y3_new, x4_new, y4_new],dim=1)

    gt_bboxes_rotate = poly2obb(bboxes_poly_rotated, version=version)

    return tuple([gt_bboxes_rotate[pos_inds]])


def translate_point(cx, cy, dx, dy):
    """
    Translate a point (cx, cy) by distance dx and dy.
    """
    return cx + dx, cy + dy

def random_translate_image(image_tensor, coordinates, gt_bboxes, angle_range=[1,360], ratio_range=[0.1,0.5], version='le90'):
    # Randomly select an angle
    angle = np.random.randint(angle_range[0], angle_range[1])
    k = np.random.uniform(ratio_range[0], ratio_range[1])
    
    C, H, W = image_tensor.shape
    d = min(H, W) * k
    radians = np.deg2rad(angle)
    dx = d * np.cos(radians)
    dy = d * np.sin(radians)

    cx, cy = W / 2, H / 2  # center of the image
    num_bboxes, _ = coordinates.shape
    ignore_gt = torch.zeros(num_bboxes).long().to(coordinates.device)

    # Initialize the translated image tensor with black
    # Use torchvision's affine to translate the image
    translated_image_tensor = TF.affine(image_tensor, angle=0, translate=[dx, dy], scale=1, shear=[0, 0], fill=0)

    num_bboxes, _ = coordinates.shape
    ignore_gt = torch.zeros(num_bboxes, dtype=torch.long, device=coordinates.device)

    # Translate the coordinates
    coordinates_translated = coordinates.clone()
    coordinates_translated[:, 0] += dx
    coordinates_translated[:, 1] += dy
    
    # Check if the translated coordinates are within the image
    ignore_gt = ((coordinates_translated[:, 0] < 0) | (coordinates_translated[:, 0] >= W) |
                 (coordinates_translated[:, 1] < 0) | (coordinates_translated[:, 1] >= H)).long()

    # Translate the bounding boxes if provided
    if gt_bboxes is not None:
        gt_bboxes_translate = gt_bboxes.clone()
        gt_bboxes_translate[:, 0] += dx  # translate x
        gt_bboxes_translate[:, 1] += dy  # translate y
    else:
        gt_bboxes_translate = None


    translated_image_tensor = torch.round(translated_image_tensor)
    
    return tuple([translated_image_tensor, coordinates_translated, ignore_gt, [angle, dx, dy], gt_bboxes_translate])
    # return translated_image_tensor, coordinates_translated, ignore_gt


def fix_translate_rbbox(image_tensor, gt_bboxes, insider, factor=[0,0,0], version='le90'):
    if gt_bboxes == None:
        return tuple([None])
    # Randomly select an angle
    angle = factor[0]
    dx = factor[1]
    dy = factor[2]

    pos_inds = (insider == 0).nonzero().reshape(-1)

    gt_bboxes_translate = gt_bboxes.clone()
    gt_bboxes_translate[:, 0] += dx  # translate x
    gt_bboxes_translate[:, 1] += dy  # translate y

    return tuple([gt_bboxes_translate[pos_inds]])


def random_point_in_quadrilateral(quads, position):
    # quads : xyxy
    M = quads.shape[0]
    widths = quads[:, 2] - quads[:, 0]
    heights = quads[:, 3] - quads[:, 1]

    space_widths = widths * (1 - position) / 2
    space_heights = heights * (1 - position) / 2

    random_widths = widths * position
    random_heights = heights * position

    random_x = torch.rand(M)
    random_y = torch.rand(M)
    
    random_x = random_x.to(quads.device).to(quads.dtype)
    random_y = random_y.to(quads.device).to(quads.dtype)

    random_points_x = quads[:, 0] + space_widths + random_x * random_widths
    random_points_y = quads[:, 1] + space_heights + random_y * random_heights

    random_points = torch.stack((random_points_x, random_points_y), dim=1)
    return random_points


def save_image(tensor, file_path):
    """
    Save a PyTorch tensor as an image file.

    Parameters:
    tensor (torch.Tensor): A tensor of size (C, H, W)
    file_path (str): Path where the image should be saved
    """
    # Convert tensor to PIL image
    to_pil = ToPILImage()
    image = to_pil(tensor)
    
    # Save image
    image.save(file_path)

def load_image(file_path):
    """
    Load an image from a file path and convert it to a PyTorch tensor.

    Parameters:
    file_path (str): Path to the image file

    Returns:
    torch.Tensor: The loaded image as a tensor of size (C, H, W)
    """
    image = Image.open(file_path).convert('RGB')
    to_tensor = ToTensor()
    return to_tensor(image)


def draw_rotated_boxes_plt(image_tensor, boxes_tensor):
    """
    Draws rotated boxes on the image tensor using matplotlib.
    
    image_tensor: tensor of shape (C, H, W)
    boxes_tensor: tensor of shape (M, 8), each box is (x1, y1, x2, y2, x3, y3, x4, y4)
    
    Returns a numpy array of the image with rotated boxes drawn on it.
    """
    C, H, W = image_tensor.shape
    
    # Convert image tensor to numpy array for plt
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    image_np = np.clip(image_np, 0, 1)  # Ensure values are in [0, 1] range
    
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(image_np)
    
    # Draw each box
    for i in range(boxes_tensor.shape[0]):
        corners = boxes_tensor[i].view(4, 2).numpy()
        polygon = Polygon(corners, closed=True, edgecolor='r', fill=False, linewidth=2)
        ax.add_patch(polygon)
    
    # Hide axes
    ax.axis('off')
    
    # Set limits to avoid white edges
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    
    # Draw canvas
    fig.canvas.draw()
    
    # Convert figure to numpy array
    img_with_boxes_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_with_boxes_np = img_with_boxes_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Close the figure
    plt.close(fig)
    
    return img_with_boxes_np





if __name__=='__main__':
    # Create a sample image tensor of size (3, 100, 100)
    soda_image_path = '/data/zhr/SODA/SODA-A/divData/train/Images_filter/00001__800__3900___0.jpg'
    save_ori_image_path = '/home/zhuhaoran/SODA/mmrotate/models/utils/image_operate/examples/original_image.png'
    save_new_image_path = '/home/zhuhaoran/SODA/mmrotate/models/utils/image_operate/examples/scaled_image.png'
    image = load_image(soda_image_path)
    coordinates = torch.Tensor([(20, 20, 7, 7, np.pi/6), (50, 50, 10, 10, np.pi/6), 
                                (450, 520, 30, 60, np.pi/4), (760, 410, 50, 60, np.pi/4)]).float()
    coordinates_xyxy = obb2poly(coordinates, version='le90')

    ### random resize
    # image_w_coord = draw_rotated_boxes_plt(image, coordinates_xyxy)
    # save_image(image_w_coord, '/home/zhuhaoran/SODA/mmrotate/models/utils/image_operate/examples/original_image.png')


    # resized_image, coordinates_rescale, ignore_gt = random_rescale_image(image, coordinates, scale_factor_uniform=[0.5,1.5], version='le90')
    # coordinates_rescale_xyxy = obb2poly(coordinates_rescale, version='le90')
    # insider = (ignore_gt!=1).nonzero().reshape(-1)
    # coordinates_rescale_xyxy = coordinates_rescale_xyxy[insider,:]
    # resized_image_w_coord = draw_rotated_boxes_plt(resized_image, coordinates_rescale_xyxy)

    # save_image(resized_image_w_coord, '/home/zhuhaoran/SODA/mmrotate/models/utils/image_operate/examples/scaled_image.png')
    # print("Original Image Shape:", image.shape)
    # print("Scaled Image Shape:", resized_image.shape)

    ### random flip 
    # save_image(image, save_ori_image_path)
    # imshow_det_rbboxes(save_ori_image_path,
    #                    bboxes=coordinates.numpy(),
    #                    labels=np.ones(len(coordinates)).astype(int),
    #                    bbox_color='red',
    #                    text_color='red',
    #                    out_file=save_ori_image_path)

    # flip_image, coordinates_flip, ignore_gt = random_flip_image(image, coordinates, version='le90')
    # save_image(flip_image, save_new_image_path)
    # insider = (ignore_gt != 1).nonzero().reshape(-1)
    # coordinates_flip = coordinates_flip[insider,:]

    # imshow_det_rbboxes(save_new_image_path,
    #                    bboxes=coordinates_flip.numpy(),
    #                    labels=np.ones(len(coordinates)).astype(int),
    #                    bbox_color='red',
    #                    text_color='red',
    #                    out_file=save_new_image_path)

    # print("Original Image Shape:", image.shape)
    # print("Scaled Image Shape:", flip_image.shape)

    ### random rotate
    # save_image(image, save_ori_image_path)
    # imshow_det_rbboxes(save_ori_image_path,
    #                    bboxes=coordinates.numpy(),
    #                    labels=np.ones(len(coordinates)).astype(int),
    #                    bbox_color='red',
    #                    text_color='red',
    #                    out_file=save_ori_image_path)

    # flip_image, coordinates_flip, ignore_gt = random_rotate_image(image, coordinates, angle_range=[1, 360], version='le90')
    # save_image(flip_image, save_new_image_path)
    # insider = (ignore_gt != 1).nonzero().reshape(-1)
    # coordinates_flip = coordinates_flip[insider,:]

    # imshow_det_rbboxes(save_new_image_path,
    #                    bboxes=coordinates_flip.numpy(),
    #                    labels=np.ones(len(coordinates)).astype(int),
    #                    bbox_color='red',
    #                    text_color='red',
    #                    out_file=save_new_image_path)

    # print("Original Image Shape:", image.shape)
    # print("Scaled Image Shape:", flip_image.shape)

    ### random translate
    save_image(image, save_ori_image_path)
    imshow_det_rbboxes(save_ori_image_path,
                       bboxes=coordinates.numpy(),
                       labels=np.ones(len(coordinates)).astype(int),
                       bbox_color='red',
                       text_color='red',
                       out_file=save_ori_image_path)
    # flip_image, _, ignore_gt, _, coordinates_flip = random_translate_image(image, coordinates=coordinates[:,:2], gt_bboxes=coordinates, angle_range=[1,360], ratio_range=[0.1,0.5], version='le90')
    # flip_image, _, ignore_gt, _, coordinates_flip = random_rotate_image(image, coordinates=coordinates[:,:2], gt_bboxes=coordinates, angle_range=[1,360], version='le90')
    flip_image, _, ignore_gt, _, coordinates_flip = random_rescale_image(image, coordinates=coordinates[:,:2], gt_bboxes=coordinates, scale_factor_uniform=[0.5,1.5], version='le90')
    save_image(flip_image, save_new_image_path)
    insider = (ignore_gt != 1).nonzero().reshape(-1)
    coordinates_flip = coordinates_flip[insider,:]

    imshow_det_rbboxes(save_new_image_path,
                       bboxes=coordinates_flip.numpy(),
                       labels=np.ones(len(coordinates)).astype(int),
                       bbox_color='red',
                       text_color='red',
                       out_file=save_new_image_path)

    print("Original Image Shape:", image.shape)
    print("Scaled Image Shape:", flip_image.shape)