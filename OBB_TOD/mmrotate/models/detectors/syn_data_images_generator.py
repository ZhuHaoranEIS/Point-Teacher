# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
import torch
import torchvision
from mmcv.ops import nms_rotated


import torch.nn.functional as F
import random
from scipy.ndimage import rotate

from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import mmcv
from matplotlib.collections import PatchCollection
from mmdet.core.visualization import palette_val
from mmdet.core.visualization.image import draw_labels, draw_masks

from mmrotate.core.visualization.palette import get_palette
import torchvision.transforms.functional as TF


EPS = 1e-2

def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def draw_rbboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw oriented bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 5).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        xc, yc, w, h, ag = bbox[:5]
        wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
        hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        poly = np.int0(np.array([p1, p2, p3, p4]))
        polygons.append(Polygon(poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax


def imshow_det_rbboxes(img,
                       bboxes=None,
                       labels=None,
                       segms=None,
                       class_names=None,
                       score_thr=0,
                       bbox_color='green',
                       text_color='green',
                       mask_color=None,
                       thickness=2,
                       font_size=13,
                       win_name='',
                       show=True,
                       wait_time=0,
                       out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or
            (n, 6).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 5 or bboxes.shape[1] == 6, \
        f' bboxes.shape[1] should be 5 or 6, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 6
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_rbboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = bboxes[:, 2] * bboxes[:, 3]
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 5] if bboxes.shape[1] == 6 else None
        # draw_labels(
        #     ax,
        #     labels[:num_bboxes],
        #     positions,
        #     scores=scores,
        #     class_names=class_names,
        #     color=text_colors,
        #     font_size=font_size,
        #     scales=scales,
        #     horizontal_alignment=horizontal_alignment)

    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)

        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            draw_labels(
                ax,
                labels[num_bboxes:],
                positions,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def get_pattern_fill(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w))
    cv2.line(p, (0, 0), (0, h - 1), 0.01, 1)
    cv2.line(p, (0, 0), (w - 1, 0), 0.01, 1)
    cv2.line(p, (w - 1, h - 1), (0, h - 1), 0.01, 1)
    cv2.line(p, (w - 1, h - 1), (w - 1, 0), 0.01, 1)
    return p


def get_pattern_line(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w))
    interval_range = [3, 6]
    xn = np.random.randint(*interval_range)
    yn = np.random.randint(*interval_range)
    for i in range(xn):
        x = np.intp(np.round((w - 1) * i / (xn - 1)))
        cv2.line(p, (x, 0), (x, h), 0.5, 1)
    for i in range(yn):
        y = np.intp(np.round((h - 1) * i / (yn - 1)))
        cv2.line(p, (0, y), (w, y), 0.5, 1)
    return torch.from_numpy(p)


def get_pattern_rose(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w), dtype=float)
    t = np.float32(range(100))
    omega_range = [2, 4]
    xn = np.random.randint(*omega_range)
    x = np.sin(t / 99 * 2 * np.pi) * np.cos(
        t / 100 * 2 * np.pi * xn) * w / 2 + w / 2
    y = np.cos(t / 99 * 2 * np.pi) * np.cos(
        t / 100 * 2 * np.pi * 2) * h / 2 + h / 2
    xy = np.stack((x, y), -1)
    cv2.polylines(p, np.int_([xy]), True, 0.5, 1)
    return torch.from_numpy(p)


def get_pattern_li(w, h):
    w, h = int(w), int(h)
    p = np.ones((h, w), dtype=float)
    t = np.float32(range(100))
    s = np.random.rand() * 8
    s2 = np.random.rand() * 0.5 + 0.1
    r = (np.abs(np.cos(t / 99 * 4 * np.pi))**s) * (1 - s2) + s2
    x = r * np.sin(t / 99 * 2 * np.pi) * w / 2 + w / 2
    y = r * np.cos(t / 99 * 2 * np.pi) * h / 2 + h / 2
    xy = np.stack((x, y), -1)
    cv2.fillPoly(p, np.int_([xy]), 1)
    cv2.polylines(p, np.int_([xy]), True, 0.5, 1)
    return torch.from_numpy(p)


def obb2xyxy(obb):
    w = obb[:, 2]
    h = obb[:, 3]
    a = obb[:, 4]
    cosa = torch.cos(a).abs()
    sina = torch.sin(a).abs()
    dw = cosa * w + sina * h
    dh = sina * w + cosa * h
    dx = obb[..., 0]
    dy = obb[..., 1]
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


def plot_one_rotated_box(img,
                         obb,
                         color=[0.0, 0.0, 128],
                         label=None,
                         line_thickness=None):
    width, height, theta = obb[2], obb[3], obb[4] / np.pi * 180
    if theta < 0:
        width, height, theta = height, width, theta + 90
    rect = [(obb[0], obb[1]), (width, height), theta]
    poly = np.intp(np.round(
        cv2.boxPoints(rect)))  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    cv2.drawContours(
        image=img, contours=[poly], contourIdx=-1, color=color, thickness=2)


def get_pattern_gaussian(w, h, device):
    w, h = int(w), int(h)
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij')
    y = (y - h / 2) / (h / 2)
    x = (x - w / 2) / (w / 2)
    ox, oy = torch.randn(2, device=device).clip(-3, 3) * 0.15
    sx, sy = torch.rand(2, device=device) + 0.3
    z = torch.exp(-((x - ox) * sx)**2 - ((y - oy) * sy)**2) * 0.9 + 0.1
    return z


def generate_sythesis(img, bb_occupied, img_syn, sca_fact, pattern, prior_size,
                      dense_cls, imgsize):
    device = img.device
    cen_range = [50, imgsize - 50]

    base_scale = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * sca_fact
    base_scale = torch.exp(base_scale)

    bb_occupied = bb_occupied.clone()
    bb_occupied[:, 2] = prior_size[bb_occupied[:, 6].long(), 0] * 0.7
    bb_occupied[:, 3] = prior_size[bb_occupied[:, 6].long(), 0] * 0.7
    bb_occupied[:, 4] = 0

    bb = []
    palette = [
        torch.zeros(0, 6, device=device) for _ in range(len(prior_size))
    ]
    adjboost = 2
    for b in bb_occupied:
        x, y = torch.rand(
            2, device=device) * (cen_range[1] - cen_range[0]) + cen_range[0]
        dw = prior_size[b[6].long(), 2]
        w = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * dw
        w = base_scale * torch.exp(w)
        dr = prior_size[b[6].long(), 3]
        r = (torch.randn(1, device=device) * 0.4).clamp(-1, 1) * dr
        h = w * torch.exp(r)
        w *= prior_size[b[6].long(), 0]
        h *= prior_size[b[6].long(), 1]
        a = torch.rand(1, device=device) * torch.pi
        x = x.clip(0.71 * w, imgsize - 1 - 0.71 * w)
        y = y.clip(0.71 * h, imgsize - 1 - 0.71 * h)
        bb.append([x, y, w, h, a, (w * h) / imgsize / imgsize + 0.1, b[6]])

        bx = torch.clip(b[0:2], 16, imgsize - 1 - 16).long()
        nbr0 = img[:, bx[1] - 2:bx[1] + 3, bx[0] - 2:bx[0] + 3].reshape(3, -1)
        nbr1 = img[:, bx[1] - 16:bx[1] + 17,
                   bx[0] - 16:bx[0] + 17].reshape(3, -1)
        c0 = nbr0.mean(1)
        c1 = nbr1[:, (nbr1.mean(0) - c0.mean()).abs().max(0)[1]]
        c = torch.cat((c0, c1), 0)
        palette[b[6].long()] = torch.cat((palette[b[6].long()], c[None]), 0)

        if np.random.random() < 0.2 and adjboost > 0:
            adjboost -= 1
            if b[6].long() in dense_cls:
                itv, dev = torch.rand(
                    1, device=device) * 4 + 2, torch.rand(
                        1, device=device) * 8 - 4
                ofx = (h + itv) * torch.sin(-a) + dev * torch.cos(a)
                ofy = (h + itv) * torch.cos(a) + dev * torch.sin(a)
                for k in range(1, 6):
                    bb.append([
                        x + k * ofx, y + k * ofy, w, h, a,
                        (w * h) / imgsize / imgsize + 0.1 - 0.001 * k, b[6]
                    ])
            else:
                itv, dev = torch.rand(
                    1, device=device) * 40 + 10, torch.rand(
                        1, device=device) * 0
                ofx = (h + itv) * torch.sin(-a) + dev * torch.cos(a)
                ofy = (h + itv) * torch.cos(a) + dev * torch.sin(a)
                for k in range(1, 4):
                    bb.append([
                        x + k * ofx, y + k * ofy, w, h, a,
                        (w * h) / imgsize / imgsize + 0.1 - 0.001 * k, b[6]
                    ])

    bb = torch.tensor(bb)
    bb = torch.cat((bb_occupied, bb), 0)
    _, keep = nms_rotated(bb[:, 0:5], bb[:, 5], 0.05)
    bb = bb[keep]
    bb = bb[bb[:, 5] < 1]

    xyxy = obb2xyxy(bb)
    mask = torch.logical_and(
        xyxy.min(-1)[0] >= 0,
        xyxy.max(-1)[0] <= imgsize - 1)
    bb, xyxy = bb[mask], xyxy[mask]

    for i in range(len(bb)):
        cx, cy, w, h, t, s, c = bb[i, :7]
        ox, oy = torch.floor(xyxy[i, 0:2]).long()
        ex, ey = torch.ceil(xyxy[i, 2:4]).long()
        sx, sy = ex - ox, ey - oy
        theta = torch.tensor(
            [[1 / w * torch.cos(t), 1 / w * torch.sin(t), 0],
             [1 / h * torch.sin(-t), 1 / h * torch.cos(t), 0]],
            dtype=torch.float,
            device=device)
        theta[:, :2] @= torch.tensor([[sx, 0], [0, sy]],
                                     dtype=torch.float,
                                     device=device)
        grid = torch.nn.functional.affine_grid(
            theta[None], (1, 1, sy, sx), align_corners=False)
        p = pattern[c.long()]
        p = p[np.random.randint(0, len(p))][None].clone()
        trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ])
        if np.random.random() < 0.2:
            p *= get_pattern_line(p.shape[2], p.shape[1]).unsqueeze(0).unsqueeze(-1)
        if np.random.random() < 0.2:
            p *= get_pattern_rose(p.shape[2], p.shape[1]).unsqueeze(0).unsqueeze(-1)
        if np.random.random() < 0.2:
            p *= get_pattern_li(p.shape[2], p.shape[1]).unsqueeze(0).unsqueeze(-1)
        p = trans(p)
        p_in = torch.ones(grid.shape[0], grid.shape[1], grid.shape[2], 3)
        # import pdb; pdb.set_trace()
        p_in[:,:,:,0] = torch.nn.functional.grid_sample(
            p[:,:,:,0][None], grid, align_corners=False, mode='nearest')[0]
        p_in[:,:,:,1] = torch.nn.functional.grid_sample(
            p[:,:,:,1][None], grid, align_corners=False, mode='nearest')[0]
        p_in[:,:,:,2] = torch.nn.functional.grid_sample(
            p[:,:,:,2][None], grid, align_corners=False, mode='nearest')[0]
        p_in = p_in.squeeze(0).permute(2,0,1)
        if np.random.random() < 0.9:
            a = get_pattern_gaussian(sx, sy, device) * (p_in != 0)
        else:
            a = (p_in != 0).float()
        img_syn[:, oy:oy + sy,
            ox:ox + sx] = (1 - a) * img_syn[:, oy:oy + sy, ox:ox + sx] + a * p_in
    return img_syn, bb


def load_basic_pattern(path):
    image_name = os.listdir(path)
    image_absolute_path = [os.path.join(path, name) for name in image_name]
    prior_size = []
    pattern = []
    for image_path in image_absolute_path:
        image = cv2.imread(image_path)
        prior_size.append([image.shape[0], image.shape[1], 0.2, 0.2])
        p = []
        p.append(torch.from_numpy(image).float())
        pattern.append(p)
    prior_size = torch.Tensor(prior_size)
    return pattern, prior_size


if __name__ == '__main__':
    pattern, prior_size = load_basic_pattern(
        '/home/zhuhaoran/SODA/mmrotate/models/detectors/basic_patterns/basic_bank')

    img = cv2.imread('/data/zhr/SODA/SODA-A/divData/train/Images_filter/00001__800__3900___0.jpg')
    img = torch.from_numpy(img).permute(2, 0, 1).float().contiguous()
    # import pdb; pdb.set_trace()
    bb = torch.tensor(((318, 84, 0, 0, 0.5, 17, 0), ) * 10)
    import time
    c = time.time()
    # img, bb1 = generate_sythesis(img, bb, 1)
    # print(time.time() - c)
    # bb1[:, 5] = 1
    c = time.time()
    img, bb2 = generate_sythesis(img, bb, img, 1.0, pattern, prior_size, range(len(pattern)), 800)
    # import pdb; pdb.set_trace()
    print(time.time() - c)
    img = img.permute(1, 2, 0).contiguous().cpu().numpy()

    cv2.imwrite('/home/zhuhaoran/SODA/mmrotate/models/detectors/vis_points/P2RB_sys.png', img)

    coordinates_flip = bb2[:,:5]
    imshow_det_rbboxes('/home/zhuhaoran/SODA/mmrotate/models/detectors/vis_points/P2RB_sys.png',
                       bboxes=coordinates_flip.numpy(),
                       labels=np.ones(len(coordinates_flip)).astype(int),
                       bbox_color='red',
                       text_color='red',
                       out_file='/home/zhuhaoran/SODA/mmrotate/models/detectors/vis_points/P2RB_sys.png')

    # print("Original Image Shape:", image.shape)
    # print("Scaled Image Shape:", flip_image.shape)
    ### 地面
    base_path = '/data/zhr/SODA/SODA-A/divData/train/Images_filter/'
    '00299__800__650___1950.jpg'
    '00302__800__1300___1300.jpg'
    '01423__800__1300___650.jpg'
    '01752__800__3900___650.jpg'
    '00520__800__3250___650.jpg'
    '00108__800__0___650.jpg'
    ### 海面
    base_path = '/data/xuchang/AI-TOD/train/'
    '7a03dc1c7.png'
    '7a9708edd.png'
    '7ac3a346a.png'
    '7ba49f141.png'
    '7beed9136.png'
    '7c4b06990.png'
    


