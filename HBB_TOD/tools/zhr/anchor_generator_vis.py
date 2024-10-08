import numpy as np
from mmcv.visualization import imshow_bboxes
import matplotlib.pyplot as plt
from mmdet.core import build_anchor_generator
import cv2

if __name__ == '__main__':
    anchor_generator_cfg = dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])
    # anchor_generator_cfg = dict(
    #     type='AnchorGenerator',
    #     octave_base_scale=4,
    #     scales_per_octave=3,
    #     ratios=[0.5, 1.0, 2.0],
    #     strides=[8, 16, 32, 64, 128])
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    # 输出原图尺度上 anchor 坐标 xyxy 左上角格式
    # base_anchors 长度为5，表示5个输出特征图，不同的特征图尺度相差的只是 strides
    # 故我们取 strides=8 的位置 anchor 可视化即可
    base_anchors = anchor_generator.base_anchors[0]

    h = 100
    w = 160
    img = np.ones([h, w, 3], np.uint8) * 255
    base_anchors[:, 0::2] += w // 2
    base_anchors[:, 1::2] += h // 2

    colors = ['green', 'red', 'blue']
    for i in range(3):
        base_anchor = base_anchors[i::3, :].cpu().numpy()
        imshow_bboxes(img, base_anchor, show=False, colors=colors[i])
    cv2.imwrite("/home/zhr/mmdet-rfla/tools/zhr/test.png", img)
    # img.savefig("/home/zhr/mmdet-rfla/tools/zhr/test.png")
