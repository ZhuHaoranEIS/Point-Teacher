import torch
from kornia.geometry.transform import warp_perspective
from cv2 import warpPerspective as applyH
import numpy as np 
from scipy.ndimage.filters import maximum_filter
import torch

def apply_nms(score_map, size):

    score_map = score_map * (score_map == maximum_filter(score_map, footprint=np.ones((size, size))))

    return score_map


def remove_borders(images, borders):
    ## input [B,C,H,W]
    shape = images.shape

    if len(shape) == 4:
        for batch_id in range(shape[0]):
            images[batch_id, :, 0:borders, :] = 0
            images[batch_id, :, :, 0:borders] = 0
            images[batch_id, :, shape[2] - borders:shape[2], :] = 0
            images[batch_id, :, :, shape[3] - borders:shape[3]] = 0
    elif len(shape) == 2:
        images[ 0:borders, :] = 0
        images[ :, 0:borders] = 0
        images[ shape[0] - borders:shape[0], :] = 0
        images[ :, shape[1] - borders:shape[1]] = 0            
    else:
        print("Not implemented")
        exit()

    return images

def create_common_region_masks(h_dst_2_src, shape_src, shape_dst):

    # Create mask. Only take into account pixels in the two images
    inv_h = np.linalg.inv(h_dst_2_src)
    inv_h = inv_h / inv_h[2, 2]

    # Applies mask to destination. Where there is no 1, we can no find a point in source.
    ones_dst = np.ones((shape_dst[0], shape_dst[1]))
    ones_dst = remove_borders(ones_dst, borders=15)
    mask_src = applyH(ones_dst, h_dst_2_src, (shape_src[1], shape_src[0]))
    mask_src = np.where(mask_src >= 0.75, 1.0, 0.0)
    mask_src = remove_borders(mask_src, borders=15)

    ones_src = np.ones((shape_src[0], shape_src[1]))
    ones_src = remove_borders(ones_src, borders=15)
    mask_dst = applyH(ones_src, inv_h, (shape_dst[1], shape_dst[0]))
    mask_dst = np.where(mask_dst >= 0.75, 1.0, 0.0)
    mask_dst = remove_borders(mask_dst, borders=15)

    return mask_src, mask_dst


def prepare_homography(hom):
    if len(hom.shape) == 1:
        h = np.zeros((3, 3))
        for j in range(3):
            for i in range(3):
                if j == 2 and i == 2:
                    h[j, i] = 1.
                else:
                    h[j, i] = hom[j * 3 + i]
    elif len(hom.shape) == 2:  ## batch
        ones = torch.ones(hom.shape[0]).unsqueeze(1)
        h = torch.cat([hom, ones], dim=1).reshape(-1, 3, 3).type(torch.float32)

    return h


def getAff(x,y,H):
    h11 = H[0,0]
    h12 = H[0,1]
    h13 = H[0,2]
    h21 = H[1,0]
    h22 = H[1,1]
    h23 = H[1,2]
    h31 = H[2,0]
    h32 = H[2,1]
    h33 = H[2,2]
    fxdx = h11 / (h31 * x + h32 * y + h33) - (h11 * x + h12 * y + h13) * h31 / (h31 * x + h32 * y + h33) ** 2
    fxdy = h12 / (h31 * x + h32 * y + h33) - (h11 * x + h12 * y + h13) * h32 / (h31 * x + h32 * y + h33) ** 2

    fydx = h21 / (h31 * x + h32 * y + h33) - (h21 * x + h22 * y + h23) * h31 / (h31 * x + h32 * y + h33) ** 2
    fydy = h22 / (h31 * x + h32 * y + h33) - (h21 * x + h22 * y + h23) * h32 / (h31 * x + h32 * y + h33) ** 2

    Aff = [[fxdx, fxdy], [fydx, fydy]]

    return np.asarray(Aff)

def apply_homography_to_points(points, h):
    new_points = []

    for point in points:

        new_point = h.dot([point[0], point[1], 1.0])

        tmp = point[2]**2+np.finfo(np.float32).eps

        Mi1 = [[1/tmp, 0], [0, 1/tmp]]
        Mi1_inv = np.linalg.inv(Mi1)
        Aff = getAff(point[0], point[1], h)

        BMB = np.linalg.inv(np.dot(Aff, np.dot(Mi1_inv, np.matrix.transpose(Aff))))

        [e, _] = np.linalg.eig(BMB)
        new_radious = 1/((e[0] * e[1])**0.5)**0.5

        new_point = [new_point[0] / new_point[2], new_point[1] / new_point[2], new_radious, point[3]]
        new_points.append(new_point)

    return np.asarray(new_points)


def find_index_higher_scores(map, num_points = 1000, threshold = -1):
    # Best n points
    if threshold == -1:

        flatten = map.flatten() 
        order_array = np.sort(flatten) 

        order_array = np.flip(order_array, axis=0)

        if order_array.shape[0] < num_points:
            num_points = order_array.shape[0]

        threshold = order_array[num_points-1]

        if threshold <= 0.0:
            ### This is the problem case which derive smaller number of keypoints than the argument "num_points".
            indexes = np.argwhere(order_array > 0.0)

            if len(indexes) == 0:
                threshold = 0.0
            else:
                threshold = order_array[indexes[len(indexes)-1]]

    indexes = np.argwhere(map >= threshold)

    return indexes[:num_points]


def get_point_coordinates(map, scale_value=1., num_points=1000, threshold=-1, order_coord='xysr'):
    ## input numpy array score map : [H, W]
    indexes = find_index_higher_scores(map, num_points=num_points, threshold=threshold)
    new_indexes = []
    for ind in indexes:

        scores = map[ind[0], ind[1]]
        if order_coord == 'xysr':
            tmp = [ind[1], ind[0], scale_value, scores]
        elif order_coord == 'yxsr':
            tmp = [ind[0], ind[1], scale_value, scores]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes)


def get_point_coordinates3D(map, scale_factor=1., up_levels=0, num_points=1000, threshold=-1, order_coord='xysr'):

    indexes = find_index_higher_scores(map, num_points=num_points, threshold=threshold)
    new_indexes = []
    for ind in indexes:
        scale_value = (scale_factor ** (ind[2] - up_levels))
        scores = map[ind[0], ind[1], ind[2]]
        if order_coord == 'xysr':
            tmp = [ind[1], ind[0], scale_value, scores]
        elif order_coord == 'yxsr':
            tmp = [ind[0], ind[1], scale_value, scores]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes)

def align_ori_maps_spatially(src_ori, dst_ori, h_dst_2_src):
    ## rotate target image toward source image
    # warning : angle in rotate function is counter clockwise
    _, _, sh, sw = src_ori.shape
    dst_ori_aligned = warp_perspective(dst_ori, h_dst_2_src, (sh, sw), align_corners=True) 

    ## make mask for rotated histogram channel wise. For count the number of valid pixels.
    # valid: 1, invalid: 0
    b, c, dh, dw = dst_ori.shape
    mask = torch.ones(b, 1, dh, dw).to(src_ori.device)
    mask = warp_perspective(mask, h_dst_2_src, (sh, sw), align_corners=True)
    mask = mask > 0  ##  valid: 1, invalid: 0

    return src_ori, dst_ori_aligned, mask

def compute_orientation_acc(src_ori, dst_ori, ang_src_2_dst, h_dst_2_src, tolerance=1):
    ## src_ori/dst_ori : (B, G, H, W), ang_src_2_dst: (B, 1), h_dst_2_src: (B, 3, 3)

    B, G, _, _  = src_ori.shape
    step = 360 // G   
    ori_threshold = step * tolerance  ## for approx acc (error difference one bin tolerance.)
    d_threshold = int(ori_threshold // step) 

    ## Compute the pred angle difference
    src_ori, dst_ori_aligned, mask = align_ori_maps_spatially(src_ori, dst_ori, h_dst_2_src)
    src_ang, dst_ang = _histograms_to_angle_values(src_ori, dst_ori_aligned)

    pred_ang_diff = ((dst_ang - src_ang + G) % G) * mask.squeeze(1)
    pred_ang_diff = torch.flatten(pred_ang_diff, start_dim=1)
    mask = torch.flatten(mask, start_dim=1)  ## invalid region mask

    ## Compute the accuracies.
    total_cnt = mask.sum(1)
    correct_cnt = _compute_correct_orientations(pred_ang_diff, ang_src_2_dst, mask, step, G)

    approx_cnt = 0
    for i in range(-d_threshold, d_threshold+1):
        approx_cnt += _compute_correct_orientations(pred_ang_diff, ang_src_2_dst, mask, step, G, i)

    return correct_cnt/total_cnt, approx_cnt/total_cnt


def _histograms_to_angle_values(src_ori, dst_ori_aligned):
    src_ang = obtain_orientation_value_map(src_ori, mode='argmax') 
    dst_ang = obtain_orientation_value_map(dst_ori_aligned, mode='argmax')  
    return src_ang, dst_ang


def _compute_correct_orientations(pred_ang_diff, GT_ang_diff, mask, step, G, toler=0):
    ## G is orientation bin size. (the order of group)
    GT_bin_diff = ((GT_ang_diff // step) + G + toler) % G
    GT_bin_diff = mask * GT_bin_diff
    return (torch.where(pred_ang_diff ==  GT_bin_diff, 1, 0) * mask ).sum(1)


########### for inference

def obtain_orientation_value_map(feat, mode='argmax'):
    ## input : B, C, H, W,  output : B, H, W
    B, C, H, W = feat.shape
    
    if mode == 'argmax':
        res = torch.argmax(feat, dim=1)
    elif mode == 'soft_argmax':
        index_kernel = torch.linspace(0, C-1, C).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(feat.device)
        # res = torch.mul(torch.softmax(feat,dim=1), index_kernel)
        res = torch.mul(feat, index_kernel)
        res = res.mean(dim=1)

    return res


def compute_orientation_loss(src_ori, dst_ori, ang_src_2_dst, h_dst_2_src):
    step = 360 // src_ori.shape[1]

    ## spatial alignment
    src_ori, dst_ori_spatial_aligned, mask = align_ori_maps_spatially(src_ori, dst_ori, h_dst_2_src)

    ## histogram alignment
    src_ori_histo_aligned = torch.zeros_like(src_ori)
    for i in range(0, len(ang_src_2_dst)):
        d = int(ang_src_2_dst[i] / step)
        src_ori_histo_aligned[i] = torch.roll(src_ori[i], d, dims=0)
    src_ori_histo_aligned *= mask

    ## calculate loss 
    loss_map = calculate_loss(src_ori_histo_aligned, dst_ori_spatial_aligned)
    loss = loss_map.sum() / mask.sum()

    return loss, loss_map


def calculate_loss(src_hist, dst_hist, eps=1e-7):
    loss_map = -(src_hist * torch.log(dst_hist + eps)).sum(1) 
    
    return loss_map


def distribution_ssl_orientation_loss(src_ori, dst_ori, ang_src_2_dst, eps=1e-7):
    num_samples = src_ori.shape[0]
    step = 360 // src_ori.shape[1]

    ## histogram alignment
    if ang_src_2_dst == 'horizontal' or ang_src_2_dst == 'vertical':
        ang_src_2_dst = 180

    d = int(ang_src_2_dst / step)
    src_ori = torch.roll(src_ori, d, dims=0)
    loss = -(src_ori * torch.log(dst_ori + eps)).sum(1) 
    loss = loss.sum() / num_samples

    return loss


def calculate_loss(src_hist, dst_hist, eps=1e-7):
    loss_map = -(src_hist * torch.log(dst_hist + eps)).sum(1) 
    
    return loss_map


