# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from ..mask.structures import bitmap_to_polygon
from ..utils import mask2ndarray
from .palette import get_palette, palette_val


import math
__all__ = [
    'color_val_matplotlib', 'draw_masks', 'draw_bboxes', 'draw_labels',
    'imshow_det_bboxes', 'imshow_det_bboxes_ht_rgb', 'imshow_det_bboxes_ht_depth','imshow_gt_det_bboxes'
]

EPS = 1e-2
def calcAndDrawHist_max_pixle_value_cutout_person(image, color,h_num=256):
    hist = cv2.calcHist([image], [0], None, [h_num], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256, 3], np.uint8)
    hpt = int(0.9 * 256)

    max_gray_scale_value=-1
    max_gray_scale_value_count = 0
    for h in range(h_num):
        if hist[h] >max_gray_scale_value_count and h>5:
            max_gray_scale_value_count=hist[h]
            max_gray_scale_value=h
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (int(h/h_num*256), 256), (int(h/h_num*256), 256 - intensity), color)
    return histImg,max_gray_scale_value
middle_the=20

def ht_find_depth(img_find_depth,bboxes):
    bboxes_depth=[]
    for bbox in bboxes:
        xmin, ymin, xmax, ymax,_ = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        depth_image_data_1_middle = img_find_depth[
                                    int((ymin + ymax - 2) / 2) - middle_the:int((ymin + ymax - 2) / 2) + middle_the + 1,
                                    int((xmin + xmax - 2) / 2) - middle_the:int((xmin + xmax - 2) / 2) + middle_the + 1,
                                    0:3]

        histimg, max_gray_scale_value = calcAndDrawHist_max_pixle_value_cutout_person(depth_image_data_1_middle,
                                                                                      (0, 0, 255))
        bboxes_depth.append(max_gray_scale_value*780/255)
    return bboxes_depth
# 深度图反光条和周围环境导致的出现无深度，深度图为nan/0值，处理
def filter_zero(dataset):
    # 使用 nonzero 函数获取非零元素的下标
    non_zero_indexes = np.nonzero(dataset)

    # 使用这些下标选择数组中的元素，以去掉包含0值的元素
    new_arr =dataset[non_zero_indexes]

    return new_arr

# 3sigma法
def three_sigma(dataset, n=3):
    mean = np.mean(dataset)
    sigma = np.std(dataset)

    remove_idx = np.where(abs(dataset - mean) > n * sigma)
    new_data = np.delete(dataset, remove_idx)

    return new_data
# 百分位法:原始参数 min=0.025， max=0.975
def percent_range(dataset, min=0.20, max=0.80):

    range_max = np.percentile(dataset, max * 100)
    range_min = -np.percentile(-dataset, (1 - min) * 100)

    # 剔除前20%和后80%的数据
    new_data = []
    for value in dataset:
        if value < range_max and value > range_min:
            new_data.append(value)
    return new_data

def ht_find_depth_new(img_find_depth,bboxes):
    bboxes_depth=[]
    for bbox in bboxes:
        xmin, ymin, xmax, ymax,_ = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        depth_image_data_1_middle = img_find_depth[
                                    int((ymin + ymax - 2) / 2) - middle_the:int((ymin + ymax - 2) / 2) + middle_the + 1,
                                    int((xmin + xmax - 2) / 2) - middle_the:int((xmin + xmax - 2) / 2) + middle_the + 1,
                                    0]
        depth_image_list = depth_image_data_1_middle.reshape(-1, 1)
        depth_image_list = depth_image_list * 780 / 255

        ## filter zero
        # 找到小于常数c的元素的索引
        indices = np.where(depth_image_list < 50)[0]

        # 弹出这些元素
        depth_image_list = np.delete(depth_image_list, indices, axis=0)

        # 使用数据集X进行训练
        cluster = cluster.fit(depth_image_list)

        # 调用属性labels_，查看聚类结果
        label = cluster.labels_

        max_count_cluster_index = np.argmax(np.bincount(label))
        # 获取最大簇的元素
        max_cluster_elems = depth_image_list[cluster.labels_ == max_count_cluster_index]

        # 计算最大簇的均值和方差
        max_cluster_var = np.var(max_cluster_elems, axis=0)
        centroid_of_largest_cluster = cluster.cluster_centers_[max_count_cluster_index]

        bboxes_depth.append(centroid_of_largest_cluster)
    return bboxes_depth

def ht_find_depth_histogram(img_find_depth,bboxes,middle_the_x_ratio=0.3,middle_the_y_ratio=0.5):
    bboxes_depth=[]
    bboxes_sigma = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax,_= bbox
        # if xmin != 0:
        #     print('ht')
        x_l = xmax-xmin
        y_l = ymax-ymin
        x_m = int(middle_the_x_ratio*x_l)
        y_m = int(middle_the_y_ratio * y_l)

        # depth_image_data_1_middle = img_find_depth[
        #                             int((ymin + ymax - 2) / 2) - middle_the:int((ymin + ymax - 2) / 2) + middle_the + 1,
        #                             int((xmin + xmax - 2) / 2) - middle_the:int((xmin + xmax - 2) / 2) + middle_the + 1,
        #                             0]

        depth_image_data_1_middle = img_find_depth[
                                    int((ymin + ymax - 2) / 2) - y_m:int((ymin + ymax - 2) / 2) + y_m + 1,
                                    int((xmin + xmax - 2) / 2) - x_m:int((xmin + xmax - 2) / 2) + x_m + 1,
                                    0:3]
        depth_image_data_1_middle = filter_zero(depth_image_data_1_middle)
        if len(depth_image_data_1_middle)>0:
            counts = np.bincount(depth_image_data_1_middle)
            mean = np.argmax(counts)
            sigma = (780/255)**2
        else:
            mean = 122.5
            sigma = 160000

        bboxes_depth.append(mean*780/255)
        bboxes_sigma.append(sigma)
    return bboxes_depth,bboxes_sigma
def ht_find_depth_var_justcrop(img_find_depth,bboxes,hw_area_thre=5000,crop_ratio=0.25):
    bboxes_depth=[]
    bboxes_depth_var = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax,_ = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        w = xmax-xmin
        h = ymax-ymin
        hw_area = h * w

        depth_image_list = img_find_depth[int(ymin-1+(1 - crop_ratio) * h / 2):int(ymin+(1 + crop_ratio) * h / 2),
                           int(xmin-1+(1 - crop_ratio) * w / 2):int(xmin+(1 + crop_ratio) * w / 2), 0].reshape(-1, 1)

        depth_image_list = depth_image_list.flatten()
        depth_image_list = depth_image_list * 780 / 255

        depth_image_list_valid = depth_image_list[depth_image_list >= 50]
        if depth_image_list_valid.shape[0] > 0:
            max_cluster_var = np.var(depth_image_list_valid)
            mean = np.mean(depth_image_list_valid)
        else:
            max_cluster_var = 10000
            mean = 8


        bboxes_depth.append(mean)
        bboxes_depth_var.append(max_cluster_var)
    return bboxes_depth,bboxes_depth_var

def ht_find_depth_var(img_find_depth,bboxes,hw_area_thre=5000):
    bboxes_depth=[]
    bboxes_depth_var = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax,_ = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        w = xmax-xmin
        h = ymax-ymin
        hw_area = h * w
        if hw_area <= hw_area_thre:
            depth_image_list = img_find_depth[ymin-1:ymax, xmin-1:xmax, 0].reshape(-1, 1)
        else:
            crop_ratio = math.sqrt(hw_area_thre / hw_area)
            depth_image_list = img_find_depth[int(ymin-1+(1 - crop_ratio) * h / 2):int(ymin+(1 + crop_ratio) * h / 2),
                               int(xmin-1+(1 - crop_ratio) * w / 2):int(xmin+(1 + crop_ratio) * w / 2), 0].reshape(-1, 1)

        depth_image_list = depth_image_list.flatten()
        depth_image_list = depth_image_list * 780 / 255

        depth_image_list_valid = depth_image_list[depth_image_list >= 50]
        if depth_image_list_valid.shape[0] > 0:
            max_cluster_var = np.var(depth_image_list_valid)
            mean = np.mean(depth_image_list_valid)
        else:
            max_cluster_var = 10000
            mean = 8


        bboxes_depth.append(mean)
        bboxes_depth_var.append(max_cluster_var)
    return bboxes_depth,bboxes_depth_var

def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples.

    Args:
        color (:obj`Color` | str | tuple | int | ndarray): Color inputs.

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


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


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def draw_bboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax



def draw_labels_ht_show_conf(ax,
                labels,
                positions,
                bboxes_depth,
                scores=None,
                class_names=None,
                color='w',
                font_size=8,
                scales=None,
                horizontal_alignment='left'):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = ''
        # label_text = class_names[
        #     label] if class_names is not None else f'class {label}'
        if scores is not None:
            # label_text += f'|{scores[i]:.02f}'
            # bboxes_depth
            # label_text += f'| depth:{bboxes_depth[i]:.02f}cm'
            label_text += f'Confidence: {scores[i]:.02f}'
            label_text += f' Depth: {bboxes_depth[i]:.02f} cm'
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(
            pos[0],
            pos[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            # fontsize=font_size_mask,
            fontsize=17,
            verticalalignment='top',
            horizontalalignment=horizontal_alignment)

    return ax




def draw_labels_ht(ax,
                labels,
                positions,
                bboxes_depth,
                scores=None,
                class_names=None,
                color='w',
                font_size=8,
                scales=None,
                horizontal_alignment='left'):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = ''
        # label_text = class_names[
        #     label] if class_names is not None else f'class {label}'
        if scores is not None:
            # label_text += f'|{scores[i]:.02f}'
            # bboxes_depth
            # label_text += f'| depth:{bboxes_depth[i]:.02f}cm'
            label_text += f'Depth: {bboxes_depth[i]:.02f} cm'
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(
            pos[0],
            pos[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            # fontsize=font_size_mask,
            fontsize=25,
            verticalalignment='top',
            horizontalalignment=horizontal_alignment)

    return ax


def draw_masks(ax, img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha

    p = PatchCollection(
        polygons, facecolor='none', edgecolors='w', linewidths=1, alpha=0.8)
    ax.add_collection(p)

    return ax, img


def imshow_det_bboxes(img,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
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
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
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
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels(
            ax,
            labels[:num_bboxes],
            positions,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

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
# 不显示置信度的
# def imshow_det_bboxes_ht_rgb(img,
#                       bboxes_depth,
#                       bboxes=None,
#                       labels=None,
#                       segms=None,
#                       class_names=None,
#                       score_thr=0,
#                       bbox_color='green',
#                       text_color='green',
#                       mask_color=None,
#                       thickness=15,
#                       font_size=100,
#                       win_name='',
#                       show=True,
#                       wait_time=0,
#                       out_file=None):
#     """Draw bboxes and class labels (with scores) on an image.
#
#     Args:
#         img (str | ndarray): The image to be displayed.
#         bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
#             (n, 5).
#         labels (ndarray): Labels of bboxes.
#         segms (ndarray | None): Masks, shaped (n,h,w) or None.
#         class_names (list[str]): Names of each classes.
#         score_thr (float): Minimum score of bboxes to be shown. Default: 0.
#         bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
#            If a single color is given, it will be applied to all classes.
#            The tuple of color should be in RGB order. Default: 'green'.
#         text_color (list[tuple] | tuple | str | None): Colors of texts.
#            If a single color is given, it will be applied to all classes.
#            The tuple of color should be in RGB order. Default: 'green'.
#         mask_color (list[tuple] | tuple | str | None, optional): Colors of
#            masks. If a single color is given, it will be applied to all
#            classes. The tuple of color should be in RGB order.
#            Default: None.
#         thickness (int): Thickness of lines. Default: 2.
#         font_size (int): Font size of texts. Default: 13.
#         show (bool): Whether to show the image. Default: True.
#         win_name (str): The window name. Default: ''.
#         wait_time (float): Value of waitKey param. Default: 0.
#         out_file (str, optional): The filename to write the image.
#             Default: None.
#
#     Returns:
#         ndarray: The image with bboxes drawn on it.
#     """
#     assert bboxes is None or bboxes.ndim == 2, \
#         f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
#     assert bboxes is None or labels.ndim == 1, \
#         f' labels ndim should be 1, but its ndim is {labels.ndim}.'
#     assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
#         f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
#     assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
#         'labels.shape[0] should not be less than bboxes.shape[0].'
#     assert segms is None or segms.shape[0] == labels.shape[0], \
#         'segms.shape[0] and labels.shape[0] should have the same length.'
#     assert segms is not None or bboxes is not None, \
#         'segms and bboxes should not be None at the same time.'
#
#     img = mmcv.imread(img).astype(np.uint8)
#
#     if score_thr > 0:
#         assert bboxes is not None and bboxes.shape[1] == 5
#         scores = bboxes[:, -1]
#         inds = scores > score_thr
#         bboxes = bboxes[inds, :]
#         labels = labels[inds]
#         if segms is not None:
#             segms = segms[inds, ...]
#
#     img = mmcv.bgr2rgb(img)
#     width, height = img.shape[1], img.shape[0]
#     img = np.ascontiguousarray(img)
#
#     fig = plt.figure(win_name, frameon=False)
#     plt.title(win_name)
#     canvas = fig.canvas
#     dpi = fig.get_dpi()
#     # add a small EPS to avoid precision lost due to matplotlib's truncation
#     # (https://github.com/matplotlib/matplotlib/issues/15363)
#     fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
#
#     # remove white edges by set subplot margin
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax = plt.gca()
#     ax.axis('off')
#
#     max_label = int(max(labels) if len(labels) > 0 else 0)
#     text_palette = palette_val(get_palette(text_color, max_label + 1))
#     # text_colors = [text_palette[label] for label in labels]
#     text_colors = [((0, 1., 0) ) for bbox_depth in bboxes_depth]
#
#     num_bboxes = 0
#     if bboxes is not None:
#         num_bboxes = bboxes.shape[0]
#         bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
#         # colors = [bbox_palette[label] for label in labels[:num_bboxes]]
#         colors = [((0, 1., 0) ) for bbox_depth in bboxes_depth]
#         draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)
#
#         horizontal_alignment = 'left'
#         positions = bboxes[:, :2].astype(np.int32) + thickness
#         areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
#         scales = _get_adaptive_scales(areas)
#         scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
#         draw_labels_ht(
#             ax,
#             labels[:num_bboxes],
#             positions,
#             bboxes_depth,
#             scores=scores,
#             class_names=class_names,
#             color=text_colors,
#             font_size=font_size,
#             scales=scales,
#             # scales=None,
#             horizontal_alignment=horizontal_alignment)
#
#     if segms is not None:
#         mask_palette = get_palette(mask_color, max_label + 1)
#         colors = [mask_palette[label] for label in labels]
#         colors = np.array(colors, dtype=np.uint8)
#         draw_masks(ax, img, segms, colors, with_edge=True)
#
#         if num_bboxes < segms.shape[0]:
#             segms = segms[num_bboxes:]
#             horizontal_alignment = 'center'
#             areas = []
#             positions = []
#             for mask in segms:
#                 _, _, stats, centroids = cv2.connectedComponentsWithStats(
#                     mask.astype(np.uint8), connectivity=8)
#                 largest_id = np.argmax(stats[1:, -1]) + 1
#                 positions.append(centroids[largest_id])
#                 areas.append(stats[largest_id, -1])
#             areas = np.stack(areas, axis=0)
#             scales = _get_adaptive_scales(areas)
#             draw_labels(
#                 ax,
#                 labels[num_bboxes:],
#                 positions,
#                 class_names=class_names,
#                 color=text_colors,
#                 font_size=font_size,
#                 scales=scales,
#                 horizontal_alignment=horizontal_alignment)
#
#     plt.imshow(img)
#
#     stream, _ = canvas.print_to_buffer()
#     buffer = np.frombuffer(stream, dtype='uint8')
#     img_rgba = buffer.reshape(height, width, 4)
#     rgb, alpha = np.split(img_rgba, [3], axis=2)
#     img = rgb.astype('uint8')
#     img = mmcv.rgb2bgr(img)
#     plt.close('all')
#
#     if show:
#         # We do not use cv2 for display because in some cases, opencv will
#         # conflict with Qt, it will output a warning: Current thread
#         # is not the object's thread. You can refer to
#         # https://github.com/opencv/opencv-python/issues/46 for details
#         # if wait_time == 0:
#         #     plt.show()
#         # else:
#         #     plt.show(block=False)
#         #     plt.pause(wait_time)
#         cv2.imshow('miner_detect', img)
#         cv2.waitKey(wait_time)
#         cv2.destroyAllWindows()
#     if out_file is not None:
#         mmcv.imwrite(img, out_file)
#
#     # plt.close('all')
#
#     return img


#显示置信度的
def imshow_det_bboxes_ht_rgb(img,
                      bboxes_depth,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
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
    assert bboxes is None or labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
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
    # text_colors = [text_palette[label] for label in labels]
    text_colors = [((0, 1., 0) ) for bbox_depth in bboxes_depth]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        # colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        colors = [((0, 1., 0) ) for bbox_depth in bboxes_depth]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels_ht_show_conf(
            ax,
            labels[:num_bboxes],
            positions,
            bboxes_depth,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

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
    plt.close('all')

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        # if wait_time == 0:
        #     plt.show()
        # else:
        #     plt.show(block=False)
        #     plt.pause(wait_time)
        cv2.imshow('miner_detect', img)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    # plt.close('all')

    return img


# def imshow_det_bboxes_ht_rgb(img,
#                       bboxes_depth,
#                       bboxes=None,
#                       labels=None,
#                       segms=None,
#                       class_names=None,
#                       score_thr=0,
#                       bbox_color='green',
#                       text_color='green',
#                       mask_color=None,
#                       thickness=2,
#                       font_size=8,
#                       win_name='',
#                       show=True,
#                       wait_time=0,
#                       out_file=None):
#     """Draw bboxes and class labels (with scores) on an image.
#
#     Args:
#         img (str | ndarray): The image to be displayed.
#         bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
#             (n, 5).
#         labels (ndarray): Labels of bboxes.
#         segms (ndarray | None): Masks, shaped (n,h,w) or None.
#         class_names (list[str]): Names of each classes.
#         score_thr (float): Minimum score of bboxes to be shown. Default: 0.
#         bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
#            If a single color is given, it will be applied to all classes.
#            The tuple of color should be in RGB order. Default: 'green'.
#         text_color (list[tuple] | tuple | str | None): Colors of texts.
#            If a single color is given, it will be applied to all classes.
#            The tuple of color should be in RGB order. Default: 'green'.
#         mask_color (list[tuple] | tuple | str | None, optional): Colors of
#            masks. If a single color is given, it will be applied to all
#            classes. The tuple of color should be in RGB order.
#            Default: None.
#         thickness (int): Thickness of lines. Default: 2.
#         font_size (int): Font size of texts. Default: 13.
#         show (bool): Whether to show the image. Default: True.
#         win_name (str): The window name. Default: ''.
#         wait_time (float): Value of waitKey param. Default: 0.
#         out_file (str, optional): The filename to write the image.
#             Default: None.
#
#     Returns:
#         ndarray: The image with bboxes drawn on it.
#     """
#     assert bboxes is None or bboxes.ndim == 2, \
#         f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
#     assert bboxes is None or labels.ndim == 1, \
#         f' labels ndim should be 1, but its ndim is {labels.ndim}.'
#     assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
#         f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
#     assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
#         'labels.shape[0] should not be less than bboxes.shape[0].'
#     assert segms is None or segms.shape[0] == labels.shape[0], \
#         'segms.shape[0] and labels.shape[0] should have the same length.'
#     assert segms is not None or bboxes is not None, \
#         'segms and bboxes should not be None at the same time.'
#
#     img = mmcv.imread(img).astype(np.uint8)
#
#     if score_thr > 0:
#         assert bboxes is not None and bboxes.shape[1] == 5
#         scores = bboxes[:, -1]
#         inds = scores > score_thr
#         bboxes = bboxes[inds, :]
#         labels = labels[inds]
#         if segms is not None:
#             segms = segms[inds, ...]
#
#     img = mmcv.bgr2rgb(img)
#     width, height = img.shape[1], img.shape[0]
#     img = np.ascontiguousarray(img)
#
#     fig = plt.figure(win_name, frameon=False)
#     plt.title(win_name)
#     canvas = fig.canvas
#     dpi = fig.get_dpi()
#     # add a small EPS to avoid precision lost due to matplotlib's truncation
#     # (https://github.com/matplotlib/matplotlib/issues/15363)
#     fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
#
#     # remove white edges by set subplot margin
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax = plt.gca()
#     ax.axis('off')
#
#     max_label = int(max(labels) if len(labels) > 0 else 0)
#     text_palette = palette_val(get_palette(text_color, max_label + 1))
#     # text_colors = [text_palette[label] for label in labels]
#     text_colors = [((0, 1., 0) ) for bbox_depth in bboxes_depth]
#
#     num_bboxes = 0
#     if bboxes is not None:
#         num_bboxes = bboxes.shape[0]
#         bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
#         # colors = [bbox_palette[label] for label in labels[:num_bboxes]]
#         colors = [((0, 1., 0) ) for bbox_depth in bboxes_depth]
#         draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)
#
#         horizontal_alignment = 'left'
#         positions = bboxes[:, :2].astype(np.int32) + thickness
#         areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
#         scales = _get_adaptive_scales(areas)
#         scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
#         draw_labels_ht(
#             ax,
#             labels[:num_bboxes],
#             positions,
#             bboxes_depth,
#             scores=scores,
#             class_names=class_names,
#             color=text_colors,
#             font_size=font_size,
#             scales=scales,
#             horizontal_alignment=horizontal_alignment)
#
#     if segms is not None:
#         mask_palette = get_palette(mask_color, max_label + 1)
#         colors = [mask_palette[label] for label in labels]
#         colors = np.array(colors, dtype=np.uint8)
#         draw_masks(ax, img, segms, colors, with_edge=True)
#
#         if num_bboxes < segms.shape[0]:
#             segms = segms[num_bboxes:]
#             horizontal_alignment = 'center'
#             areas = []
#             positions = []
#             for mask in segms:
#                 _, _, stats, centroids = cv2.connectedComponentsWithStats(
#                     mask.astype(np.uint8), connectivity=8)
#                 largest_id = np.argmax(stats[1:, -1]) + 1
#                 positions.append(centroids[largest_id])
#                 areas.append(stats[largest_id, -1])
#             areas = np.stack(areas, axis=0)
#             scales = _get_adaptive_scales(areas)
#             draw_labels(
#                 ax,
#                 labels[num_bboxes:],
#                 positions,
#                 class_names=class_names,
#                 color=text_colors,
#                 font_size=font_size,
#                 scales=scales,
#                 horizontal_alignment=horizontal_alignment)
#
#     plt.imshow(img)
#
#     stream, _ = canvas.print_to_buffer()
#     buffer = np.frombuffer(stream, dtype='uint8')
#     img_rgba = buffer.reshape(height, width, 4)
#     rgb, alpha = np.split(img_rgba, [3], axis=2)
#     img = rgb.astype('uint8')
#     img = mmcv.rgb2bgr(img)
#     plt.close('all')
#
#     if show:
#         # We do not use cv2 for display because in some cases, opencv will
#         # conflict with Qt, it will output a warning: Current thread
#         # is not the object's thread. You can refer to
#         # https://github.com/opencv/opencv-python/issues/46 for details
#         # if wait_time == 0:
#         #     plt.show()
#         # else:
#         #     plt.show(block=False)
#         #     plt.pause(wait_time)
#         cv2.imshow('miner_detect', img)
#         cv2.waitKey(wait_time)
#         cv2.destroyAllWindows()
#     if out_file is not None:
#         mmcv.imwrite(img, out_file)
#
#     # plt.close('all')
#
#     return img

def imshow_det_bboxes_ht_rgb_noshow(img,win_name='',
                    show=True,
                    wait_time=0,
                    out_file=None):




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


    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)
    plt.close('all')

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        # if wait_time == 0:
        #     plt.show()
        # else:
        #     plt.show(block=False)
        #     plt.pause(wait_time)
        cv2.imshow('miner_detect', img)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    # plt.close('all')

    return img


def imshow_det_bboxes_ht_depth(img,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
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
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img_find_depth=img.copy()


    # 深度值计算
    bboxes_depth=ht_find_depth_new(img_find_depth,bboxes)


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
    # text_colors = [text_palette[label] for label in labels]
    text_colors = [((0,1.,0)) for bbox_depth in bboxes_depth]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        # bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        # bbox_palette = [(1.,0,0),(0,1.,0)]

        # colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        # colors =[(1.,0,0),(0,1.,0)]
        colors =[((0,1.,0) ) for bbox_depth in bboxes_depth]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels_ht(
            ax,
            labels[:num_bboxes],
            positions,
            bboxes_depth,
            scores=scores,
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
    plt.close('all')

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        # if wait_time == 0:
        #     plt.show()
        # else:
        #     plt.show(block=False)
        #     plt.pause(wait_time)
        cv2.imshow('miner_detect',img)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    # plt.close('all')

    return img,bboxes,bboxes_depth

def imshow_det_bboxes_ht_depth_var(img,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
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
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img_find_depth=img.copy()


    # 深度值计算
    # bboxes_depth,bboxes_depth_var=ht_find_depth_var(img_find_depth,bboxes)
    bboxes_depth,bboxes_depth_var=ht_find_depth_histogram(img_find_depth,bboxes)


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
    # text_colors = [text_palette[label] for label in labels]
    text_colors = [((1.,0,0) ) for bbox_depth in bboxes_depth]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        # bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        # bbox_palette = [(1.,0,0),(0,1.,0)]

        # colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        # colors =[(1.,0,0),(0,1.,0)]
        colors =[((1.,0,0) ) for bbox_depth in bboxes_depth]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels_ht(
            ax,
            labels[:num_bboxes],
            positions,
            bboxes_depth,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

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
                bboxes_depth,
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
    plt.close('all')

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        # if wait_time == 0:
        #     plt.show()
        # else:
        #     plt.show(block=False)
        #     plt.pause(wait_time)
        cv2.imshow('miner_detect',img)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    # plt.close('all')

    return img,bboxes,bboxes_depth,bboxes_depth_var

def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(61, 102, 255),
                         gt_text_color=(200, 200, 200),
                         gt_mask_color=(61, 102, 255),
                         det_bbox_color=(241, 101, 72),
                         det_text_color=(200, 200, 200),
                         det_mask_color=(241, 101, 72),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None):
    """General visualization GT and result function.

    Args:
      img (str | ndarray): The image to be displayed.
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'.
      result (tuple[list] | list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown. Default: 0.
      gt_bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      gt_text_color (list[tuple] | tuple | str | None): Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      gt_mask_color (list[tuple] | tuple | str | None, optional): Colors of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      det_bbox_color (list[tuple] | tuple | str | None):Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      det_text_color (list[tuple] | tuple | str | None):Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      det_mask_color (list[tuple] | tuple | str | None, optional): Color of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      thickness (int): Thickness of lines. Default: 2.
      font_size (int): Font size of texts. Default: 13.
      win_name (str): The window name. Default: ''.
      show (bool): Whether to show the image. Default: True.
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
          Default: None.

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(result, (tuple, list, dict)), 'Expected ' \
        f'tuple or list or dict, but get {type(result)}'

    gt_bboxes = annotation['gt_bboxes']
    gt_labels = annotation['gt_labels']
    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    gt_seg = annotation.get('gt_semantic_seg', None)
    if gt_seg is not None:
        pad_value = 255  # the padding value of gt_seg
        sem_labels = np.unique(gt_seg)
        all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
        all_labels, counts = np.unique(all_labels, return_counts=True)
        stuff_labels = all_labels[np.logical_and(counts < 2,
                                                 all_labels != pad_value)]
        stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
        gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
        gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)),
                                  axis=0)
        # If you need to show the bounding boxes,
        # please comment the following line
        # gt_bboxes = None

    img = mmcv.imread(img)

    img = imshow_det_bboxes(
        img,
        gt_bboxes,
        gt_labels,
        gt_masks,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)

    if not isinstance(result, dict):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            segms = mask_util.decode(segms)
            segms = segms.transpose(2, 0, 1)
    else:
        assert class_names is not None, 'We need to know the number ' \
                                        'of classes.'
        VOID = len(class_names)
        bboxes = None
        pan_results = result['pan_results']
        # keep objects ahead
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != VOID
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (pan_results[None] == ids[:, None, None])

    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms=segms,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color=det_bbox_color,
        text_color=det_text_color,
        mask_color=det_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    return img
