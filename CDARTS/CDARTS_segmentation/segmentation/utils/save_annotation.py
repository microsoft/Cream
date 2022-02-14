# ------------------------------------------------------------------------------
# Saves output to png image for visualization.
# Reference: https://github.com/tensorflow/models/blob/master/research/deeplab/utils/save_annotation.py
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import numpy as np
import PIL.Image as img
from PIL import ImageDraw

from .flow_vis import flow_compute_color

# Refence: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py#L14
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


def random_color(rgb=False, maximum=255):
    """
    Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py#L111
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1
    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def save_annotation(label,
                    save_dir,
                    filename,
                    add_colormap=True,
                    normalize_to_unit_values=False,
                    scale_values=False,
                    colormap=None,
                    image=None):
    """Saves the given label to image on disk.
    Args:
        label: The numpy array to be saved. The data will be converted
            to uint8 and saved as png image.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        add_colormap: Boolean, add color map to the label or not.
        normalize_to_unit_values: Boolean, normalize the input values to [0, 1].
        scale_values: Boolean, scale the input values to [0, 255] for visualization.
        colormap: A colormap for visualizing segmentation results.
        image: merge label with image if provided
    """
    # Add colormap for visualizing the prediction.
    if add_colormap:
        colored_label = label_to_color_image(label, colormap)
    else:
        colored_label = label
    if normalize_to_unit_values:
        min_value = np.amin(colored_label)
        max_value = np.amax(colored_label)
        range_value = max_value - min_value
        if range_value != 0:
            colored_label = (colored_label - min_value) / range_value

    if scale_values:
        colored_label = 255. * colored_label

    if image is not None:
        colored_label = 0.5 * colored_label + 0.5 * image

    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def label_to_color_image(label, colormap=None):
    """Adds color defined by the dataset colormap to the label.
    Args:
        label: A 2D array with integer type, storing the segmentation label.
        colormap: A colormap for visualizing segmentation results.
    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the dataset color map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))

    if colormap is None:
        raise ValueError('Expect a valid colormap.')

    return colormap[label]


def save_instance_annotation(label,
                             save_dir,
                             filename,
                             stuff_id=0,
                             image=None):
    """Saves the given label to image on disk.
    Args:
        label: The numpy array to be saved. The data will be converted
            to uint8 and saved as png image.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        stuff_id: Integer, id that not want to plot.
        image: merge label with image if provided
    """
    # Add colormap for visualizing the prediction.
    ids = np.unique(label)
    num_colors = len(ids)
    colormap = np.zeros((num_colors, 3), dtype=np.uint8)
    # Maps label to continuous value.
    for i in range(num_colors):
        label[label == ids[i]] = i
        colormap[i, :] = random_color(rgb=True, maximum=255)
        if ids[i] == stuff_id:
            colormap[i, :] = np.array([0, 0, 0])
    colored_label = colormap[label]

    if image is not None:
        colored_label = 0.5 * colored_label + 0.5 * image

    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def save_panoptic_annotation(label,
                             save_dir,
                             filename,
                             label_divisor,
                             colormap=None,
                             image=None):
    """Saves the given label to image on disk.
    Args:
        label: The numpy array to be saved. The data will be converted
            to uint8 and saved as png image.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        colormap: A colormap for visualizing segmentation results.
        image: merge label with image if provided
    """
    if colormap is None:
        raise ValueError('Expect a valid colormap.')
    
    # Add colormap to label.
    colored_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    taken_colors = set([0, 0, 0])

    def _random_color(base, max_dist=30):
        new_color = base + np.random.randint(low=-max_dist,
                                             high=max_dist + 1,
                                             size=3)
        return tuple(np.maximum(0, np.minimum(255, new_color)))

    for lab in np.unique(label):
        mask = label == lab
        base_color = colormap[lab // label_divisor]
        if tuple(base_color) not in taken_colors:
            taken_colors.add(tuple(base_color))
            color = base_color
        else:
            while True:
                color = _random_color(base_color)
                if color not in taken_colors:
                    taken_colors.add(color)
                    break
        colored_label[mask] = color
    
    if image is not None:
        colored_label = 0.5 * colored_label + 0.5 * image

    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def save_center_image(image,
                      center_points,
                      save_dir,
                      filename,
                      radius=3):
    """Saves image with center points.
    Args:
        image: The image.
        center_points: List of tuple [(y, x)], center point coordinates.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        radius: Int, radius of the center point.
    """
    pil_image = img.fromarray(image.astype(dtype=np.uint8))
    draw = ImageDraw.Draw(pil_image)
    r = radius
    assigned_colors = [list(random_color(rgb=True, maximum=255)) + [255] for _ in range(len(center_points))]
    for i, point in enumerate(center_points):
        leftUpPoint = (point[1] - r, point[0] - r)
        rightDownPoint = (point[1] + r, point[0] + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, fill=tuple(assigned_colors[i]))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def save_heatmap_image(image,
                       center_heatmap,
                       save_dir,
                       filename,
                       ratio=0.5):
    """Saves image with heatmap.
    Args:
        image: The image.
        center_heatmap: Ndarray, center heatmap.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        radio: Float, ratio to mix heatmap and image, out = ratio * heatmap + (1 - ratio) * image.
    """
    center_heatmap = center_heatmap[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
    center_heatmap = center_heatmap.clip(0, 255)
    image = ratio * center_heatmap + (1 - ratio) * image
    pil_image = img.fromarray(image.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def save_heatmap_and_center_image(image,
                                  center_heatmap,
                                  center_points,
                                  save_dir,
                                  filename,
                                  ratio=0.5,
                                  radius=25,
                                  binarize_heatmap=True):
    """Saves image with non-negative heatmap and center radius.
    Args:
        image: The image.
        center_heatmap: Ndarray, center heatmap.
        center_points: List of tuple [(y, x)], center point coordinates.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
        radio: Float, ratio to mix heatmap and image, out = ratio * heatmap + (1 - ratio) * image.
        radius: Int, radius of the center point.
    """
    if binarize_heatmap:
        center_heatmap = (center_heatmap[:, :, None] > 0) * np.array([255, 0, 0]).reshape((1, 1, 3))
    else:
        center_heatmap = center_heatmap[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
    center_heatmap = center_heatmap.clip(0, 255)
    image = ratio * center_heatmap + (1 - ratio) * image
    pil_image = img.fromarray(image.astype(dtype=np.uint8))
    draw = ImageDraw.Draw(pil_image)
    r = radius
    assigned_colors = [list(random_color(rgb=True, maximum=255)) + [255] for _ in range(len(center_points))]
    for i, point in enumerate(center_points):
        leftUpPoint = (point[1] - r, point[0] - r)
        rightDownPoint = (point[1] + r, point[0] + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        if binarize_heatmap:
            draw.ellipse(twoPointList, outline='blue')
        else:
            draw.ellipse(twoPointList, fill=tuple(assigned_colors[i]))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')


def save_offset_image(offset,
                      save_dir,
                      filename):
    """Saves image with heatmap.
    Args:
        image: The offset to save.
        save_dir: String, the directory to which the results will be saved.
        filename: String, the image filename.
    """
    offset_image = flow_compute_color(offset[:, :, 1], offset[:, :, 0])
    pil_image = img.fromarray(offset_image.astype(dtype=np.uint8))
    with open('%s/%s.png' % (save_dir, filename), mode='wb') as f:
        pil_image.save(f, 'PNG')
