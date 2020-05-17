import tensorflow as tf
import numpy as np

# a small value
EPSILON = 1e-8
SCALE_FACTORS = 5.0

"""
Tools for dealing with bounding boxes.
All boxes are of the format [ymin, xmin, ymax, xmax]
if not stated otherwise.
And box coordinates are normalized to [0, 1] range.
"""


def iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between two box collections.

    Arguments:
        boxes1: a float tensor with shape [N, 4].GT
        boxes2: a float tensor with shape [M, 4].ANCHOR
    Returns:
        a float tensor with shape [N, M] representing pairwise iou scores.
    """

    intersections = intersection(boxes1, boxes2)  #####################transfored wrong

    areas1 = area(boxes1)
    areas2 = area(boxes2)
    unions = np.expand_dims(areas1, 1) + np.expand_dims(areas2, 0) - intersections

    return np.clip(intersections / unions, 0.0, 1.0)


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise intersections.
    """
    ##########np transformed wrong need review
    ymin1, xmin1, ymax1, xmax1 = np.split(boxes1, indices_or_sections=4, axis=1)
    ymin2, xmin2, ymax2, xmax2 = np.split(boxes2, indices_or_sections=4, axis=1)
    # they all have shapes like [None, 1]

    all_pairs_min_ymax = np.minimum(ymax1, np.transpose(ymax2))
    all_pairs_max_ymin = np.maximum(ymin1, np.transpose(ymin2))

    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(xmax1, np.transpose(xmax2))
    all_pairs_max_xmin = np.maximum(xmin1, np.transpose(xmin2))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    # they all have shape [N, M]
    return intersect_heights * intersect_widths


def area(boxes):
    """Computes area of boxes.

    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N] representing box areas.
    """

    ymin, xmin, ymax, xmax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return (ymax - ymin) * (xmax - xmin)


def to_minmax_coordinates(boxes):
    """Convert bounding boxes of the format
    [cy, cx, h, w] to the format [ymin, xmin, ymax, xmax].

    Arguments:
        boxes: a list of float tensors with shape [N]
            that represent cy, cx, h, w.
    Returns:
        a list of float tensors with shape [N]
        that represent ymin, xmin, ymax, xmax.
    """

    cy, cx, h, w = boxes
    ymin, xmin = cy - 0.5 * h, cx - 0.5 * w
    ymax, xmax = cy + 0.5 * h, cx + 0.5 * w
    return [ymin, xmin, ymax, xmax]


def to_center_coordinates(boxes):
    """Convert bounding boxes of the format
    [ymin, xmin, ymax, xmax] to the format [cy, cx, h, w].

    Arguments:
        boxes: a list of float tensors with shape [N]
            that represent ymin, xmin, ymax, xmax.
    Returns:
        a list of float tensors with shape [N]
        that represent cy, cx, h, w.
    """

    ymin, xmin, ymax, xmax = boxes
    h = ymax - ymin
    w = xmax - xmin
    cy = ymin + 0.5 * h
    cx = xmin + 0.5 * w
    return [cy, cx, h, w]


def encode(boxes, anchors):
    """Encode boxes with respect to anchors.

    Arguments:
        boxes: a float tensor with shape [N, 4].   yxyx
        anchors: a float tensor with shape [N, 4]. yxyx
    Returns:
        a float tensor with shape [N, 4],
        anchor-encoded boxes of the format [ty, tx, th, tw].
    """

    anchors_h = anchors[:, 2] - anchors[:, 0]
    anchors_w = anchors[:, 3] - anchors[:, 1]
    anchors_y = anchors[:, 0] + anchors_h / 2
    anchors_x = anchors[:, 1] + anchors_w / 2

    for i in range(0, 7):
        boxes[:, 2 * i] = (boxes[:, 2 * i] - anchors_y) / anchors_h * SCALE_FACTORS
        boxes[:, 2 * i + 1] = (boxes[:, 2 * i + 1] - anchors_x) / anchors_w * SCALE_FACTORS

    return boxes


def decode(codes, anchors):
    """Decode relative codes to boxes.

    Arguments:
        codes: a float tensor with shape [N, 14],
            anchor-encoded boxes of the format [ty, tx, ty, tx].
        anchors: a float tensor with shape [N, 4].  yxyx
    Returns:
        a float tensor with shape [N, 4],
        bounding boxes of the format [ymin, xmin, ymax, xmax].
    """
    with tf.name_scope('decode_predictions'):
        anchors_h = anchors[:, 2] - anchors[:, 0]
        anchors_w = anchors[:, 3] - anchors[:, 1]
        anchors_y = anchors[:, 0] + anchors_h / 2
        anchors_x = anchors[:, 1] + anchors_w / 2

        ty1, tx1, ty2, tx2, tly1, tlx1, tly2, tlx2, tly3, tlx3, tly4, tlx4, tly5, tlx5 = tf.unstack(codes, axis=1)

        y1 = ty1 / SCALE_FACTORS * anchors_h + anchors_y
        x1 = tx1 / SCALE_FACTORS * anchors_w + anchors_x
        y2 = ty2 / SCALE_FACTORS * anchors_h + anchors_y
        x2 = tx2 / SCALE_FACTORS * anchors_w + anchors_x

        ly1 = tly1 / SCALE_FACTORS * anchors_h + anchors_y
        lx1 = tlx1 / SCALE_FACTORS * anchors_w + anchors_x
        ly2 = tly2 / SCALE_FACTORS * anchors_h + anchors_y
        lx2 = tlx2 / SCALE_FACTORS * anchors_w + anchors_x
        ly3 = tly3 / SCALE_FACTORS * anchors_h + anchors_y
        lx3 = tlx3 / SCALE_FACTORS * anchors_w + anchors_x
        ly4 = tly4 / SCALE_FACTORS * anchors_h + anchors_y
        lx4 = tlx4 / SCALE_FACTORS * anchors_w + anchors_x
        ly5 = tly5 / SCALE_FACTORS * anchors_h + anchors_y
        lx5 = tlx5 / SCALE_FACTORS * anchors_w + anchors_x

        return tf.stack([y1, x1, y2, x2, ly1, lx1, ly2, lx2, ly3, lx3, ly4, lx4, ly5, lx5], axis=1)


def batch_decode(box_encodings, anchors):
    """Decodes a batch of box encodings with respect to the anchors.

    Arguments:
        box_encodings: a float tensor with shape [batch_size, num_anchors, 14].
        anchors: a float tensor with shape [num_anchors, 4].
    Returns:
        a float tensor with shape [batch_size, num_anchors, 14].
        It contains the decoded boxes.
    """
    batch_size = tf.shape(box_encodings)[0]
    num_anchors = tf.shape(box_encodings)[1]

    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors, 0),
        [batch_size, 1, 1]
    )  # shape [batch_size, num_anchors, 4]

    decoded_boxes = decode(
        tf.reshape(box_encodings, [-1, 14]),
        tf.reshape(tiled_anchor_boxes, [-1, 4])
    )  # shape [batch_size * num_anchors, 4]

    decoded_boxes = tf.reshape(
        decoded_boxes,
        [batch_size, num_anchors, 14]
    )
    decoded_boxes = tf.clip_by_value(decoded_boxes, 0.0, 1.0)
    return decoded_boxes
