import cv2
import numpy as np
from skimage import transform as trans
from .standards import draw_landmark


def normalize_face_image(bgr_images=None, size=112):
    bgr_images = np.array(bgr_images).astype(np.float32) / 255.

    if len(bgr_images.shape) == 3:
        bgr_images = np.expand_dims(bgr_images, axis=0)

    return np.array([cv2.resize(img, (size, size)) for img in bgr_images])


def augment_face(face_image, landmark):
    ladms = np.ravel(landmark)
    y1 = int((ladms[5] + ladms[1])//2)

    face_image = face_image.copy()
    face_image[y1+5:, :, :] = 0
    return face_image


def get_image_face_from_bboxes(image, bboxes):

    origin_face_images = []
    augmented_face_images = []

    for e in bboxes:
        origin, augmented = preprocess(image, e[:4], e[4:14])
        origin_face_images.append(origin)
        augmented_face_images.append(augmented)

    # if len(origin_face_images) != 0:
    #     cv2.imshow('origin', origin_face_images[0])
    #     cv2.waitKey(0)
    origin_face_images.extend(augmented_face_images)
    return origin_face_images


def preprocess(img, bbox=None, landmark=None, image_size=(112, 112), **kwargs):
    M = None

    if landmark is not None:
        assert len(image_size) == 2
        if len(landmark) == 10:
            landmark = np.reshape(landmark, (5, 2))
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        # draw_landmark(warped, src)
        augmented_face = augment_face(warped, src)
        return warped, augmented_face
