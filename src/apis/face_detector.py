import tensorflow as tf
import numpy as np
import cv2
import time
from src.models.facebox.net import FaceBoxes
from src.config.default import MIN_FACE_SCORE


class FaceDetector:
    def __init__(self, model_path):
        """
        Arguments:
            model_path: a string, path to the model params file.
        """

        # self.model = tf.saved_model.load(model_path)
        self.model = FaceBoxes()
        self.model.load_weights(model_path)

    def __call__(self, images, score_threshold=MIN_FACE_SCORE):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].

        """
        # images = np.array(images)
        # if len(images.shape) == 3:
        #     images = np.expand_dims(images, 0)

        images_fornet = None
        scale_xs = []
        scale_ys = []
        for image in images:
            image_fornet, scale_x, scale_y = self.preprocess(image, 512, 512)
            # images_fornet.append(image_fornet)
            scale_xs.append(scale_x)
            scale_ys.append(scale_y)
            if images_fornet is None:
                images_fornet = image_fornet
            else:
                images_fornet = np.concatenate([images_fornet, image_fornet])

        # start = time.time()
        res = self.model.inference(images_fornet)

        # print('fps', 1 / (time.time() - start))
        # print('ms:', (time.time() - start)*1000)
        boxes = res['boxes'].numpy()
        scores = res['scores'].numpy()
        num_boxes = res['num_boxes'].numpy()

        results = []
        for e in zip(scale_xs, scale_ys, boxes, scores, num_boxes):
            scale_x, scale_y, box, score, num_box = e
            results.append(self._handle_result(scale_x, scale_y, box, score, num_box, score_threshold))

        return results

    def _handle_result(self, scale_x, scale_y, boxes, scores, num_boxes, score_threshold):
        boxes = boxes[:num_boxes]

        scores = scores[:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        ###recorver to raw image
        scaler = np.array([512 / scale_y,
                           512 / scale_x,
                           512 / scale_y,
                           512 / scale_x,
                           512 / scale_y,
                           512 / scale_x,
                           512 / scale_y,
                           512 / scale_x,
                           512 / scale_y,
                           512 / scale_x,
                           512 / scale_y,
                           512 / scale_x,
                           512 / scale_y,
                           512 / scale_x], dtype='float32')
        boxes = boxes * scaler

        scores = np.expand_dims(scores, 0).reshape([-1, 1])

        #####the tf.nms produce ymin,xmin,ymax,xmax,  swap it in to xmin,ymin,xmax,ymax
        for i in range(boxes.shape[0]):
            boxes[i] = np.array([boxes[i][1], boxes[i][0],
                                 boxes[i][3], boxes[i][2],
                                 boxes[i][5], boxes[i][4],
                                 boxes[i][7], boxes[i][6],
                                 boxes[i][9], boxes[i][8],
                                 boxes[i][11], boxes[i][10],
                                 boxes[i][13], boxes[i][12]])

        return np.concatenate([boxes, scores], axis=1)

    def preprocess(self, image, target_height, target_width, label=None):

        # sometimes use in objs detects
        h, w, c = image.shape

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype) + np.array([123., 116., 103.],
                                                                                                dtype=image.dtype)

        long_side = max(h, w)

        scale_x = scale_y = target_height / long_side

        image = cv2.resize(image, None, fx=scale_x, fy=scale_y)

        h_, w_, _ = image.shape
        bimage[:h_, :w_, :] = image

        return np.expand_dims(bimage, 0), scale_x, scale_y

    def init_model(self, *args):

        if len(args) == 1:
            use_pb = True
            pb_path = args[0]
        else:
            use_pb = False
            meta_path = args[0]
            restore_model_path = args[1]

        def ini_ckpt():
            graph = tf.Graph()
            graph.as_default()
            configProto = tf.ConfigProto()
            configProto.gpu_options.allow_growth = True
            sess = tf.Session(config=configProto)
            # load_model(model_path, sess)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, restore_model_path)

            print("Model restred!")
            return graph, sess

        def init_pb(model_path):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            compute_graph = tf.Graph()
            compute_graph.as_default()
            sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            # saver = tf.train.Saver(tf.global_variables())
            # saver.save(sess, save_path='./tmp.ckpt')
            return (compute_graph, sess)

        if use_pb:
            model = init_pb(pb_path)
        else:
            model = ini_ckpt()

        graph = model[0]
        sess = model[1]

        return graph, sess


def draw_image(image, boxes):
    for box in boxes:
        print(box[-1])
        box = box[:14]
        box = np.array(box).astype(int)
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        for x, y in zip(box[4::2], box[5::2]):
            cv2.circle(image, (x, y), 1, (0, 0, 255), 2)

    return image


def test_image(path, face_detector):
    image = cv2.imread(path)
    result = face_detector(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.36)
    image = draw_image(image, result[0])
    cv2.imshow('test', image)
    cv2.waitKey(0)


def test_video(vd, face_detector):
    video = cv2.VideoCapture(vd)
    while True:
        ret, image = video.read()
        result = face_detector(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image = draw_image(image, result[0])
        cv2.imshow('test', image)
        cv2.waitKey(1)


if __name__ == '__main__':
    import os
    print(os. getcwd())
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    face_detector = FaceDetector('./trained_model/faceboxes/epoch_299_train_lost_4.388730_val_loss3.445847/save')
    from glob import glob
    imgs = glob('/home/tupm/datasets/VN-celeb/156/*')
    for img in imgs:
        test_image(img, face_detector)
    # test_image('/home/tupm/datasets/VN-celeb/156/5.png', face_detector)
    # test_video('/home/tupm/SSD/datasets/video_test/Truyền File [17-05-2020 10_11]/40251493728313724542.mp4', face_detector)
    # print(result.shape)
