import tensorflow as tf
from src.models.facebox.utils.box_utils import batch_decode
from src.models.facebox.utils.nms import batch_non_max_suppression
from src.config.default import anchors


def batch_norm():
    return tf.keras.layers.BatchNormalization(fused=True, momentum=0.997, epsilon=1e-5)


class RDCL(tf.keras.Model):

    def __init__(self,
                 kernel_initializer='glorot_normal'):
        super(RDCL, self).__init__()

        self.conv1_1 = tf.keras.layers.Conv2D(filters=12,
                                              kernel_size=(7, 7),
                                              strides=2,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn1_1 = batch_norm()
        self.conv1_2 = tf.keras.layers.Conv2D(filters=24,
                                              kernel_size=(3, 3),
                                              strides=2,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn1_2 = batch_norm()

        self.conv2_1 = tf.keras.layers.Conv2D(filters=32,
                                              kernel_size=(3, 3),
                                              strides=2,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn2_1 = batch_norm()
        self.conv2_2 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=(3, 3),
                                              strides=2,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn2_2 = batch_norm()

    def __call__(self, x, training=False):
        x = tf.nn.relu(self.bn1_1(self.conv1_1(x), training=training))
        x = tf.nn.crelu(self.bn1_2(self.conv1_2(x), training=training))
        x = tf.nn.relu(self.bn2_1(self.conv2_1(x), training=training))
        x = tf.nn.crelu(self.bn2_2(self.conv2_2(x), training=training))

        return x


class Inception(tf.keras.Model):

    def __init__(self,
                 kernel_initializer='glorot_normal'):
        super(Inception, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer)
        self.bn1 = batch_norm()

        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                        strides=1,
                                                        padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer)
        self.bn2 = batch_norm()

        self.conv3_1 = tf.keras.layers.Conv2D(filters=24,
                                              kernel_size=(1, 1),
                                              strides=1,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn3_1 = batch_norm()
        self.conv3_2 = tf.keras.layers.Conv2D(filters=32,
                                              kernel_size=(3, 3),
                                              strides=1,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn3_2 = batch_norm()

        self.conv4_1 = tf.keras.layers.Conv2D(filters=24,
                                              kernel_size=(1, 1),
                                              strides=1,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn4_1 = batch_norm()
        self.conv4_2 = tf.keras.layers.Conv2D(filters=32,
                                              kernel_size=(3, 3),
                                              strides=1,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn4_2 = batch_norm()
        self.conv4_3 = tf.keras.layers.Conv2D(filters=32,
                                              kernel_size=(3, 3),
                                              strides=1,
                                              padding='same',
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer)
        self.bn4_3 = batch_norm()

    def __call__(self, x, training=False):
        path1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))

        path2_pool = self.avgpool(x)
        path2 = tf.nn.relu(self.bn2(self.conv2(path2_pool), training=training))

        path3_1 = tf.nn.relu(self.bn3_1(self.conv3_1(x), training=training))
        path3 = tf.nn.relu(self.bn3_2(self.conv3_2(path3_1), training=training))

        path4_1 = tf.nn.relu(self.bn4_1(self.conv4_1(x), training=training))
        path4_2 = tf.nn.relu(self.bn4_2(self.conv4_2(path4_1), training=training))
        path4 = tf.nn.relu(self.bn4_3(self.conv4_3(path4_2), training=training))

        return tf.concat([path1, path2, path3, path4], axis=3)


class DecreaseBlock(tf.keras.Model):

    def __init__(self,
                 kernel_initializer='glorot_normal'):
        super(DecreaseBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer)
        self.bn1 = batch_norm()

        self.conv2 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer)
        self.bn2 = batch_norm()

    def __call__(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))

        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))

        return x


class FaceBoxesHead(tf.keras.Model):
    def __init__(self,
                 kernel_initializer='glorot_normal'):
        super(FaceBoxesHead, self).__init__()
        self.conv_reg_1 = tf.keras.layers.Conv2D(filters=14 * 21,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='same',
                                                 use_bias=True,
                                                 kernel_initializer=kernel_initializer)
        self.conv_cls_1 = tf.keras.layers.Conv2D(filters=2 * 21,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='same',
                                                 use_bias=True,
                                                 kernel_initializer=kernel_initializer)

        self.conv_reg_2 = tf.keras.layers.Conv2D(filters=14 * 1,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='same',
                                                 use_bias=True,
                                                 kernel_initializer=kernel_initializer)
        self.conv_cls_2 = tf.keras.layers.Conv2D(filters=2 * 1,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='same',
                                                 use_bias=True,
                                                 kernel_initializer=kernel_initializer)

        self.conv_reg_3 = tf.keras.layers.Conv2D(filters=14 * 1,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='same',
                                                 use_bias=True,
                                                 kernel_initializer=kernel_initializer)
        self.conv_cls_3 = tf.keras.layers.Conv2D(filters=2 * 1,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='same',
                                                 use_bias=True,
                                                 kernel_initializer=kernel_initializer)

    def __call__(self, fms, training=False):
        fm1, fm2, fm3 = fms

        reg_1 = self.conv_reg_1(fm1)
        reg_1 = tf.reshape(reg_1, ([-1, 32, 32, 21, 14]))
        reg_1 = tf.reshape(reg_1, ([-1, 32 * 32 * 21, 14]))

        reg_2 = self.conv_reg_2(fm2)
        reg_2 = tf.reshape(reg_2, ([-1, 16, 16, 1, 14]))
        reg_2 = tf.reshape(reg_2, ([-1, 16 * 16, 14]))

        reg_3 = self.conv_reg_3(fm3)
        reg_3 = tf.reshape(reg_3, ([-1, 8, 8, 1, 14]))
        reg_3 = tf.reshape(reg_3, ([-1, 8 * 8, 14]))

        reg = tf.concat([reg_1, reg_2, reg_3], axis=1, name='reg_concat')

        cls_1 = self.conv_cls_1(fm1)
        cls_1 = tf.reshape(cls_1, ([-1, 32, 32, 21, 2]))
        cls_1 = tf.reshape(cls_1, ([-1, 32 * 32 * 21, 2]))

        cls_2 = self.conv_cls_2(fm2)
        cls_2 = tf.reshape(cls_2, ([-1, 16, 16, 1, 2]))
        cls_2 = tf.reshape(cls_2, ([-1, 16 * 16, 2]))

        cls_3 = self.conv_cls_3(fm3)
        cls_3 = tf.reshape(cls_3, ([-1, 8, 8, 1, 2]))
        cls_3 = tf.reshape(cls_3, ([-1, 8 * 8, 2]))

        cls = tf.concat([cls_1, cls_2, cls_3], axis=1, name='cls_concat')

        return reg, cls


class FaceBoxes(tf.keras.Model):
    def __init__(self,
                 kernel_initializer='glorot_normal'):
        super(FaceBoxes, self).__init__()

        self.RDCL = RDCL(kernel_initializer=kernel_initializer)
        self.inception_blocks = [Inception(kernel_initializer=kernel_initializer) for i in range(3)]

        self.decrease_blocks = [DecreaseBlock(kernel_initializer=kernel_initializer) for i in range(2)]

        self.head = FaceBoxesHead(kernel_initializer=kernel_initializer)

    def call(self, images, training):

        fms = []

        x = self.preprocess(images)

        x = self.RDCL(x, training=training)

        for inception in self.inception_blocks:
            x = inception(x, training=training)
        fms.append(x)

        for decrese in self.decrease_blocks:
            x = decrese(x, training=training)
            fms.append(x)

        loc_predict, cls_predict = self.head(fms, training=training)

        return loc_predict, cls_predict

    def predict(self, images):
        fms = []

        x = self.preprocess(images)

        x = self.RDCL(x, training=False)

        for inception in self.inception_blocks:
            x = inception(x, training=False)
        fms.append(x)

        for decrese in self.decrease_blocks:
            x = decrese(x, training=False)
            fms.append(x)

        loc_predict, cls_predict = self.head(fms, training=False)

        return loc_predict, cls_predict

    def preprocess(self, image):
        image = tf.cast(image, tf.float32)

        mean = [123., 116., 103.]

        image_mean = tf.constant(mean, dtype=tf.float32)
        # image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean)  # *image_invstd

        return image

    def get_predictions(self, box_encodings, cla, anchors, score_threshold=0.05, iou_threshold=0.3, max_boxes=100):
        """Postprocess outputs of the network.
        Returns:
            boxes: a float tensor with shape [batch_size, N, 14].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.
            where N = max_boxes.
        """
        boxes = batch_decode(box_encodings, anchors)
        # if the images were padded we need to rescale predicted boxes:

        boxes = tf.clip_by_value(boxes, 0.0, 1.0)
        # it has shape [batch_size, num_anchors, 14]

        scores = tf.nn.softmax(cla, axis=2)[:, :, 1]
        # it has shape [batch_size, num_anchors],  background are ignored

        with tf.device('/cpu:0'):
            boxes, scores, num_detections = batch_non_max_suppression(
                boxes, scores, score_threshold, iou_threshold, max_boxes
            )

        boxes = tf.identity(boxes, name='boxes')
        scores = tf.identity(scores, name='scores')
        num_detections = tf.identity(num_detections, name='num_detections')

        return {'boxes': boxes, 'scores': scores, 'num_boxes': num_detections}

    @tf.function(input_signature=[tf.TensorSpec([None, 512, 512, 3], tf.float32)])
    def inference(self, images):

        reg_prediction, cls_prediction = self.call(images, False)
        res = self.get_predictions(reg_prediction, cls_prediction, anchors=anchors)
        return res


if __name__ == '__main__':
    input = tf.zeros(shape=[1, 512, 512, 3], )

    model = FaceBoxes()
    model.load_weights('trained_model/faceboxes/epoch_299_train_lost_4.388730_val_loss3.445847/save')

    x = model(input, True)
    print(x)
