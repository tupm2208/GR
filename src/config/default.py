from src.models.facebox.anchor_generator import AnchorGenerator
import sys


feature_maps_size = [[32, 32], [16, 16], [8, 8]]

anchor_generator = AnchorGenerator()
anchors = anchor_generator(feature_maps_size, (1024, 1024))
del anchor_generator

# face detector checkpoint path
FACEBOXES_CKPT = 'trained_model/faceboxes/epoch_299_train_lost_4.388730_val_loss3.445847/save'
MIN_FACE_SCORE = 0.8
SCORE_THRESHOLD = 0.9

# face verification ckpt path
ARC_FACE_CKPT='trained_model/pb_model'

TIME_TO_LIVE = 10  # fps
BOUNDING_LIMIT = 20
TRACKING_INTERVAL = 1
IS_RECORD = False

VECTORS_SET_PATH = 'assets/embedding_data/data.pkl'
AUG_VECTORS_SET_PATH = 'assets/embedding_data/aug_data.pkl'

COUNTER_STREAM = 'assets/output_videos/23_06_2020_14_32_44/raw_counter.avi'
GATE_STREAM = 'assets/output_videos/23_06_2020_14_32_44/raw_gate.avi'

