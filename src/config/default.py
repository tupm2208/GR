from src.models.facebox.anchor_generator import AnchorGenerator

feature_maps_size = [[32, 32], [16, 16], [8, 8]]

anchor_generator = AnchorGenerator()
anchors = anchor_generator(feature_maps_size, (1024, 1024))
del anchor_generator

# face detector checkpoint path
FACEBOXES_CKPT = '/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/faceboxes/epoch_299_train_lost_4.388730_val_loss3.445847/save'

# face verification ckpt path
ARC_FACE_CKPT='/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/arc_mbv2_ccrop/e_8_b_40000.ckpt'
