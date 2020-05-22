import tensorflow as tf
import pathlib
import numpy as np
import time


def convert_tflite():
    converter = tf.lite.TFLiteConverter.from_saved_model(
        '/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/pb_model')
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()

    tflite_models_dir = pathlib.Path("/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/")
    tflite_model_quant_file = tflite_models_dir / "model.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model)


def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=str('/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/model.tflite'))


    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.resize_tensor_input(input_index, [1, 112, 112, 3])
    interpreter.allocate_tensors()
    img = np.zeros((1, 112, 112, 3), dtype='float32')

    for e in range(10):
        t1 = time.time()
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)
        # print(predictions.shape)
        print(1/(time.time()-t1))

def test_model():
    model = tf.saved_model.load('/home/tupm/HDD/projects/3dface/facial_verification_system/trained_model/pb_model')

    img = np.zeros((10, 112, 112, 3), dtype='float32')

    for e in range(10):
        t1 = time.time()
        model(img)
        # print(predictions.shape)
        print(1 / (time.time() - t1))

# load_tflite_model()
test_model()