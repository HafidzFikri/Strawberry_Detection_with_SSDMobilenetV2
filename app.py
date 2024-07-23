import tensorflow as tf
import numpy as np
import PIL
import cv2
import object_detection
import matplotlib.pyplot as plt
import os

from object_detection.utils import config_util, dataset_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


model_dir = 'C:\\Users\\hafid\\Downloads\\Strawberry_Detection_with_SSDMobilenetV2\\'
configs = config_util.get_configs_from_pipeline_file(os.path.join(model_dir,'checkpoint\\pipeline.config'))


model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'checkpoint\\ckpt-21')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(f'{model_dir}\\checkpoint\\label.pbtxt')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    height, width, _ = image_np.shape
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Membuat tensor input dengan dimensi yang benar
    input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)

    # Memanggil fungsi deteksi
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Hitung jumlah strawberry matang dan belum matang
    matang_count = 0
    belum_matang_count = 0
    for i in range(num_detections):
        class_id = detections['detection_classes'][i] + label_id_offset
        score = detections['detection_scores'][i]
        if score > 0.3:  # Threshold untuk memastikan hanya deteksi dengan skor tinggi yang dihitung
            if class_id == 1:  # Asumsikan 1 adalah ID untuk strawberry matang
                matang_count += 1
            elif class_id == 2:  # Asumsikan 2 adalah ID untuk strawberry belum matang
                belum_matang_count += 1

    # Visualisasi deteksi pada frame
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=100,
                min_score_thresh=.3,
                agnostic_mode=False)

    # Tampilkan jumlah deteksi pada frame
    cv2.putText(image_np_with_detections, f'Matang: {matang_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image_np_with_detections, f'Belum Matang: {belum_matang_count}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
