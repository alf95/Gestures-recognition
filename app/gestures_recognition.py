import numpy as np
import cv2
import time
import pyautogui
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from pynput.mouse import Button, Controller

mouse = Controller()


MODEL_NAME = '../training_demo/trained-inference-graphs/output_inference_graph_v1.pb'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '../training_demo/annotations/label_map.pbtxt'

# Number of classes to detect
NUM_CLASSES = 2

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


width_screen, height_screen = pyautogui.size()














capture = cv2.VideoCapture(1)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
width_frame = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height_frame = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
min_score_thresh = .9
max_boxes_to_draw = 1
c_x_new = 0
c_y_new = 0
# Detection


def getCoordinatesRettangle(boxes, scores, max_boxes_to_draw, min_score_thresh):
    box = tuple()
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
    return box


def getCenterCoordinates(box):
    ymin, xmin, ymax, xmax = box
    xmin = xmin * width_frame
    xmax = xmax * width_frame
    ymin = ymin * height_frame
    ymax = ymax * height_frame
    # x, y = (result['topleft']['x'], result['topleft']['y'])
    # x1, y1 = (result['bottomright']['x'], result['bottomright']['y'])
    # lenght of the side parallel to the axis x
    # a = abs(x1 - x)
    # lenght of the side parallel to the axis y
    # b = abs(y - y1)
    # coordinates of the center of the rectangle
    c_x = (xmin + xmax) / 2
    c_y = (ymin + ymax) / 2
    coordinates = {'center_x': c_x, 'center_y': c_y}
    return coordinates


def getLabelPrediction(boxes, scores, classes, category_index, max_boxes_to_draw, min_score_thresh):
    class_name =  ""
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
    return class_name


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            stime = time.time()

            #reset center coordinates
            c_x_old = c_x_new
            c_y_old = c_y_new

            # Read frame from camera
            ret, image_np = capture.read()
            image_np = cv2.flip(image_np, 1)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes_resized = np.squeeze(boxes)
            scores_resized =  np.squeeze(scores)
            classes_resized = np.squeeze(classes).astype(np.int32)
            label_prediction = ""

            box = getCoordinatesRettangle(boxes_resized, scores_resized, max_boxes_to_draw, min_score_thresh)
            if(box.__len__() != 0):
                new_coordinates = getCenterCoordinates(box)
                label_prediction = getLabelPrediction(boxes_resized, scores_resized, classes_resized, category_index, max_boxes_to_draw, min_score_thresh)

            if (label_prediction == 'five'):
                c_x_new = new_coordinates['center_x']
                c_y_new = new_coordinates['center_y']
            elif(label_prediction == 'fist'):
                c_x_new = c_x_old
                c_y_new = c_y_old


            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes_resized,
                    classes_resized,
                    scores_resized,
                    category_index,
                    min_score_thresh=min_score_thresh,
                    max_boxes_to_draw=max_boxes_to_draw,
                    use_normalized_coordinates=True,
                    line_thickness=8)

            # Display output
            if(label_prediction == 'five'):
                # pyautogui.moveRel((c_x_new - c_x_old) * width_screen / width_frame, (c_y_new - c_y_old) * height_screen / height_frame, duration=(time.time() - stime))
                #pyautogui.moveTo(c_x_new * width_screen / width_frame, c_y_new * height_screen / height_frame, duration=0)
                mouse._position_set([c_x_new * width_screen / width_frame,c_y_new * height_screen / height_frame])
            elif(label_prediction == 'fist'):
                pyautogui.hotkey('volumeup')
            cv2.imshow('object detection', image_np)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

capture.release()
cv2.destroyAllWindows()








# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



