import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import collections
import statistics
import math
import tarfile
import os.path

from threading import Lock, Thread
from time import sleep

import cv2

# ZED imports
import pyzed.sl as sl

sys.path.append('utils')

# ## Object detection imports
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_depth_into_numpy_array(depth):
    ar = depth.get_data()
    ar = ar[:, :, 0:4]
    (im_height, im_width, channels) = depth.get_data().shape
    return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)


lock = Lock()
width = 704
height = 416
confidence = 0.35

image_np_global = np.zeros([width, height, 3], dtype=np.uint8)
depth_np_global = np.zeros([width, height, 4], dtype=np.float)

exit_signal = False
new_data = False


# ZED image capture thread function
def capture_thread_func(svo_filepath=None):
    global image_np_global, depth_np_global, exit_signal, new_data

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    input_type = sl.InputType()
    if svo_filepath is not None:
        input_type.set_from_svo_file(svo_filepath)

    init_params = sl.InitParameters(input_t=input_type)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.svo_real_time_mode = False


    # Open the camera
    err = zed.open(init_params)
    print(err)
    while err != sl.ERROR_CODE.SUCCESS:
        err = zed.open(init_params)
        print(err)
        sleep(1)

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    image_size = sl.Resolution(width, height)

    while not exit_signal:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT, resolution=image_size)
            zed.retrieve_measure(depth_mat, sl.MEASURE.XYZRGBA, resolution=image_size)
            lock.acquire()
            image_np_global = load_image_into_numpy_array(image_mat)
            depth_np_global = load_depth_into_numpy_array(depth_mat)
            new_data = True
            lock.release()

        sleep(0.01)

    zed.close()


def display_objects_distances(image_np, depth_np, num_detections, boxes_, classes_, scores_, category_index):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)

    research_distance_box = 30

    for i in range(num_detections):
        if scores_[i] > confidence:
            box = tuple(boxes_[i].tolist())
            if classes_[i] in category_index.keys():
                class_name = category_index[classes_[i]]['name']
            display_str = str(class_name)
            if not display_str:
                display_str = '{}%'.format(int(100 * scores_[i]))
            else:
                display_str = '{}: {}%'.format(display_str, int(100 * scores_[i]))

            # Find object distance
            ymin, xmin, ymax, xmax = box
            x_center = int(xmin * width + (xmax - xmin) * width * 0.5)
            y_center = int(ymin * height + (ymax - ymin) * height * 0.5)
            x_vect = []
            y_vect = []
            z_vect = []

            min_y_r = max(int(ymin * height), int(y_center - research_distance_box))
            min_x_r = max(int(xmin * width), int(x_center - research_distance_box))
            max_y_r = min(int(ymax * height), int(y_center + research_distance_box))
            max_x_r = min(int(xmax * width), int(x_center + research_distance_box))

            if min_y_r < 0: min_y_r = 0
            if min_x_r < 0: min_x_r = 0
            if max_y_r > height: max_y_r = height
            if max_x_r > width: max_x_r = width

            for j_ in range(min_y_r, max_y_r):
                for i_ in range(min_x_r, max_x_r):
                    z = depth_np[j_, i_, 2]
                    if not np.isnan(z) and not np.isinf(z):
                        x_vect.append(depth_np[j_, i_, 0])
                        y_vect.append(depth_np[j_, i_, 1])
                        z_vect.append(z)

            if len(x_vect) > 0:
                x = statistics.median(x_vect)
                y = statistics.median(y_vect)
                z = statistics.median(z_vect)
                
                distance = math.sqrt(x * x + y * y + z * z)

                display_str = display_str + " " + str('% 6.2f' % distance) + " m "
                box_to_display_str_map[box].append(display_str)
                box_to_color_map[box] = vis_util.STANDARD_COLORS[classes_[i] % len(vis_util.STANDARD_COLORS)]

    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box

        vis_util.draw_bounding_box_on_image_array(
            image_np,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=4,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=True)

    return image_np


def main(args):
    svo_filepath = None
    if len(args) > 1:
        svo_filepath = args[1]

    # This main thread will run the object detection, the capture thread is loaded later

    # What model to download and load
    #MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
    MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    #MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    #MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
    #MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # Accurate but heavy

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = 'data/' + MODEL_NAME + '/frozen_inference_graph.pb'

    # Check if the model is already present
    if not os.path.isfile(PATH_TO_FROZEN_GRAPH):
        print("Downloading model " + MODEL_NAME + "...")

        MODEL_FILE = MODEL_NAME + '.tar.gz'
        MODEL_PATH = 'data/' + MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_PATH)
        tar_file = tarfile.open(MODEL_PATH)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, 'data/')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    # Start the capture thread with the ZED input
    print("Starting the ZED")
    capture_thread = Thread(target=capture_thread_func, kwargs={'svo_filepath': svo_filepath})
    capture_thread.start()
    # Shared resources
    global image_np_global, depth_np_global, new_data, exit_signal

    # Load a (frozen) Tensorflow model into memory.
    print("Loading model " + MODEL_NAME)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Limit to a maximum of 50% the GPU memory usage taken by TF https://www.tensorflow.org/guide/using_gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Detection
    with detection_graph.as_default():
        with tf.Session(config=config, graph=detection_graph) as sess:
            while not exit_signal:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                if new_data:
                    lock.acquire()
                    image_np = np.copy(image_np_global)
                    depth_np = np.copy(depth_np_global)
                    new_data = False
                    lock.release()

                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    num_detections_ = num_detections.astype(int)[0]

                    # Visualization of the results of a detection.
                    image_np = display_objects_distances(
                        image_np,
                        depth_np,
                        num_detections_,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index)

                    cv2.imshow('ZED object detection', cv2.resize(image_np, (width, height)))
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        exit_signal = True
                else:
                    sleep(0.01)

            sess.close()

    exit_signal = True
    capture_thread.join()


if __name__ == '__main__':
    main(sys.argv)
