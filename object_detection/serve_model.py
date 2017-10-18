import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

sys.path.append('..')

from collections import defaultdict
from io import StringIO, BytesIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import scipy.misc

from utils import label_map_util
from utils import visualization_utils as vis_util

from flask import Flask, request, jsonify
import subprocess
import uuid
import base64


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('Loading label...')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# helper code.
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK'
    })


@app.route('/', methods=['POST'])
def detect():
    print('Start reading image')
    data = request.json

    uid = str(uuid.uuid4())[:10]
    image_binary = base64.b64decode(data['image'])

    image_f = BytesIO()
    image_f.write(image_binary)
    image_f.seek(0)

    image = Image.open(image_f)
    print('Start detecting object')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
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
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            vis_file = BytesIO()
            scipy.misc.imsave(vis_file, image_np, format='png')
            vis_file.seek(0)
            vis_binary = vis_file.read()

            print('Finished detecting object')
            return jsonify({
                'boxes': list(map(lambda l: [float(x) for x in l], boxes[0])),
                #'scores': list(scores[0]),
                #'classes': list(classes[0]),
                #'num_detections': int(num_detections),
                'vis': base64.b64encode(vis_binary).decode('utf-8'),
            })


if __name__ == '__main__':
    app.run(debug=False, port=5900, host='0.0.0.0')
