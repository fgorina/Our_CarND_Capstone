import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from PIL import ImageFont
import time
from scipy.stats import norm
import argparse


plt.style.use('ggplot')


# Colors (one for each class)
cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])
LABEL_LIST = [str(i) for i in range(len(cmap))]

#
# Utility funcs
#


def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)

    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes


def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].

    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords

def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    font = ImageFont.truetype(font="xerox.ttf", size=10, index=0, encoding='')
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        label = LABEL_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
        draw.text(((left+right)/2.0, (top+bot)/2.0), "hello", font=font)

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

# img must be a single image of a vertical traffic light
def get_vertical_color(img):

    h = img.shape[0]
    w = img.shape[1]

    dl = int(h/3)
    dx0 = int(h/9)
    dx1 = 2*dx0
    print("dl : ", dl)


    red = np.sum(bimg[0:dl-1, dx0:dx1, :])
    orange = np.sum(bimg[dl:(2*dl)-1, dx0:dx1, :])
    green = np.sum(bimg[(dl*2):(dl*3)-1, dx0:dx1, :])


    if green > orange and green > red:
        color = "Green"

    elif orange > red and orange > green:
        color = "Orange"

    elif red > orange and red > green:
        color = "Red"

    else:
        color = "Unknown"

    print("Color ", color)

    return color


# img must be a single image of a horizontal traffic light

def get_horizontal_color(img):

    h = img.shape[0]
    w = img.shape[1]

    dl = int(w/3)
    dx0 = int(w/9)
    dx1 = 2*dx0

    print("dl : ", dl)


    red = np.sum(bimg[dx0:dx1, 0:dl-1, :])
    orange = np.sum(bimg[dx0:dx1, dl:(2*dl)-1, :])
    green = np.sum(bimg[dx0:dx1, (dl*2):(dl*3)-1, :])


    if green > orange and green > red:
        color = "Green"

    elif orange > red and orange > green:
        color = "Orange"

    elif red > orange and red > green:
        color = "Red"

    else:
        color = "Unknown"

    print("Color ", color)

    return color


def get_color(img):

    if img.shape[0] > img.shape[1]:
        return get_vertical_color(img)
    else:
        return get_horizontal_color(img)

parser = argparse.ArgumentParser()
#parser.add_argument('--graph_file', dest='graph_file', required=True)
parser.add_argument('--image_path', dest='image_path', required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    #GRAPH_FILE = args.graph_file
    image_dir = args.image_path




    GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'
    detection_graph = load_graph(GRAPH_FILE)

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Load a sample image.
    image = Image.open(image_dir)
    no_image = np.asarray(image, dtype=np.uint8)[:, :, 0:3]
    image_np = np.expand_dims(no_image, 0)



    with tf.Session(graph=detection_graph) as sess:
        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                            feed_dict={image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.6
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        draw_boxes(image, box_coords, classes)

        # class=10 is the traffic light

        total_color = "green"
        for (box, c) in zip(box_coords, classes):
            if int(c) == 10:        # Class 10 is traffic light
                bimg = image_np[0, int(box[0]):int(box[2]), int(box[1]):int(box[3]),:]
                color = get_color(bimg)
                print("Color : " , color)
                if color == "orange" and total_color == "green":
                    total_color = "orange"
                elif color == "red":
                    total_color = "red"

        print("Total Color ", color)


        plt.figure(figsize=(12, 8))
        plt.imshow(image)

        #plt.imshow(bimg)

    # Es evident de les proves que el color del llum no es important
    # i sí que ho es la seva posició. Necessitem doncs una funció que:
    # calculi en quin terç esta la màxima intensitat.
