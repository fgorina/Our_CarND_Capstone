from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

debug = True # Change here to disable Image debugging

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.detection_graph = self.load_graph(self.get_graph_path())
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.detection_graph)

        if debug == True:
            self.image_debug_pub = rospy.Publisher('/image_debug', Image, queue_size=1)
            self.bridge = CvBridge()

    def get_graph_path(self):
        tf_version = tf.__version__
        path = "../../ssd_bags_20K/frozen_inference_graph.pb"
        if tf_version <= '1.3.0':
            path = "../../ssd_bags_20K_tf1.3/frozen_inference_graph.pb"
        print("Tensorflow ", tf_version , ": using SSD reference model from path ", path)
        return path

    def filter_boxes(self, min_score, boxes, scores, classes):
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        return filtered_boxes, filtered_scores, filtered_classes


    def load_graph(self, graph_file):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic lightA

        

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8)[:, :, 0:3], 0)
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        #rospy.logwarn("Max Score " + str(max(scores)))

        confidence_cutoff = 0.6
        (boxes, scores, classes) = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        detected_value = TrafficLight.UNKNOWN

        if len(classes) > 0:
            color = min(classes)
            #rospy.logwarn("Image color " + str(classes)+ "  " + str(color))

	    if color == 3:
                detected_value = TrafficLight.GREEN
            elif color == 2:
                detected_value = TrafficLight.YELLOW
            else:
                detected_value = TrafficLight.RED

        if debug == True:
            self.show_detection(image, boxes, detected_value)
        return detected_value

    def show_detection(self, image, boxes, detected_value):
        """Display the detection as a new topic with the associated color"""

        img_debug = np.copy(image)
        for i, box in enumerate(boxes):
            x1 = int(box[0] * image.shape[0])
            y1 = int(box[1] * image.shape[1])
            x2 = int(box[2] * image.shape[0])
            y2 = int(box[3] * image.shape[1])

            MIX_IMAGE_HEIGHT = 50 # too small detection, discard it
            if image.shape[0] < MIX_IMAGE_HEIGHT:
                continue

            color = [255,255,255]
            if detected_value == TrafficLight.RED:
                color = [0,0,255]
            elif detected_value == TrafficLight.YELLOW:
                color = [0,255,255]
            elif detected_value == TrafficLight.GREEN:
                color = [0,255,0]
            cv2.rectangle(img_debug, (y1, x1), (y2, x2), color, thickness=-1)
        alpha = 0.4
        cv2.addWeighted(img_debug, alpha, image, 1 - alpha, 0, img_debug)

        new_image = self.bridge.cv2_to_imgmsg(img_debug, encoding='bgr8')
        self.image_debug_pub.publish(new_image)

