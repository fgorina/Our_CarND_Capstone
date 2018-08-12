from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from time import time

debug = True # Change here to disable Image debugging

class TLClassifier(object):
    def __init__(self, encoding='rgb8'):
        #TODO load classifier

        self.detection_graph = self.load_graph(self.get_graph_path())
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.detection_graph)

        self.encoding = encoding
        self.now = None
        self.then = None
        self.elapsed_time = 0.

        if debug == True:
            self.set_colors()
            self.image_debug_pub = rospy.Publisher('/image_debug', Image, queue_size=1)
            self.bridge = CvBridge()

    def get_graph_path(self):
        tf_version = tf.__version__
        path = "../../ssd_bags_20K/frozen_inference_graph.pb"
        if tf_version <= '1.5.0':
            path = "../../ssd_bags_20K_tf1.3/frozen_inference_graph.pb"
        print("Tensorflow ", tf_version , ": using SSD reference model from path ", path)
        return path

    def set_colors(self):
        self.green = [0,255,0]
        if self.encoding == 'rgb8':
            self.red = [255,0,0]
            self.yellow = [255,255,0]
        else:
            self.red = [0,0,255]
            self.yellow = [0,255,255]

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

    def to_image_coords(self, boxes, height, width):
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    # class : 0 -> Unknown 1-> Red 2-> Yellow 3->Green
    # 
    def get_color(image):
        
        h = img.shape[0]
        w = img.shape[1]
    
        dl = int(h/3)
        dx0 = int(h/9)
        dx1 = 2*dx0
        
        red = np.sum(bimg[0:dl-1, dx0:dx1, :])
        orange = np.sum(bimg[dl:(2*dl)-1, dx0:dx1, :])
        green = np.sum(bimg[(dl*2):(dl*3)-1, dx0:dx1, :])
    
    
        if green > orange and green > red:
            color = 3
    
        elif orange > red and orange > green:
            color = 2
    
        elif red > orange and red > green:
            color = 1
    
        else:
            color = 0
    
        return color
    
        


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
        self.then = time()
        
        compute_color = False   # Specifies if we must compute color from the traffic light image

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8)[:, :, 0:3], 0)
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        width, height = image.shape[1], image.shape[0]

        #rospy.logwarn("Max Score " + str(max(scores)))

        confidence_cutoff = 0.7

        (boxes, scores, classes) = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        
        
        # Here we compute light colors manually forgetting the result of the SSD
        
        if compute_color:
        
            for i in range(len(boxes)):
                box = boxes[i]
                bimg = image_np[0, int(box[0]):int(box[2]), int(box[1]):int(box[3]),:]
                color = get_color(bimg)
                classes[i] = color

        detected_value = TrafficLight.UNKNOWN

        box_coords = self.to_image_coords(boxes, height, width)
        number_of_boxes = np.array([0, 0, 0, 0])

        for(box, c, s) in zip(box_coords, classes, scores):
            r = (box[2]-box[0])/(box[3]-box[1])
            if r > 1.8 and (box[3] - box[1]) > 18.0:    # Relation > 1.8 and minim size 18
                number_of_boxes[int(c)] += 1

        

        if len(classes) > 0:
            color = np.argmax(number_of_boxes)
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

        detection = "No detection"
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
                detection = "Red"
                color = self.red
            elif detected_value == TrafficLight.YELLOW:
                detection = "Yellow"
                color = self.yellow
            elif detected_value == TrafficLight.GREEN:
                detection = "Green"
                color = self.green
            cv2.rectangle(img_debug, (y1, x1), (y2, x2), color, thickness=-1)
        alpha = 0.4
        cv2.addWeighted(img_debug, alpha, image, 1 - alpha, 0, img_debug)

        self.now = time()
        self.elapsed_time = self.now - self.then
        label = "%s: %.2fs" % (detection, self.elapsed_time)
        cv2.putText(img_debug, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        new_image = self.bridge.cv2_to_imgmsg(img_debug, encoding=self.encoding)
        self.image_debug_pub.publish(new_image)

