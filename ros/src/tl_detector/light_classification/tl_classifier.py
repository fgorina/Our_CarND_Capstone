from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.detection_graph = self.load_graph("../../ssd_bags_20K/frozen_inference_graph.pb")
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.detection_graph)

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

        #rospy.logwarn("Received Image")
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8)[:, :, 0:3], 0)
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        #rospy.logwarn("Max Score " + str(max(scores)))

        confidence_cutoff = 0.6
        (boxes, scores, classes) = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        if len(classes) > 0:
            color = min(classes)
            # rospy.logwarn("Image color " + str(classes)+ "  " + str(color))

	    if color == 3:
                return TrafficLight.GREEN
            elif color == 2:
                return TrafficLight.YELLOW
            else:
                return TrafficLight.RED

        return TrafficLight.UNKNOWN


