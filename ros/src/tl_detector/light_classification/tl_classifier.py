from styx_msgs.msg import TrafficLight

import rospy
import os
import numpy      as np
import tensorflow as tf
from attrdict import AttrDict
import time


LIGHT_ID_TO_NAME = AttrDict({2: "Red",
                     3:"Yellow",
                     1:"Green",
                     4:"Unknown"})

class TLClassifier(object):
    def __init__(self, environment, model_name, thresh=0.4):
        #TODO load classifier

        self.init_light = TrafficLight.UNKNOWN
        self.detection_threshhold = thresh

        curr_dir = os.path.dirname(os.path.realpath(__file__))

        model_path = os.path.join(curr_dir, model_name)

        self.image_np_deep = None
        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


        rospy.loginfo("Loading SSD Model for detecting traffic ligths for {}".format(environment))
        start = time.time()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        end = time.time()
        rospy.loginfo("Model load time is = {} sec".format(end - start))

    def do_infer(self, image):

        image_expanded = np.expand_dims(image, axis=0)
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        return boxes, scores, classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        self.current_light = self.init_light
        boxes, scores, classes = self.do_infer(image)

        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > self.detection_threshhold:
                class_name = LIGHT_ID_TO_NAME[classes[i]]


                if class_name == 'Red':
                    self.current_light = TrafficLight.RED
                elif class_name == 'Green':
                    self.current_light = TrafficLight.GREEN
                elif class_name == 'Yellow':
                    self.current_light = TrafficLight.YELLOW
                elif class_name == 'Unknown':
                    self.current_light = TrafficLight.UNKNOWN

                self.image_np_deep = image

        return self.current_light