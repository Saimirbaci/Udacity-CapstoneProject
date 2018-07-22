#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
from attrdict import AttrDict
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

LIGHT_DB = AttrDict({TrafficLight.RED: "RED",
                     TrafficLight.YELLOW:"YELLOW",
                     TrafficLight.GREEN:"GREEN",
                     TrafficLight.UNKNOWN:"UNKNOWN"})

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2d = None
        self.closest_waypoint_id = 0
        self.last_car_position = 0
        self.last_light_pos_wp = []
        self.waypoints_tree = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.environment_param = rospy.get_param("/traffic_light_environment")
        self.model_path = rospy.get_param("/traffic_light_model_name")
        self.detection_threshhold = rospy.get_param("/traffic_light_detection_threshhold")
        self.max_light_distance = rospy.get_param("/traffic_light_max_distance")

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.TrafficLightState = rospy.Publisher('/traffic_light_state', Int32, queue_size=1)

        self.light_classifier = TLClassifier(self.environment_param, self.model_path, self.detection_threshhold)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        light_color = LIGHT_DB(state)

        rospy.loginfo('Detected Light Color= {} | state= {}'.format(light_color, Int32(state)))
        rospy.loginfo('\n')

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1


    def get_closest_waypoint_light(self, x, y):

        closest_idx = -1
        if self.waypoints_tree is not None:
            closest_idx = self.waypoints_tree.query([x, y], 1)[1]
        return closest_idx


    def get_closest_waypoint(self, pose):

        #TODO implement
        closest_idx = -1
        if self.waypoints is not None:
            closest_idx = self.waypoints_tree.query([pose.position.x, pose.position.y], 1)[1]

            self.closest_waypoint_id = closest_idx
        return closest_idx


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)


    def process_traffic_lights(self):
        closest_light = None
        line_wp_idx = None

        light_positions = self.config['stop_line_positions']

        next_light_distance = float('inf')
        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose)

            for light in light_positions:
                light_wp_idx = self.get_closest_waypoint_light(light[0], light[1])
                car_xy = self.waypoints.waypoints[car_wp_idx].pose.pose.position
                light_xy = self.waypoints.waypoints[light_wp_idx].pose.pose.position

                traffic_light_distance = math.sqrt(((light_xy.x - car_xy.x) ** 2) + ((light_xy.y - car_xy.y) ** 2))
                if traffic_light_distance < next_light_distance and \
                        traffic_light_distance < self.max_light_distance and \
                        car_wp_idx < light_wp_idx:
                    next_light_distance = traffic_light_distance
                    closest_light = light
                    line_wp_idx = light_wp_idx

            rospy.loginfo('Next traffic light obstacle is in = {}'.format(next_light_distance))

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
