from styx_msgs.msg import TrafficLight

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import six.moves.urllib as urllib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from glob import glob

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
