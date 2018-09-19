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
import sys


class TLClassifier(object):
    def __init__(self):
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        PATH_TO_FROZEN_GRAPH = 'light_classification/'+MODEL_NAME+ '/frozen_inference_graph.pb'

	# configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # load a frozen Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
            self.sess = tf.Session(graph = self.detection_graph, config =  config)

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
	
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')
        
        
    def load_image_into_numpy_array(self,image):
        (im_width, im_height,im_depth) = image.shape
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)      

    def get_classification(self, image, RED_THRESHOLD):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Run inference
        with self.detection_graph.as_default():
             (detection_boxes, detection_scores, detection_classes, num_detections) = self.sess.run(
                 [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                 feed_dict={self.image_tensor: np.expand_dims(image, axis=0)})
                
        # all outputs are float32 numpy arrays, so convert types as appropriate
        detection_boxes=np.squeeze(detection_boxes)
        detection_classes =np.squeeze(detection_classes)
        detection_scores = np.squeeze(detection_scores)
        
        idx_vec = [i for i, v in enumerate(detection_classes.tolist()) if ((v == 10.) and (detection_scores[i] > 0.3))]
        
        true_box = [];
        if not (len(idx_vec) ==0):
            for idx in idx_vec:
                dim = image.shape[0:2]
                box = self.box_normal_to_pixel(detection_boxes[idx], dim)
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                
                #Consider boxes smaller than 20 pixels to be false positive
                if ((box_h>20) and (box_w>20)):
                    
                    #Only consider the first true detection
                    true_box = box
                    break;
        
        # Identify light color
        red_count = 0;

        if len(true_box):
            # crop the input image       
            resize_img = cv2.resize(image[true_box[0]:true_box[2], true_box[1]:true_box[3]], (32, 32))
            # convert to hsv colorspace, to reduce the negative effect of bightness 
            HSV_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)
	    
	    pixel = 0
            for row in range(32):
		for col in range(32):
		    pixel = HSV_img[row,col]
                # in hsv space, hue value of red is roughly in 160-180 and 0-8
                    if((pixel[0] > 160 and pixel[0] < 180) or (pixel[0] > 0 and pixel[0] < 8)):
                        red_count += 1
            
	    print('red count:',red_count)
            if red_count > RED_THRESHOLD:
                return TrafficLight.RED
		print ('detect red light')
            
        return TrafficLight.UNKNOWN
