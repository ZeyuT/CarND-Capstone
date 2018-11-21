from styx_msgs.msg import TrafficLight

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import six.moves.urllib as urllib
from collections import defaultdict
from io import StringIO
from glob import glob
import sys

# Uncomment if need visualization detection
# import visualization_utils as vis_util



class TLClassifier(object):
    def __init__(self):
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        PATH_TO_FROZEN_GRAPH = 'light_classification/'+MODEL_NAME+ '/frozen_inference_graph.pb'

	self.number = 0
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
    
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)      

	
    # Gamma transform.
    def gamma_trans(self, img, gamma):
        # resize in input image to speed up saturation calculation.
        res_img = cv2.resize(img, (120, 160))
	hsv = cv2.cvtColor(res_img, cv2.COLOR_BGR2HSV)
	h, w, ch = hsv.shape
 	mean_saturation = 0
	for m in range(0,h):
	   for n in range(0,w):
		mean_saturation += hsv[m,n][1]
	mean_saturation = mean_saturation/(h*w)	
	# Only apply gamma transformation to images with low saturation
	if mean_saturation > 25:
	    return img
	gamma_table = [np.power(x/255.0,gamma) * 255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	return cv2.LUT(img,gamma_table)
    """
    # Adjust contrast and brightness
    def contrast_brightness(self,img, a, g):
        h, w, ch = img.shape
        temp = np.zeros([h, w, ch], img.dtype)
	dst = cv2.addWeighted(img, a, temp, 1-a, g)
	return dst
    """

    def get_classification(self, image,COLOR_THRESHOLD,SCORE_THRESHOLD,gamma):
        """Determines the color of the traffic light in the image
c
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	image = self.gamma_trans(image,gamma)
        # Run inference
        with self.detection_graph.as_default():
             (detection_boxes, detection_scores, detection_classes, num_detections) = self.sess.run(
                 [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                 feed_dict={self.image_tensor: np.expand_dims(image, axis=0)})
                
        # all outputs are float32 numpy arrays, so convert types as appropriate
        detection_boxes=np.squeeze(detection_boxes)
        detection_classes =np.squeeze(detection_classes)
        detection_scores = np.squeeze(detection_scores)
		
	# height and width of box of true detection should be as least 30 pixels. Normaliz threshold to 0~1
	box_height_threshold = 30/image.shape[0]
	box_width_threshold = 30/image.shape[1]		
        idx_vec = [i for i, v in enumerate(detection_classes.tolist()) if ((v == 10.) and 
			(detection_boxes[i][2] - detection_boxes[i][0] > box_height_threshold) and
			(detection_boxes[i][3] - detection_boxes[i][1] > box_width_threshold))]

        true_box = [];
	max_score = 0.0;

	# find the detection with the max score
        if not (len(idx_vec) ==0):
            for idx in idx_vec:
  		if detection_scores[idx] > max_score:
		    max_score = detection_scores[idx]
		    max_score_idx = idx
	    
	    #print('max score: {}'.format(detection_scores[max_score_idx]))
	    # Only consider the detection with score larger than SCORE_THRESHOLD
	    if detection_scores[max_score_idx] > SCORE_THRESHOLD:
                true_box = self.box_normal_to_pixel(detection_boxes[max_score_idx], image.shape[0:2])
        
        # Identify light color
	red_count = 0
	green_count = 0

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
			
		# Uncomment if need red and green color detection
		"""
		# in hsv space, hue value of green is roughly in 35-77
		    if(pixel[0] > 35 and pixel[0] < 77):
			green_count += 1
                """
		"""
	    ## Visualize the detection output and save in 'detected' file##
	    # Only show detection of traffic lights
	    show_classes = []
	    show_boxes = []
 	    show_scores = []
	    for idx in range(0,len(detection_classes)):
	        if detection_classes[idx] == 10:
		    show_classes.append(10)
		    show_boxes.append(detection_boxes[idx])
		    show_scores.append(detection_scores[idx])

	    category_index = {10:{'name': 'traffic light','id':10}}	
	    vis_util.visualize_boxes_and_labels_on_image_array(
	    image,
	    np.array(show_boxes),
	    np.array(show_classes),
	    np.array(show_scores),
	    category_index,
	    use_normalized_coordinates=True,min_score_thresh=SCORE_THRESHOLD,
	    line_thickness=3)
	
	    cv2.imwrite('./processed_image/{}.jpg'.format(self.number),image)
	    self.number += 1
                """
	    # Uncomment if need only red color detection
	    #print("red count:",red_count)
            if red_count > COLOR_THRESHOLD:
                return TrafficLight.RED
            
        return TrafficLight.UNKNOWN

	# Uncomment if need red and green color detection
	"""
 	if red_count > COLOR_THRESHOLD:
             return TrafficLight.RED
         elif green_count > COLOR_THRESHOLD:
 	    return TrafficLight.GREEN
 	else:
             return TrafficLight.UNKNOWN
	"""
