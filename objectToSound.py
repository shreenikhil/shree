# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:46:08 2019
From https://gist.github.com/tedmiston/6060034
@author: kms
"""

# Not the best way to edit the path, but okay for a one-off test
import sys
sys.path.insert(0, 'C:\\Users\\Shree Nikhil MC\\Downloads\\models\\research\\object_detection')
sys.path.insert(0, 'C:\\Users\\Shree Nikhil MC\\Downloads\\models\\research')
#sys.path.insert(0, 'E:\\research\\object_detection')
#sys.path.insert(0, 'E:\\research')
#import matplotlib; #matplotlib.use('Agg')
import numpy as np
#import os
#import six.moves.urllib as urllib
#import tarfile
import tensorflow as tf
#import zipfile
import cv2    # I did pip install opencv-python to get this

#from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 

#import speech_recognition as sr
#import pyglet
#import time
'''
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util
'''

search_item="person"


def set_base_directories_default():
    '''Set the model, label and test directories to those under the current working directory
    default names are used assuming the setup is the same each time.
    At time of writing, working directory is C:/tfmodels
    See the settings below for actual directory names.
    Use set_base_directories() to set the paths to something different'''
    global MODEL_BASE
    global LABEL_BASE
    global TEST_BASE
    MODEL_BASE="E:\\project\\frozen_models"
    LABEL_BASE="E:\\project\\label_maps"
  #  TEST_BASE="C:/tfmodels/test_images"


def load_inference_graph(model_name):
    # Load detection graph into memory
    model_path = MODEL_BASE+'/'+model_name+'/frozen_inference_graph.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_label_map(dataset):
    '''Load the label map for the named data set'''
    datasets={"MSCOCO":{"file":"mscoco_label_map.pbtxt","classes":90}}
    if not dataset in datasets:
        print("No data set called",dataset)
        return None
    
    label_file=datasets[dataset]["file"]
    num_classes=datasets[dataset]["classes"]
        
    path_to_labels=LABEL_BASE+"/"+label_file
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

# Function to load an image into a numpy array
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)




def continuous_inference(graph,camix=0):
  cam = cv2.VideoCapture(camix)
  ret_val, image = cam.read()
  global notated  
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      
      # Run inference
      while True:
        ret_val, image = cam.read()
        output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})
    
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      
        horz,vert=find_item(image,output_dict,search_item)
        if horz != None:
            sound_at_location(horz,vert)
        notated=notate_image(image,output_dict)
        cv2.imshow('my webcam', notated)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
  cv2.destroyAllWindows()

# Function to run detection on a single image with the given inference graph
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def summarise_output(output_dict):
    num_detections=output_dict['num_detections']
    print("File: "+output_dict['file'])
    print("Found {0} objects".format(num_detections))
    for i in range(num_detections):
        classix=output_dict['detection_classes'][i]
        score=output_dict['detection_scores'][i]
        objname=category_index[classix]['name']
        print(classix,objname,score)

def inference_image_file(image_path):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    output_dict['file']=image_path
    return output_dict

def show_webcam(mirror=True):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def notated_webcam():
    cam = cv2.VideoCapture(0)
    ret_val, img = cam.read()
    img = cv2.flip(img, 1)
    cv2.imshow('my webcam', img)
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)
        if cv2.waitKey(1) == 97:
            output_dict=run_inference_for_single_image(img, detection_graph)
            notated=notate_image(img,output_dict)
            horz,vert=find_item(img,output_dict,"bottle")
            if horz != None:
                sound_at_location(horz,vert)
            cv2.imshow('my webcam', notated)
            cv2.waitKey(10000)
        else:
            cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def capture_and_notate():
    img=webcam_capture()
    output_dict = run_inference_for_single_image(img, detection_graph)
    output_dict['file']="Web cam image"        
    summarise_output(output_dict)
    notated=notate_image(img,output_dict)
    cv2.imshow('Notated', notated)

def webcam_capture():
    cam = cv2.VideoCapture(0)
    ret_val, img = cam.read()
    cam.release()
    return img

def notate_image(image_np,output_dict):
    image_copy = image_np.copy()
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_copy,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=1)
    return image_copy

def find_item(img,output_dict,item):   
    num_detections=output_dict['num_detections']
    for i in range(num_detections):
        classix=output_dict['detection_classes'][i]
        objname=category_index[classix]['name']
        score=output_dict['detection_scores'][i]
        if(objname==item and score>0.5):
            pos=output_dict['detection_boxes'][i]
            print("Found",item,"at",pos)
            return pos[1]*20-10, round(10-pos[0]*20)   # First X coordinate scaled -10 to 10
    return None,None

def note_to_freq(note):
    '''Convert note to frequency where middle A (A4) A=0, A#=1, B=2 etc.'''
    return 440 * pow(2, note / 12)
'''
def sound_at_location(horz,vert):
    hz=note_to_freq(vert)
   # sin = pyglet.media.procedural.Digitar(2,hz)
    s=pyglet.media.Player()
    s.queue(sin)
    spos=[horz,0,0]
    s.position=(spos)
    s.play()

# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    global search_item,pi,notated
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    try:
        #word=recognizer.recognize_google(audio)
        word=r.recognize_sphinx(audio,keyword_entries=[['bottle',1],['cup',1],['save',1]])
        word=word.split(" ")[0]
    except sr.UnknownValueError:
        print("Didn't get word")
        word=""
    except sr.RequestError:
        print("Didn't get word")
        word=""
    
    print("Heard",word)
    
    #if word=="save":
    
    fn="shot"+str(pi)+".png"
    pi=pi+1
    cv2.imwrite(fn,notated)
    print("Saved",fn)
    
    if(word=="cap" or word=="cup"):
        search_item="cup"
    if(word=="bottle"):
        search_item="bottle"
    if(word=="brush" or word=="toothbrush"):
        search_item="toothbrush"
    print("Looking for",search_item)

cam = cv2.VideoCapture(0)
pi=0
ret_val, notated = cam.read()
r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

'''

set_base_directories_default()
#detection_graph=load_inference_graph("mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28")
detection_graph=load_inference_graph("ssd_mobilenet_v1_coco_2017_11_17")
category_index=load_label_map("MSCOCO")

# start listening in the background (note that we don't have to do this inside a `with` statement)
#stop_listening = r.listen_in_background(m, callback)
# `stop_listening` is now a function that, when called, stops background listening


continuous_inference(detection_graph)


stop_listening(wait_for_stop=False)
print("All Done")



#show_webcam()
#capture_and_notate()
#print("Ready")
#notated_webcam()
