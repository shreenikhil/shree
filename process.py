from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import ImageCompare
import math

execution_path = os.getcwd()
color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}
resized = False
n_X = 0
n_Y = 0
b_trackable = False
array_Subject = []
x_offset = 0
y_offset = 0
n_thresholdBoxSquare = 10
s_img = cv2.imread("icon.png")
position_person = (0,0,0,0)
position_target = (0,0,0,0)
n_frameCountthreshold = 10
n_personCount = 0
n_targetCount = 0
#process mouse event
def getXY(prevx, prevy,curx,cury):
    n_rx = (curx-prevx)
    n_ry = (cury-prevy)
    d = math.sqrt(n_rx * n_rx + n_ry * n_ry)
    if(d <= 0):
        return (prevx,prevy)
    print(n_rx,n_ry)
    k = 10
    x = (int)(prevx + k*(n_rx/d))
    y = (int)(prevy + k*(n_ry/d))
    return (x,y)
def processFrame(output_array,detected_frame):
    global n_X,n_Y,b_trackable,array_Subject
    # activateSubject(output_array,detected_frame)
    displayText(output_array,detected_frame)
    if(b_trackable == False):
        return
    if(len(array_Subject)>1):
        cv2.destroyWindow("Subject")
    b_trackable = False
    n_min = 99999999
    for obj in output_array:
        # print(obj,"checkmehere")
        box_point = obj['box_points']
        if(n_X>box_point[0] and n_X<box_point[2] and n_Y>box_point[1] and n_Y<box_point[3]):
            # print(detected_frame,"checkmedetectedframe")
            #get image which area is min and contain point
            if(n_min > (box_point[2]-box_point[1])*(box_point[3]-box_point[1]) and (box_point[2]-box_point[1])*(box_point[3]-box_point[1])>n_thresholdBoxSquare):
                n_min = (box_point[2]-box_point[1])*(box_point[3]-box_point[1])
                array_Subject = detected_frame[box_point[1]:box_point[3], box_point[0]:box_point[2]]
            pass
    if(len(array_Subject)>0):
        if(array_Subject.shape[0]>0 and array_Subject.shape[1]>0):
            cv2.imshow("Subject",array_Subject)
    return
def activateSubject(output_array,detected_frame):
    global array_Subject
    n_MinNorm = 999999
    array_Similarity = []
    n_x1 ,n_x2,n_y1,n_y2 = (0,0,0,0)
    if(len(array_Subject)<=0):
        cv2.imshow("image",detected_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          return
        return
    for obj in output_array:
        box_point = obj['box_points']
        array_ToCompare = detected_frame[box_point[1]:box_point[3], box_point[0]:box_point[2]]
        #compare to array_subject
        n_m, n_0 = ImageCompare.compare_images(array_ToCompare,array_Subject)
        if(n_MinNorm>n_m and n_m<199999):
            n_MinNorm = n_m
            n_x1 = box_point[0]
            n_x2 = box_point[2]
            n_y1 = box_point[1]
            n_y2 = box_point[3]
            array_Similarity =  array_ToCompare
        else:
            #disapper target.
            pass
        pass
    #move marker to new goal position
    global x_offset,y_offset,s_img
    if((n_x1- n_x2)*(n_y1-n_y2)>n_thresholdBoxSquare):
        array_Subject = array_Similarity
        x_offset, y_offset = getXY(x_offset,y_offset,(int)((n_x1+n_x2)/2 - s_img.shape[0]/2),(int)((n_y1+n_y2)/2 - s_img.shape[1]/2))
        # detected_frame[y_offset:y_offset+s_img.shape[1], x_offset:x_offset+s_img.shape[0]] = s_img
        cv2.circle(detected_frame,(x_offset,y_offset), 13, (255,255,255), 1)
    else:
        #if no goald , then display this text
        cv2.putText(detected_frame,'target disappeared pls select target',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)
        pass
    cv2.imshow("image",detected_frame)
    if(len(array_Similarity)>0):
        cv2.imshow("person",array_Similarity)
        cv2.waitKey(25)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        return
    return
def find_person(output_array):
    global position_person,n_personCount
    b_findPerson = False
    for obj in output_array:
        if(obj['name'] == "person"):
            box_point = obj['box_points']
            position_person = box_point
            b_findPerson = True
            return position_person
            pass
    if(b_findPerson == True):
        return position_person
        n_personCount = 0
    else:
        if(n_personCount > n_frameCountthreshold):
            n_personCount  = 0
            position_person = (0,0,0,0)
        n_personCount += 1
    return position_person

def find_target(output_array,detected_frame):
    global array_Subject , position_target,n_targetCount
    n_MinNorm = 999999
    n_x1 ,n_x2,n_y1,n_y2 = (0,0,0,0)
    b_find = False
    for obj in output_array:
        box_point = obj['box_points']
        array_ToCompare = detected_frame[box_point[1]:box_point[3], box_point[0]:box_point[2]]
        #compare to array_subject
        n_m, n_0 = ImageCompare.compare_images(array_ToCompare,array_Subject)
        if(n_MinNorm>n_m and n_m<199999):
            n_MinNorm = n_m
            if((box_point[0]-box_point[2])*(box_point[1]-box_point[3])>n_thresholdBoxSquare):
                array_Subject = array_ToCompare
                n_x1 = box_point[0]
                n_x2 = box_point[2]
                n_y1 = box_point[1]
                n_y2 = box_point[3]
                b_find = True
                position_target = box_point
                cv2.imshow("same?",array_Subject)
        else:
            #disapper target.
            pass
    if(b_find == True):
        return position_target
    else:
        if(n_targetCount>n_frameCountthreshold):
            position_target = (0,0,0,0)
            n_targetCount = 0
        else:
            n_targetCount += 1
    return position_target
def getTextToDisplay(position_person,position_target):
    if((position_person[0]-position_person[2])*(position_person[1]-position_person[3])<=n_thresholdBoxSquare
        or(position_target[0]-position_target[2])*(position_target[1]-position_target[3])<=n_thresholdBoxSquare ):
        return "pick Object"
    top = position_person[1]
    left = position_person[0]
    bottom = position_person[3]
    right = position_person[2]
    persionCX = (int)((left+right)/2)
    persionCY = (int)((bottom+top)/2)

    targetCY = (int)((position_target[1]+position_target[3])/2)
    targetCX = (int)((position_target[0]+position_target[2])/2)
    if(abs((persionCX-targetCX) * (persionCY-targetCY)) < 70):
        print(position_person,"position_person",position_target,"position_target" ,(persionCX-targetCX) , (targetCY-persionCY))
        return "Good"
        pass
    STR_LEFT_DIRECTION = "LEFT"
    STR_RIGHT_DIRECTION = "RIGHT"
    STR_TOP_DIRECTION = "TOP"
    STR_BOTTOM_DIRECTION = "BOTTOM"
    STR_TOPLEFT_DIRECTION = "TOPLEFT"
    STR_TOPRIGHT_DIRECTION = "TOPRIGHT"
    STR_BOTTOMLEFT_DIRECTION = "BOTTOMLEFT"
    STR_BOTTOMRIGHT_DIRECTION = "BOTTOMRIGHT"

    if targetCX < persionCX:
        directionX = STR_LEFT_DIRECTION
    else:  
        directionX = STR_RIGHT_DIRECTION

    if targetCY < persionCY:
        directionY = STR_TOP_DIRECTION
    else:
        directionY = STR_BOTTOM_DIRECTION

    if top <= targetCY <= bottom:
        return directionX
    if left <= targetCX <= right:
        return directionY
    if directionX == STR_LEFT_DIRECTION and directionY == STR_TOP_DIRECTION:
        return STR_TOPLEFT_DIRECTION
    if directionX == STR_LEFT_DIRECTION and directionY == STR_BOTTOM_DIRECTION:
        return STR_BOTTOMLEFT_DIRECTION
    if directionX == STR_RIGHT_DIRECTION and directionY == STR_TOP_DIRECTION:
        return STR_TOPRIGHT_DIRECTION
    if directionX == STR_RIGHT_DIRECTION and directionY == STR_BOTTOM_DIRECTION:
        return STR_BOTTOMRIGHT_DIRECTION
    pass

def displayText(output_array,detected_frame):
    try:
        if(output_array):
            position_person = find_person(output_array)
            position_target = find_target(output_array,detected_frame)
            text = getTextToDisplay(position_person,position_target)
            # cv2.putText(detected_frame,text,(position_person[0]-40,position_person[1]-40),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
            # cv2.putText(detected_frame,text,((int)(detected_frame.shape[0]/2),(int)(detected_frame.shape[1]/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
            cv2.putText(detected_frame,text,((int)(5),(int)(15)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
    finally:
        cv2.imshow("image",detected_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return
    pass
def click_and_crop(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global n_X,n_Y,b_trackable
        n_X = x
        n_Y = y
        b_trackable = True

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
def frameShowOnlyImage(frame_number,output_array,output_count,detected_frame):
    processFrame(output_array,detected_frame)
def forFrame(frame_number, output_array, output_count,detected_frame):
    # print("FOR FRAME " , frame_number)
    # print("Output for each object : ", output_array)
    # print("Output count for unique objects : ", output_count)
    # print("------------END OF A FRAME --------------")
    frameShowOnlyImage(frame_number, output_array, output_count,detected_frame)

def forSeconds(second_number, output_arrays, count_arrays, average_output_count,detected_frame):
    # print("SECOND : ", second_number)
    # print("Array for the outputs of each frame ", output_arrays)
    # print("Array for output count for unique objects in each frame : ", count_arrays)
    # print("Output average count for unique objects in the last second: ", average_output_count)
    # print("------------END OF A SECOND --------------")
    pass

def forMinute(minute_number, output_arrays, count_arrays, average_output_count,detected_frame):
    # print("MINUTE : ", minute_number)
    # print("Array for the outputs of each frame ", output_arrays)
    # print("Array for output count for unique objects in each frame : ", count_arrays)
    # print("Output average count for unique objects in the last minute: ", average_output_count)
    # print("------------END OF A MINUTE --------------")
    pass
def show():
    plt.show()
def endProcess():
    cv2.destroyAllWindows()