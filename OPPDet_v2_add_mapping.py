import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv
import argparse
import sys
import os.path
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class Measurement_Map:
    def __init__(self):
        self.in_center = (0, 0)
        self.out_center = (0, 0)
        self.cell_size = 0
        self.InCell = [[0 for cols in range(4)] for rows in range(86)]
        self.OutCell = [[0 for cols in range(4)] for rows in range(93)]
        for i in range(0, 86):
            self.InCell[i] = [0, 0, 0, 0]
        for i in range(0, 93):
            self.OutCell[i] = [0, 0, 0, 0]

    def set_in_cell(self, size, x, y):
        x = x - (size / 2)
        y = y - (size / 2)
        self.InCell[0] = [x - size, y - size, 3 * size, 2 * size]
        self.InCell[1] = [x + (9 * size), y, size, size]
        self.InCell[2] = [x + (8 * size), y - size, size, size]
        self.InCell[3] = [x + (8 * size), y, size, size]
        self.InCell[4] = [x + (7 * size), y - (2 * size), size, size]
        self.InCell[5] = [x + (7 * size), y - (size), size, size]
        self.InCell[6] = [x + (7 * size), y, size, size]
        self.InCell[7] = [x + (6 * size), y - (3 * size), size, size]
        self.InCell[8] = [x + (6 * size), y - (2 * size), size, size]
        self.InCell[9] = [x + (6 * size), y - (size), size, size]
        self.InCell[10] = [x + (6 * size), y, size, size]
        self.InCell[11] = [x + (5 * size), y - (4 * size), size, size]
        self.InCell[12] = [x + (5 * size), y - (3 * size), size, size]
        self.InCell[13] = [x + (5 * size), y - (2 * size), size, size]
        self.InCell[14] = [x + (5 * size), y - (size), size, size]
        self.InCell[15] = [x + (5 * size), y, size, size]
        self.InCell[16] = [x + (4 * size), y - (5 * size), size, size]
        self.InCell[17] = [x + (4 * size), y - (4 * size), size, size]
        self.InCell[18] = [x + (4 * size), y - (3 * size), size, size]
        self.InCell[19] = [x + (4 * size), y - (2 * size), size, size]
        self.InCell[20] = [x + (4 * size), y - (size), size, size]
        self.InCell[21] = [x + (4 * size), y, size, size]
        self.InCell[22] = [x + (3 * size), y - (6 * size), size, size]
        self.InCell[23] = [x + (3 * size), y - (5 * size), size, size]
        self.InCell[24] = [x + (3 * size), y - (4 * size), size, size]
        self.InCell[25] = [x + (3 * size), y - (3 * size), size, size]
        self.InCell[26] = [x + (3 * size), y - (2 * size), size, size]
        self.InCell[27] = [x + (3 * size), y - (size), size, size]
        self.InCell[28] = [x + (3 * size), y, size, size]
        self.InCell[29] = [x + (2 * size), y - (6 * size), size, size]
        self.InCell[30] = [x + (2 * size), y - (5 * size), size, size]
        self.InCell[31] = [x + (2 * size), y - (4 * size), size, size]
        self.InCell[32] = [x + (2 * size), y - (3 * size), size, size]
        self.InCell[33] = [x + (2 * size), y - (2 * size), size, size]
        self.InCell[34] = [x + (2 * size), y - (size), size, size]
        self.InCell[35] = [x + (2 * size), y, size, size]
        self.InCell[36] = [x + (size), y - (6 * size), size, size]
        self.InCell[37] = [x + (size), y - (5 * size), size, size]
        self.InCell[38] = [x + (size), y - (4 * size), size, size]
        self.InCell[39] = [x + (size), y - (3 * size), size, size]
        self.InCell[40] = [x + (size), y - (2 * size), size, size]
        self.InCell[41] = [x, y - (6 * size), size, size]
        self.InCell[42] = [x, y - (5 * size), size, size]
        self.InCell[43] = [x, y - (4 * size), size, size]
        self.InCell[44] = [x, y - (3 * size), size, size]
        self.InCell[45] = [x, y - (2 * size), size, size]
        self.InCell[46] = [x - (size), y - (6 * size), size, size]
        self.InCell[47] = [x - (size), y - (5 * size), size, size]
        self.InCell[48] = [x - (size), y - (4 * size), size, size]
        self.InCell[49] = [x - (size), y - (3 * size), size, size]
        self.InCell[50] = [x - (size), y - (2 * size), size, size]
        self.InCell[51] = [x - (2 * size), y - (6 * size), size, size]
        self.InCell[52] = [x - (2 * size), y - (5 * size), size, size]
        self.InCell[53] = [x - (2 * size), y - (4 * size), size, size]
        self.InCell[54] = [x - (2 * size), y - (3 * size), size, size]
        self.InCell[55] = [x - (2 * size), y - (2 * size), size, size]
        self.InCell[56] = [x - (2 * size), y - (size), size, size]
        self.InCell[57] = [x - (2 * size), y, size, size]
        self.InCell[58] = [x - (3 * size), y - (6 * size), size, size]
        self.InCell[59] = [x - (3 * size), y - (5 * size), size, size]
        self.InCell[60] = [x - (3 * size), y - (4 * size), size, size]
        self.InCell[61] = [x - (3 * size), y - (3 * size), size, size]
        self.InCell[62] = [x - (3 * size), y - (2 * size), size, size]
        self.InCell[63] = [x - (3 * size), y - (size), size, size]
        self.InCell[64] = [x - (3 * size), y, size, size]
        self.InCell[65] = [x - (4 * size), y - (5 * size), size, size]
        self.InCell[66] = [x - (4 * size), y - (4 * size), size, size]
        self.InCell[67] = [x - (4 * size), y - (3 * size), size, size]
        self.InCell[68] = [x - (4 * size), y - (2 * size), size, size]
        self.InCell[69] = [x - (4 * size), y - (size), size, size]
        self.InCell[70] = [x - (4 * size), y, size, size]
        self.InCell[71] = [x - (5 * size), y - (4 * size), size, size]
        self.InCell[72] = [x - (5 * size), y - (3 * size), size, size]
        self.InCell[73] = [x - (5 * size), y - (2 * size), size, size]
        self.InCell[74] = [x - (5 * size), y - (size), size, size]
        self.InCell[75] = [x - (5 * size), y, size, size]
        self.InCell[76] = [x - (6 * size), y - (3 * size), size, size]
        self.InCell[77] = [x - (6 * size), y - (2 * size), size, size]
        self.InCell[78] = [x - (6 * size), y - (size), size, size]
        self.InCell[79] = [x - (6 * size), y, size, size]
        self.InCell[80] = [x - (7 * size), y - (2 * size), size, size]
        self.InCell[81] = [x - (7 * size), y - (size), size, size]
        self.InCell[82] = [x - (7 * size), y, size, size]
        self.InCell[83] = [x - (8 * size), y - (size), size, size]
        self.InCell[84] = [x - (8 * size), y, size, size]
        self.InCell[85] = [x - (9 * size), y, size, size]

    def set_out_cell(self, size, x, y):
        x = x - (size / 2)
        y = y - (size / 2)
        self.OutCell[0] = [x - size, y - size, 3 * size, 3 * size]
        self.OutCell[1] = [x + (9 * size), y, size, size]
        self.OutCell[2] = [x + (8 * size), y - size, size, size]
        self.OutCell[3] = [x + (8 * size), y, size, size]
        self.OutCell[4] = [x + (8 * size), y + (size), size, size]
        self.OutCell[5] = [x + (7 * size), y - (2 * size), size, size]
        self.OutCell[6] = [x + (7 * size), y - (size), size, size]
        self.OutCell[7] = [x + (7 * size), y, size, size]
        self.OutCell[8] = [x + (7 * size), y + size, size, size]
        self.OutCell[9] = [x + (6 * size), y - (3 * size), size, size]
        self.OutCell[10] = [x + (6 * size), y - (2 * size), size, size]
        self.OutCell[11] = [x + (6 * size), y - (size), size, size]
        self.OutCell[12] = [x + (6 * size), y, size, size]
        self.OutCell[13] = [x + (6 * size), y + (size), size, size]
        self.OutCell[14] = [x + (5 * size), y - (4 * size), size, size]
        self.OutCell[15] = [x + (5 * size), y - (3 * size), size, size]
        self.OutCell[16] = [x + (5 * size), y - (2 * size), size, size]
        self.OutCell[17] = [x + (5 * size), y - (size), size, size]
        self.OutCell[18] = [x + (5 * size), y, size, size]
        self.OutCell[19] = [x + (5 * size), y + size, size, size]
        self.OutCell[20] = [x + (4 * size), y - (5 * size), size, size]
        self.OutCell[21] = [x + (4 * size), y - (4 * size), size, size]
        self.OutCell[22] = [x + (4 * size), y - (3 * size), size, size]
        self.OutCell[23] = [x + (4 * size), y - (2 * size), size, size]
        self.OutCell[24] = [x + (4 * size), y - (size), size, size]
        self.OutCell[25] = [x + (4 * size), y, size, size]
        self.OutCell[26] = [x + (4 * size), y + (size), size, size]
        self.OutCell[27] = [x + (3 * size), y - (5 * size), size, size]
        self.OutCell[28] = [x + (3 * size), y - (4 * size), size, size]
        self.OutCell[29] = [x + (3 * size), y - (3 * size), size, size]
        self.OutCell[30] = [x + (3 * size), y - (2 * size), size, size]
        self.OutCell[31] = [x + (3 * size), y - (size), size, size]
        self.OutCell[32] = [x + (3 * size), y, size, size]
        self.OutCell[33] = [x + (3 * size), y + (size), size, size]
        self.OutCell[34] = [x + (2 * size), y - (5 * size), size, size]
        self.OutCell[35] = [x + (2 * size), y - (4 * size), size, size]
        self.OutCell[36] = [x + (2 * size), y - (3 * size), size, size]
        self.OutCell[37] = [x + (2 * size), y - (2 * size), size, size]
        self.OutCell[38] = [x + (2 * size), y - (size), size, size]
        self.OutCell[39] = [x + (2 * size), y, size, size]
        self.OutCell[40] = [x + (2 * size), y + (size), size, size]
        self.OutCell[41] = [x + size, y - (5 * size), size, size]
        self.OutCell[42] = [x + size, y - (4 * size), size, size]
        self.OutCell[43] = [x + size, y - (3 * size), size, size]
        self.OutCell[44] = [x + size, y - (2 * size), size, size]
        self.OutCell[45] = [x, y - (5 * size), size, size]
        self.OutCell[46] = [x, y - (4 * size), size, size]
        self.OutCell[47] = [x, y - (3 * size), size, size]
        self.OutCell[48] = [x, y - (2 * size), size, size]
        self.OutCell[49] = [x - size, y - (5 * size), size, size]
        self.OutCell[50] = [x - size, y - (4 * size), size, size]
        self.OutCell[51] = [x - size, y - (3 * size), size, size]
        self.OutCell[52] = [x - size, y - (2 * size), size, size]
        self.OutCell[53] = [x - (2 * size), y - (5 * size), size, size]
        self.OutCell[54] = [x - (2 * size), y - (4 * size), size, size]
        self.OutCell[55] = [x - (2 * size), y - (3 * size), size, size]
        self.OutCell[56] = [x - (2 * size), y - (2 * size), size, size]
        self.OutCell[57] = [x - (2 * size), y - (size), size, size]
        self.OutCell[58] = [x - (2 * size), y, size, size]
        self.OutCell[59] = [x - (2 * size), y + (size), size, size]
        self.OutCell[60] = [x - (3 * size), y - (5 * size), size, size]
        self.OutCell[61] = [x - (3 * size), y - (4 * size), size, size]
        self.OutCell[62] = [x - (3 * size), y - (3 * size), size, size]
        self.OutCell[63] = [x - (3 * size), y - (2 * size), size, size]
        self.OutCell[64] = [x - (3 * size), y - (size), size, size]
        self.OutCell[65] = [x - (3 * size), y, size, size]
        self.OutCell[66] = [x - (3 * size), y + (size), size, size]
        self.OutCell[67] = [x - (4 * size), y - (5 * size), size, size]
        self.OutCell[68] = [x - (4 * size), y - (4 * size), size, size]
        self.OutCell[69] = [x - (4 * size), y - (3 * size), size, size]
        self.OutCell[70] = [x - (4 * size), y - (2 * size), size, size]
        self.OutCell[71] = [x - (4 * size), y - (size), size, size]
        self.OutCell[72] = [x - (4 * size), y, size, size]
        self.OutCell[73] = [x - (4 * size), y + (size), size, size]
        self.OutCell[74] = [x - (5 * size), y - (4 * size), size, size]
        self.OutCell[75] = [x - (5 * size), y - (3 * size), size, size]
        self.OutCell[76] = [x - (5 * size), y - (2 * size), size, size]
        self.OutCell[77] = [x - (5 * size), y - (size), size, size]
        self.OutCell[78] = [x - (5 * size), y, size, size]
        self.OutCell[79] = [x - (5 * size), y + (size), size, size]
        self.OutCell[80] = [x - (6 * size), y - (3 * size), size, size]
        self.OutCell[81] = [x - (6 * size), y - (2 * size), size, size]
        self.OutCell[82] = [x - (6 * size), y - (size), size, size]
        self.OutCell[83] = [x - (6 * size), y, size, size]
        self.OutCell[84] = [x - (6 * size), y + (size), size, size]
        self.OutCell[85] = [x - (7 * size), y - (2 * size), size, size]
        self.OutCell[86] = [x - (7 * size), y - (size), size, size]
        self.OutCell[87] = [x - (7 * size), y, size, size]
        self.OutCell[88] = [x - (7 * size), y + (size), size, size]
        self.OutCell[89] = [x - (8 * size), y - (size), size, size]
        self.OutCell[90] = [x - (8 * size), y, size, size]
        self.OutCell[91] = [x - (8 * size), y + (size), size, size]
        self.OutCell[92] = [x - (9 * size), y, size, size]

    # 셀 매핑, x,y 좌표 전달을 통해서 셀 번호 도출
    def check_in_cell_map(self, x, y):
        no = 9999  # 에러 셀 번호 9999, 정상 0~85
        for i in range(0, 86):
            if (x >= self.InCell[i][0] and self.InCell[i][0] + self.InCell[i][2] >= x and y >= self.InCell[i][1] and
                    self.InCell[i][1] + self.InCell[i][3] >= y):
                no = i
                break
        return no

    def check_out_cell_map(self, x, y):
        no = 9999  # 에러 셀 번호 9999, 정상 0~92
        for i in range(0, 93):
            if (x >= self.OutCell[i][0] and self.OutCell[i][0] + self.OutCell[i][2] >= x and y >= self.OutCell[i][1] and
                    self.OutCell[i][1] + self.OutCell[i][3] >= y):
                no = i
                break
        return no

    # CV2 상에서 Measurement Map 표시
    def show_in_cell_map(self, image):
        cv2.rectangle(image, (int(self.InCell[0][0]), int(self.InCell[0][1])),
                      (int(self.InCell[0][0] + self.InCell[0][2]), int(self.InCell[0][1] + self.InCell[0][3])),
                      (0, 255, 0), 2, 1)
        cv2.putText(image, 'IN', (
        int(self.InCell[0][0] + (self.InCell[0][2] / 2)), int(self.InCell[0][1] + (self.InCell[0][3])) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 180, 50), 1)
        for i in range(1, 86):
            cv2.rectangle(image, (int(self.InCell[i][0]), int(self.InCell[i][1])),
                          (int(self.InCell[i][0] + self.InCell[i][2]), int(self.InCell[i][1] + self.InCell[i][3])),
                          (0, 0, 255), 1, 1)
            cv2.putText(image, str(int(i)), (
            int(self.InCell[i][0] + (self.InCell[i][2] / 2)), int(self.InCell[i][1] + (self.InCell[i][3] / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 180, 50), 1)
        return image

    def show_out_cell_map(self, image):
        cv2.rectangle(image, (int(self.OutCell[0][0]), int(self.OutCell[0][1])),
                      (int(self.OutCell[0][0] + self.OutCell[0][2]), int(self.OutCell[0][1] + self.OutCell[0][3])),
                      (0, 255, 0), 2, 1)
        cv2.putText(image, 'OUT', (
        int(self.OutCell[0][0] + (self.OutCell[0][2] / 2)), int(self.OutCell[0][1] + (self.OutCell[0][3])) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 180, 50), 1)
        for i in range(1, 93):
            cv2.rectangle(image, (int(self.OutCell[i][0]), int(self.OutCell[i][1])),
                          (int(self.OutCell[i][0] + self.OutCell[i][2]), int(self.OutCell[i][1] + self.OutCell[i][3])),
                          (0, 0, 255), 1, 1)
            cv2.putText(image, str(int(i)), (
            int(self.OutCell[i][0] + (self.OutCell[i][2] / 2)), int(self.OutCell[i][1] + (self.OutCell[i][3] / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 180, 50), 1)
        return image

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold

inpWidth = 416     # Width of network's input image
inpHeight = 416    # Height of network's input image

prevTime = 0

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
#parser.add_argument('--video', help='Path to video file.')
#parser.add_argument('--video', default="C:/Users/HJ/Desktop/video_cap_test/video/1.mp4", help='Path to video file.')
parser.add_argument('--video', default="C:/Users/HJ/Desktop/video_cap_test/spv/t80.mp4", help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "custom_steel_cfg/steel_plate.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "custom_steel_cfg/steel_plate.cfg";
modelWeights = "weights/steel_plate_8n.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

bbox_coor_x = []
bbox_coor_y = []

bbox_rl = []

x1_predict_future = 0
y1_predict_future = 0
x2_predict_future = 0
y2_predict_future = 0

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
    label = '%.2f' % conf

    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (0, 0, 255), 1)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        # print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:

            confidence = scores[classId]
            # if detection[4]>confThreshold:
            # print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
            # print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

                boxes_sep = [left, top, width, height]

                # bbox_rl.append(boxes_sep)

                l = boxes_sep[0]
                t = boxes_sep[1]
                w = boxes_sep[2]
                h = boxes_sep[3]

                bbox_coor_x.append(l + w / 2)
                bbox_coor_y.append(t + h / 2)

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        #print('--------------')
        # print(i)
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = left + width
        bottom = top + height

        drawPred(classIds[i], confidences[i], left, top, right, bottom)

outputFile = "grid_mapping.avi"
outputFile2 = "yolo_out_py.jpg"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

# frame_width =  int(cap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
# frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # float

i=50
grid_bg_in = np.zeros((1024, 1280, 3), np.uint8)
grid_bg_out = np.zeros((1024, 1280, 3), np.uint8)

while cap.isOpened():

    hasFrame, frame = cap.read()

    Map = Measurement_Map()
    # [540, 800], [540, 950]
    Map.set_out_cell(50, 540, 800)
    Map.set_in_cell(50, 540, 950)

    #grid_bg_in = np.zeros((1024, 1280, 3), np.uint8)
    #grid_bg_out = np.zeros((1024, 1280, 3), np.uint8)

    grid_bg_in = Map.show_in_cell_map(grid_bg_in)
    grid_bg_out = Map.show_out_cell_map(grid_bg_out)


    '''curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    str = "FPS : %0.1f" % fps
    cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))'''

    '''if cv2.waitKey(1) == ord('e'):
        break;'''

    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(300)
        break
    if x1_predict_future > 0 or y1_predict_future > 0 or x2_predict_future > 0 or y2_predict_future > 0:
        #cv2.circle(frame, (int(x1_predict_future), int(y1_predict_future)), 10, (0, 0, 255), -1)
        #cv2.circle(frame, (int(x2_predict_future), int(y2_predict_future)), 10, (255, 0, 0), -1)
        cv2.circle(grid_bg_out, (int(x1_predict_future), int(y1_predict_future)), 10, (0, 0, 255), -1)
        cv2.circle(grid_bg_in, (int(x2_predict_future), int(y2_predict_future)), 10, (255, 0, 0), -1)

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    postprocess(frame, outs)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    grid_bg = np.hstack((grid_bg_out, grid_bg_in))
    frame = np.hstack((frame, grid_bg))

    #final_frame = cv.resize(frame, (600, 600))# Resize image
    final_frame= cv.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv.imshow('licence plate detection', final_frame)

    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))
        cv.imwrite(outputFile2, frame.astype(np.uint8));

    if cv2.waitKey(1) == ord('q'):
    #if len(bbox_coor_x) > 30:

        #i+=1

        bbox_coor_x1 = []
        bbox_coor_x2 = []
        bbox_coor_y1 = []
        bbox_coor_y2 = []

        time_axis_1 = []
        time_axis_2 = []

        bbox_coor_x_d = np.array(bbox_coor_x)
        bbox_coor_y_d = np.array(bbox_coor_y)

        for i in range(0, 19):
            if (bbox_coor_y_d[i] < bbox_coor_y_d[i + 1]):
                bbox_coor_x1.append(bbox_coor_x_d[i])
                bbox_coor_y1.append(bbox_coor_y_d[i])

            else:
                bbox_coor_x2.append(bbox_coor_x_d[i])
                bbox_coor_y2.append(bbox_coor_y_d[i])

        for j in range(10, len(bbox_coor_x_d) - 1):
            k1 = ((bbox_coor_x1[-1] - bbox_coor_x_d[1 + j]) ** 2 + (bbox_coor_y1[-1] - bbox_coor_y_d[1 + j]) ** 2) ** (
                        1 / 2)
            k2 = ((bbox_coor_x2[-1] - bbox_coor_x_d[1 + j]) ** 2 + (bbox_coor_y2[-1] - bbox_coor_y_d[1 + j]) ** 2) ** (
                        1 / 2)
            if (k1 < k2):
                bbox_coor_x1.append(bbox_coor_x[1 + j])
                bbox_coor_y1.append(bbox_coor_y[1 + j])
            else:
                bbox_coor_x2.append(bbox_coor_x[1 + j])
                bbox_coor_y2.append(bbox_coor_y[1 + j])

        for i in range(len(bbox_coor_x1)):
            time_axis_1.append(i)

        for i in range(len(bbox_coor_x2)):
            time_axis_2.append(i)

        time_axis_d1 = np.array(time_axis_1)
        time_axis_d2 = np.array(time_axis_2)


        plt.plot(bbox_coor_x1, bbox_coor_y1, 'o')
        plt.plot(bbox_coor_x1, bbox_coor_y1, 'r')

        plt.plot(bbox_coor_x2, bbox_coor_y2, 'o')
        plt.plot(bbox_coor_x2, bbox_coor_y2, 'g')

        plt.xlim([10, 500])
        plt.ylim([10, 500])

        #plt.show()

        time_axis_d1 = np.reshape(time_axis_d1, (-1, 1))
        time_axis_d2 = np.reshape(time_axis_d2, (-1, 1))

        bbox_coor_x1 = np.reshape(bbox_coor_x1, (-1, 1))
        bbox_coor_y1 = np.reshape(bbox_coor_y1, (-1, 1))

        bbox_coor_x2 = np.reshape(bbox_coor_x2, (-1, 1))
        bbox_coor_y2 = np.reshape(bbox_coor_y2, (-1, 1))


        '''        
        # x 좌표 선형회귀 예측
        line_fitter1 = LinearRegression()
        line_fitter1.fit(time_axis_d1, bbox_coor_x1)

        y1_predicted = line_fitter1.predict(time_axis_d1)

        line_fitter1.coef_
        line_fitter1.intercept_

        line_fitter2 = LinearRegression()
        line_fitter2.fit(time_axis_d2, bbox_coor_x2)

        y2_predicted = line_fitter2.predict(time_axis_d2)

        line_fitter2.coef_
        line_fitter2.intercept_

        line_fitter1.predict([[70]])
        line_fitter2.predict([[70]])

        # y 좌표 선형회귀 예측
        line_fitter1 = LinearRegression()
        line_fitter1.fit(time_axis_d1, bbox_coor_y1)

        y1_predicted = line_fitter1.predict(time_axis_d1)

        line_fitter1.coef_
        line_fitter1.intercept_

        line_fitter2 = LinearRegression()
        line_fitter2.fit(time_axis_d2, bbox_coor_y2)

        y2_predicted = line_fitter2.predict(time_axis_d2)

        line_fitter2.coef_
        line_fitter2.intercept_

        line_fitter1.predict([[70]])
        line_fitter2.predict([[70]])
        '''

        poly_feature1 = PolynomialFeatures(degree=2, include_bias=False)
        time_axis_poly_x1 = poly_feature1.fit_transform(time_axis_d1)

        line_fitter_nonlinear_x1 = LinearRegression()
        line_fitter_nonlinear_x1.fit(time_axis_poly_x1, bbox_coor_x1)

        print('intercept_x:', line_fitter_nonlinear_x1.intercept_, end=' ')
        print('coefficients_x:', line_fitter_nonlinear_x1.coef_, end=' ')

        time_axis_poly_x1 = poly_feature1.fit_transform(time_axis_d1)
        x_pred1 = line_fitter_nonlinear_x1.predict(time_axis_poly_x1)  # lin_reg 선형회귀분석을 토대로 predict 실행
        # print(time_axis_poly_x)

        plt.plot(time_axis_d1, bbox_coor_x1, 'o')
        plt.plot(time_axis_d1, x_pred1, 'r')  # predict한 결과를 그래프로 구현
        # plt.show()

        poly_feature1 = PolynomialFeatures(degree=2, include_bias=False)
        time_axis_poly_y1 = poly_feature1.fit_transform(time_axis_d1)

        line_fitter_nonlinear_y1 = LinearRegression()
        line_fitter_nonlinear_y1.fit(time_axis_poly_y1, bbox_coor_y1)

        print('intercept_y:', line_fitter_nonlinear_y1.intercept_, end=' ')
        print('coefficients_y:', line_fitter_nonlinear_y1.coef_, end=' ')

        time_axis_poly_y1 = poly_feature1.fit_transform(time_axis_d1)
        y_pred1 = line_fitter_nonlinear_y1.predict(time_axis_poly_y1)  # lin_reg 선형회귀분석을 토대로 predict 실행
        # print(time_axis_poly_y)

        plt.plot(time_axis_d1, bbox_coor_y1, 'o')
        plt.plot(time_axis_d1, y_pred1, 'r')  # predict한 결과를 그래프로 구현
        # plt.show()

        # print(y1_pred)

        poly_feature2 = PolynomialFeatures(degree=2, include_bias=False)
        time_axis_poly_x2 = poly_feature2.fit_transform(time_axis_d2)

        line_fitter_nonlinear_x2 = LinearRegression()
        line_fitter_nonlinear_x2.fit(time_axis_poly_x2, bbox_coor_x2)

        print('intercept_x:', line_fitter_nonlinear_x2.intercept_, end=' ')
        print('coefficients_x:', line_fitter_nonlinear_x2.coef_, end=' ')

        time_axis_poly_x2 = poly_feature2.fit_transform(time_axis_d2)
        x_pred2 = line_fitter_nonlinear_x2.predict(time_axis_poly_x2)  # lin_reg 선형회귀분석을 토대로 predict 실행
        # print(time_axis_poly_x)

        plt.plot(time_axis_d2, bbox_coor_x2, 'o')
        plt.plot(time_axis_d2, x_pred2, 'r')  # predict한 결과를 그래프로 구현
        # plt.show()

        poly_feature2 = PolynomialFeatures(degree=2, include_bias=False)
        time_axis_poly_y2 = poly_feature2.fit_transform(time_axis_d2)

        line_fitter_nonlinear_y2 = LinearRegression()
        line_fitter_nonlinear_y2.fit(time_axis_poly_y2, bbox_coor_y2)

        print('intercept_y:', line_fitter_nonlinear_y2.intercept_, end=' ')
        print('coefficients_y:', line_fitter_nonlinear_y2.coef_, end=' ')

        time_axis_poly_y2 = poly_feature2.fit_transform(time_axis_d2)
        y_pred2 = line_fitter_nonlinear_y2.predict(time_axis_poly_y2)  # lin_reg 선형회귀분석을 토대로 predict 실행
        # print(time_axis_poly_y)

        plt.plot(time_axis_d2, bbox_coor_y2, 'o')
        plt.plot(time_axis_d2, y_pred2, 'r')  # predict한 결과를 그래프로 구현
        # plt.show()

        for i in range(99, 100):
            #xy1_predict_want = time_axis_d1[-1][-1]+20
            xy1_predict_want = i

            #print(xy1_predict_want)
            x1_predict_future = line_fitter_nonlinear_x1.intercept_[0] + line_fitter_nonlinear_x1.coef_[0][0] * (
                xy1_predict_want) + \
                               line_fitter_nonlinear_x1.coef_[0][1] * (xy1_predict_want ** 2)# + \
                               #line_fitter_nonlinear_x1.coef_[0][
                               #    2] * (xy1_predict_want ** 3)+line_fitter_nonlinear_x1.coef_[0][
                               #    3] * (xy1_predict_want ** 4)
            y1_predict_future = line_fitter_nonlinear_y1.intercept_[0] + line_fitter_nonlinear_y1.coef_[0][0] * (
                xy1_predict_want) + \
                               line_fitter_nonlinear_y1.coef_[0][1] * (xy1_predict_want ** 2)#+ \
                               #line_fitter_nonlinear_y1.coef_[0][
                               #    2] * (xy1_predict_want ** 3)+line_fitter_nonlinear_y1.coef_[0][
                               #    3] * (xy1_predict_want ** 4)

            #xy2_predict_want = time_axis_d2[-1][-1]+20
            xy2_predict_want = i

            #print(xy2_predict_want)
            x2_predict_future = line_fitter_nonlinear_x2.intercept_[0] + line_fitter_nonlinear_x2.coef_[0][0] * (
                xy2_predict_want) + \
                                line_fitter_nonlinear_x2.coef_[0][1] * (xy2_predict_want ** 2)# + \
                                #line_fitter_nonlinear_x2.coef_[0][
                                #    2] * (xy2_predict_want ** 3)+line_fitter_nonlinear_x2.coef_[0][
                                #    3] * (xy2_predict_want ** 4)
            y2_predict_future = line_fitter_nonlinear_y2.intercept_[0] + line_fitter_nonlinear_y2.coef_[0][0] * (
                xy2_predict_want) + \
                                line_fitter_nonlinear_y2.coef_[0][1] * (xy2_predict_want ** 2)# + \
                                #line_fitter_nonlinear_y2.coef_[0][
                                #    2] * (xy2_predict_want ** 3)+line_fitter_nonlinear_y2.coef_[0][
                                #    3] * (xy2_predict_want ** 4)

            print(x1_predict_future)
            print(y1_predict_future)

            print(x2_predict_future)
            print(y2_predict_future)

            cv2.circle(frame, (int(x1_predict_future), int(y1_predict_future)), 10, (0, 0, 255), -1)
            cv2.circle(frame, (int(x2_predict_future), int(y2_predict_future)), 10, (255, 0, 0), -1)

            Map = Measurement_Map()
            # [561, 821], [541, 963]
            #Map.set_out_cell(50, 550, 820)
            #Map.set_in_cell(50, 550, 960)
            Map.set_out_cell(50, 540, 800)
            Map.set_in_cell(50, 540, 950)

            #grid_bg_in = np.zeros((1024, 1280, 3), np.uint8)
            #grid_bg_out = np.zeros((1024, 1280, 3), np.uint8)

            '''
            x1_predict_future += 0
            x2_predict_future += 0
            y1_predict_future += 220
            y2_predict_future += 220
            '''

            cv2.circle(grid_bg_out, (int(x1_predict_future), int(y1_predict_future)), 10, (0, 0, 255), -1)
            cv2.circle(grid_bg_in, (int(x2_predict_future), int(y2_predict_future)), 10, (255, 0, 0), -1)

            grid_bg_in = Map.show_in_cell_map(grid_bg_in)
            grid_bg_out = Map.show_out_cell_map(grid_bg_out)

            #grid_bg = np.hstack((grid_bg_out, grid_bg_in))
            #grid_bg = cv2.resize(grid_bg, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            #cv2.imshow('grid_bg', grid_bg)

            print(Map.check_out_cell_map(x1_predict_future, y1_predict_future))
            print(Map.check_in_cell_map(x2_predict_future, y2_predict_future))

            #vid_writer.write(frame.astype(np.uint8))

            # cv.imwrite(args.video[:-4] + outputFile2, frame.astype(np.uint8))