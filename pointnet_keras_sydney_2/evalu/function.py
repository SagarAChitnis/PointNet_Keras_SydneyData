import numpy as np

from operator import itemgetter
from functools import reduce

from tensorflow.keras import backend as K

# [IoU](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
def IoU(box0, box1):
    d_a = box0[0]
    box_a = box0[1]
    r0 = d_a / 2
    s0 = np.squeeze(box_a - r0)
    e0 = np.squeeze(box_a + r0)

    d_b = box1[0]
    box_b = box1[1]
    r1 = d_b / 2
    s1 = np.squeeze(box_b - r1)
    e1 = np.squeeze(box_b + r1)

    overlap = [max(0,min(e0[i], e1[i]) - max(s0[i], s1[i])) for i in range(3)]
    intersection = reduce(lambda x,y:x*y, overlap)
    union = d_a[0]*d_a[1]*d_a[2] + d_b[0]*d_b[1]*d_b[2] - intersection

    return intersection / union

def mAP(IoU:list, theadsold:float):
    TP = 0
    FP = 0
    PR_curve = []

    for idx, iou in enumerate(IoU):
        if iou >= theadsold:
            TP = TP + 1
        else:
            FP = FP + 1
        precision = TP/(TP + FP)
        recall = (idx + 1)/len(IoU)
        PR_curve.append([precision, recall])

    interpolated_points = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    max_precision = []

    for point in interpolated_points:
        interpolated_curve = []
        for PR in PR_curve:
            if PR[1] >= point:
                interpolated_curve.append(PR)
        if len(interpolated_curve) == 0:
            max_precision.append(0)
        else:
            max_precision.append(max(interpolated_curve, key=itemgetter(0))[0])

    AP = sum(max_precision)/len(max_precision)

    return AP

def f1_score(classification:list):
    TP = sum(classification)
    FP = len(classification) - sum(classification)
    precision = TP/(TP + FP)
    recall = TP/(TP + 0)

    f1 = 2 * precision * recall/(precision + recall)

    return f1

def smoothL1(y_true, y_pred):
    HUBER_DELTA = 0.5

    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))

    return  K.sum(x)