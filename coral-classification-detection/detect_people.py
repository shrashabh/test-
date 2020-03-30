# USAGE
# python detect_video.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from advertodev.centroidtracker import CentroidTracker
from advertodev.trackableobject import TrackableObject
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import PIL
from PIL import Image, ImageDraw, ImageFont
import argparse
import time
import cv2
import datetime
import numpy as np
import dlib
import time, threading
import os
import math
import csv, json

class position:
    position = None

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", required=True,
    help="path to labels file")
ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.01,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
    help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the labels dictionary
print("[INFO] parsing class labels...")
labels = {}

# loop over the class labels file
for row in open(args["labels"]):
    # unpack the row and update the labels dictionary
    (classID, label) = row.strip().split(maxsplit=1)
    labels[int(classID)] = label.strip()

# load the Google Coral object detection model
print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=1).start()

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=100, maxDistance=180)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

EntranceCounter = 0
ExitCounter = 0
OffsetRefLines = 20

#The magic begins
BASE_DIR = "cd /home/mendel/coral-classification-detection/data_saved"
#DATA_DIR = "/home/pi/Desktop/main/program/data_sent/"
#DATA_DIR1 = "/home/pi/Desktop/main/program/data_sent1/"

global ptimestamp
totalUp = 0
updated_up = 0
updated_down = 0
c_updated_up = 0
c_updated_down = 0
xtimestamp = datetime.datetime.now()
ptimestamp = xtimestamp.strftime('%Y-%m-%dT%H:%M:%S')
ztimestamp = ptimestamp
objectID = 0
frame_number = 0
shortest_d = 0
a = 0
b = 0
c = 0
#send_loc=os.listdir(BASE_DIR)
#length=len(send_loc)

def lineFromPoints(P,Q):
    global a
    global b
    global c
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a*(P[0]) + b*(P[1])
    c = -c

    if(b<0):
        print("The line passing through points P and Q is:",
              a ,"x + ",b ,"y = ",c ,"\n")
    else:
        print("The line passing through points P and Q is: ",
              a ,"x + " ,b ,"y = ",c ,"\n")



# Finding perpendicular distance
def shortest_distance(x1, y1, a, b, c):
    # global a
    # global b
    # global c
    global shortest_d

    shortest_d = (a * x1 + b * y1 + c) / (math.sqrt(a * a + b * b))
    shortest_d = int(shortest_d)
    print("Perpendicular distance is",shortest_d, type(d))
    print("vars**", x1, y1, a, b, c)

time.sleep(2.0)

# loop over the frames from the video stream
s = time.time()


while True:
    #try:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        if frame is None:
            continue
        height = np.size(frame,0)
        width = np.size(frame,1)

        frame_number = frame_number + 1
        if args["input"] is not None and frame is None:
            break
        frame = imutils.resize(frame, width=500)

        orig = frame.copy()
        d = time.strftime('%a %d-%b')
        t = time.strftime('%H:%M:%S')
        I1 = cv2.putText(orig,t,(573, 458),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        I1 = cv2.putText(orig,d,(576, 469),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.rectangle(orig, (5,5), (634,474), (250, 250, 250), 1)
        text1 = "PROCESSING GRID..."
        cv2.putText(orig, text1, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        text2 = "JBP-MP-IND"
        cv2.putText(orig, text2, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (181, 209, 0), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        if W is None or H is None:
            (H, W) = rgb.shape[:2]
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (W, H), True)
        status = "WAITING"
        rects = []

#        if totalFrames % args["skip_frames"] == 0:
        if totalFrames - totalFrames == 0:
            status = "Detecting"
            trackers = []
            start = time.time()
            results = model.DetectWithImage(frame, threshold=args["confidence"],
                keep_aspect_ratio=True, relative_coord=False)
            end = time.time()
            for r in results:
                box = r.bounding_box.flatten().astype("int")
                (startX, startY, endX, endY) = box
                label = labels[r.label_id]
                if label != "person":
                    continue
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
#        else:
#            for tracker in trackers:
                status = "ACTIVE"
#                wizard = tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                overlay = orig.copy()
                opacity = 0.6
                rects.append((startX, startY, endX, endY))
                cv2.rectangle(orig, (startX, startY), (endX, endY), (250, 200, 200), cv2.FILLED)
                cv2.addWeighted(overlay, opacity, orig, 1-opacity, 0, orig)
                cv2.rectangle(orig, (startX, startY), (endX, endY), (250, 250, 250), 1)
                cv2.rectangle(orig, (startX, startY-5), (startX+36, startY-19), (250, 250, 250), cv2.FILLED)
                cv2.rectangle(orig, (startX+36+5, startY-5), (startX+180, startY-21), (250, 250, 250), 1)
        else:
            print("Something Unusual")
        cv2.line(orig, (W // 2, H), (W, 0), (181, 209, 0), 1)
        print("-----////...", W //2, H, W, 0)
        P = [W // 2, H]
        Q = [W, 0]
        lineFromPoints(P,Q)
        print("doubt------", P, Q)
        output = ct.update(rects)
        for key, value in output.items():
            if key == "startX":
                startX = value
            if key == "startY":
                startY = value
        objects = output["objects"]
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y1 = [c[1] for c in to.centroids]
                x1 = [c[0] for c in to.centroids]
                for i in x1:
                    x1 = int(i)
                for i in y1:
                    y1 = int(i)
                print("x1", x1, type(x1))
                print("y1", y1, type(x1))
                print("a", a, type(a))
                print("b", b, type(b))
                print("c", c, type(c))
                print("---------------y", x1, y1)
                direction = centroid[1] - np.mean(y1)
                hydra = centroid[1] - (H//2)
                to.centroids.append(centroid)
                shortest_distance(x1, y1, a, b, c)
                #print("shortest_d)))))", shortest_d, type(shortest_d))
                print("People Detected----", objectID)
                if not to.counted:
                    if to.position == None:
                        print("shortest_d------", shortest_d, type(shortest_d))
                        to.position = 'down' if (shortest_d > 0) else 'up'
                        curr_pos = to.position
                    if to.position == 'down':
                        print("shortest_d down------", shortest_d, type(shortest_d))
                        if shortest_d < 0:
                            totalUp += 1
                            to.counted = True
                    else:
                        print("shortest_d up------", shortest_d, type(shortest_d))
                        if shortest_d > 0:
                            totalDown += 1
                            to.counted = True
                    position = None
            trackableObjects[objectID] = to
            print("******((()))", centroid)
            CoordXCentroid = centroid[0]
            CoordYCentroid = centroid[1]
            tx = centroid[2]
            ty = centroid[3]
            eX = int(centroid[2]*0.083)
            eY = int(centroid[3]*0.022)
            getX = int(tx+46)
            getY = int(ty-9)
            get1X = int(getX/20)
            getX2 = int(tx+46-get1X)
            text = "FOUND:"
            cv2.putText(orig, text, (getX, getY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            text = " PERSON {}".format(objectID)
            cv2.putText(orig, text, (getX+42, getY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (181, 209, 0), 1)
            cv2.circle(orig, (centroid[0], centroid[1]), 1, (255, 255, 255), -3)
        info = [
            ("RECORD: NEURAL NETWORK", "" ),
            ("MOVEMENT UP:    ",  totalUp),
            ("MOVEMENT DOWN:   ", totalDown),
            ("TRACKING STATUS:", status),
        ]
        for (i, (k, v)) in enumerate(info):
            if v == 'WAITING':
                if i == 3:
                    x = 130
                else:
                    x = 167
                text = "{} {}".format(k, "")
                cv2.putText(orig, text, (10, H - ((i * 15) + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                text = "{} {}".format("", v)
                cv2.putText(orig, text, (x, H - ((i * 15) + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (2, 169, 249), 1)
            else:
                if i == 3:
                    x = 135
                else:
                    x = 167
                text = "{} {}".format(k, "")
                cv2.putText(orig, text, (10, H - ((i * 15) + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                text = "{} {}".format("", v)
                cv2.putText(orig, text, (x, H - ((i * 15) + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (181, 209, 0), 1)
        if writer is not None:
            writer.write(orig)
        cv2.imshow("Frame", orig)
#        print(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            proofofplay()
            break
        totalFrames += 1
        fps.update()

#    except Exception as e:
#        print("Exception----:", e)
#        print("*****", frame)
#        print("Exp succeed")
#        proofofplay()
#        break
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] approx. People Detected: ", objectID)
if writer is not None:
    writer.release()
if not args.get("input", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()
