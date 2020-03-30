# USAGE
# python detect_video.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from edgetpu.classification.engine import ClassificationEngine
from imutils.video import VideoStream
from PIL import Image
import argparse
import imutils
import time
import cv2
import os
from threading import Thread
import numpy as np
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to TensorFlow Lite object detection model")
ap.add_argument("-m1", "--model1", required=True,
	help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", help="path to labels file")
ap.add_argument("-l1", "--labels1", help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the labels dictionary
print("[INFO] parsing class labels...")
labels = {}

# loop over the class labels file
for row in open(args["labels"]):
	# unpack the row and update the labels dictionary
	(classID, label) = row.strip().split(" ", maxsplit=1)
	label = label.strip().split(",", maxsplit=1)[0]
	labels[int(classID)] = label
# loop over the class labels file
#for row in open(args["labels"]):
	# unpack the row and update the labels dictionary
#	(classID, label) = row.strip().split(maxsplit=1)
#	labels[int(classID)] = label.strip()

# load the Google Coral object detection model
print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])
model1 = ClassificationEngine(args["model1"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
#vs = VideoStream(src=1).start()
vs = VideoStream(src="rtsp://192.168.31.38:5554/onvif1").start()
#vs = VideoStream(src="rtsp://192.168.31.38:5554/onvif1").start()
#vs = cv2.VideoCapture('rtsp://192.168.43.1:5554/video')
#vs = VideoStream(usePiCamera=False).start()
started = False

def Timer():
	print("Thread started")
	global timer
	timer = 5
	while timer != 0:
		time.sleep(1.0)
		timer = timer - 1
		print("timer----", timer)

def crop_face(imgarray, section, margin=0, size=224):
	"""
	:param imgarray: full image
	:param section: face detected area (x, y, w, h)
	:param margin: add some margin to the face detected area to include a full head
	:param size: the result image resolution with be (size x size)
	:return: resized image in numpy array with shape (size x size x 3)
	"""
	img_h, img_w, _ = imgarray.shape
	#if section is None:
	#	section = [0, 0, img_w, img_h]
	(x, y, w, h) = section
	margin = int(min(w,h) * margin / 100)
	x_a = x
	y_a = y
	x_b = w
	y_b = h
	# if x_a < 0:
	# 	x_b = min(x_b - x_a, img_w-1)
	# 	x_a = 0
	# if y_a < 0:
	# 	y_b = min(y_b - y_a, img_h-1)
	# 	y_a = 0
	# if x_b > img_w:
	# 	x_a = max(x_a - (x_b - img_w), 0)
	# 	x_b = img_w
	# if y_b > img_h:
	# 	y_a = max(y_a - (y_b - img_h), 0)
	# 	y_b = img_h
	cropped = imgarray[y_a: y_b, x_a: x_b]
	resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
	resized_img = np.array(resized_img)
	return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

Frame_number = 0
time.sleep(2.0)
s = time.time()
FPS = 0

face_size = 224
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 500 pixels
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame
	if frame is None:
		continue
	height = np.size(frame,0)
	width = np.size(frame,1)
	frame = imutils.resize(frame, width=900)
	orig = frame.copy()
	e = time.time()
	Frame_number = Frame_number + 1
#    print("e", e)
#    print("s", s)
	second = e - s
#    print("second", second)
	if second >= 1:
		FPS = Frame_number
		Frame_number = 0
		s = time.time()

	# prepare the frame for object detection by converting (1) it
	# from BGR to RGB channel ordering and then (2) from a NumPy
	# array to PIL image format
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#print("framecvt: ", frame)
	framecvt = frame
	frame = Image.fromarray(frame)
	#print("framesh: ", frame)

	# make predictions on the input frame
	start = time.time()
	results = model.detect_with_image(frame, threshold=args["confidence"],
	keep_aspect_ratio=True, relative_coord=False)
	end = time.time()

	if len(results) >= 2:
		if not started:
			print("About to start")
			Thread(target=Timer).start()
			print("Parallel exee")
			started = True
		text = " Timer {}".format(timer)
		cv2.putText(orig, text, (42, 72),
			cv2.FONT_HERSHEY_SIMPLEX, 0.4, (181, 209, 0), 1)
		if timer == 0:
			cv2.imwrite("orig.jpg", orig)
			timer = 5
			started = False

#    print("----)", len(results))

	# loop over the results
	face_imgs = np.empty((len(results), face_size, face_size, 3))
	for i, r in enumerate(results):
	# extract the bounding box and box and predicted class label
		#print("r", r)
		box = r.bounding_box.flatten().astype("int")
		print("box", box)
		(startX, startY, endX, endY) = box
		#print("frame", frame)
		face_img, cropped = crop_face(framecvt, box, margin=0, size=face_size)
		#print(face_img)
		face_imgs[i,:,:,:] = face_img
		frame1 = Image.fromarray(face_img)
	#		label = labels[r.label_id]


		if len(face_imgs) > 0:
			start = time.time()
			results1 = model1.ClassifyWithImage(frame1)
			end = time.time()
			cv2.imshow("Cropped", face_img)
			if len(results1) > 0:
				(classID, score) = results1[0]
				print("-->", (labels[classID]))
				if labels[classID] == "with_mask":
					green = True
				elif labels[classID] == "without_mask":
					green = False
				print(green)
				text = "{}: {:.2f}% ({:.4f} sec)".format(labels[classID],
					score * 100, end - start)
				cv2.putText(orig, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0, 0, 255), 2)
		# draw the bounding box and label on the image
		if green == True:
			cv2.rectangle(orig, (startX, startY), (endX, endY),
			(0, 255, 0), 4)
			text = "{}: {:.2f}% ({:.4f} sec)".format(labels[classID],
				score * 100, end - start)
			cv2.putText(orig, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				1, (0, 255, 0), 4)
		else:
			cv2.rectangle(orig, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
			text = "{}: {:.2f}% ({:.4f} sec)".format(labels[classID],
				score * 100, end - start)
			cv2.putText(orig, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				1, (0, 0, 255), 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		text = "{}: {:.2f}%".format("face", r.score * 100)
		cv2.putText(orig, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	# show the output frame and wait for a key press
	window_name = 'projector'
	cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
	#cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
	cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
						  cv2.WINDOW_FULLSCREEN)
	cv2.imshow(window_name, orig)
	#cv2.imshow("Frame", orig)
#    print("FPS: ", FPS)
	key = cv2.waitKey(5) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
