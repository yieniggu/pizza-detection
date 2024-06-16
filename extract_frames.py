from ultralytics import YOLO
import cv2
import sys
import supervision as sv

# Load an official or custom model
model = YOLO('best.pt')  # Load an official Detect model

cap = cv2.VideoCapture("/home/basti/Desktop/datasets/pizza_yolov8/pizza_test.mp4")
if (cap.isOpened() == False):
	print('!!! Unable to open URL')
	sys.exit(-1)

# retrieve FPS and calculate how long to wait between each frame to be display
fps = cap.get(cv2.CAP_PROP_FPS)
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
wait_ms = int(1000/fps)
print('FPS:', fps)
print('wait_ms:', wait_ms)

print("input_width: ", input_width)
print("input_height: ", input_height)

frame_counter = 0
vid_counter = 1
while(cap.isOpened() & frame_counter < 2000):
	ret, frame = cap.read()

	frame = cv2.resize(frame, (1366, 768))
	
	if vid_counter % 15 == 0:
		# save frame
		cv2.imwrite("frames/frame_%d.jpg" % frame_counter, frame)     # save frame as JPEG file      
		frame_counter += 1

	vid_counter += 1

cap.release()
cv2.destroyAllWindows()
