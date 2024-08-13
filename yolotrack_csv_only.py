from ultralytics import YOLO
import cv2
import sys
import supervision as sv
import numpy as np
from imutils.video import FPS
import datetime
import time  # The time module is easier for our purpose.
import pandas as pd
import argparse


def process_video(video_path):
	initial_time = datetime.datetime(2024, 5, 14, 12, 0, 0)

	# Load an official or custom model
	model = YOLO('mut_v2.pt')  # Load an official Detect model
	byte_tracker = sv.ByteTrack()

	cap = cv2.VideoCapture(video_path)
	if (cap.isOpened() == False):
			print('!!! Unable to open video file')
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

	polygon = np.array([
							[397, 335], [511, 335], [511, 420], [386, 420]]).astype(int) # topleft, top-right, bottom-right, bottom-left

	bounding_box_annotator = sv.BoundingBoxAnnotator()
	label_annotator = sv.LabelAnnotator(text_scale=0.3)
	start, end = sv.Point(x=400, y=310), sv.Point(x=680, y=310)
	line_zone = sv.LineZone(start=start, end=end)

	delivered = []

	total_in = total_out = 0
	frame_count = 1
	fps = FPS().start()
	while(cap.isOpened()):
		ret, frame = cap.read()

		if not ret:
			break

		frame = cv2.resize(frame, (1024, 768))

		result = model.predict(frame, conf=0.1)[0]

		detections = sv.Detections.from_ultralytics(result)
		detections = byte_tracker.update_with_detections(detections)

		labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

		crossed_in, crossed_out = line_zone.trigger(detections)
		if np.any(crossed_in):
			delivered.append(["returned", frame_count])
			total_in += 1

		if np.any(crossed_out):
			# register delivered event
			delivered.append(["delivered", frame_count])
			total_out += 1

		frame_count += 1

	df = pd.DataFrame(delivered, columns=["delivered_event", "delivered_at_frame"])

	filename = video_path.split(".")[0]
	df.to_csv(filename + "_delivered_results.csv", index=False)

	df2 = pd.DataFrame([frame_count], columns=["total_frames"])
	df2.to_csv(filename + "_total_frames.csv", index=False)

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument("-v", "--video", type=str, 
			required=True)
	
	args = parser.parse_args()

	process_video(args.video)