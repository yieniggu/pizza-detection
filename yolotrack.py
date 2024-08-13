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
	# Load an official or custom model
	model = YOLO('mut_v2.pt')  # Load an official Detect model
	byte_tracker = sv.ByteTrack()

	filename = video_path.split(".")[0]

	video_result = cv2.VideoWriter(filename + '_results.avi',  
													cv2.VideoWriter_fourcc(*'MJPG'), 
													22, (1024, 768)) 

	cap = cv2.VideoCapture(video_path)
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
		# time when we finish processing for this frame 
		new_frame_time = time.time() 	
		ret, frame = cap.read()

		if not ret:
			break

		frame = cv2.resize(frame, (1024, 768))

		result = model.predict(frame, conf=0.1)[0]

		detections = sv.Detections.from_ultralytics(result)
		detections = byte_tracker.update_with_detections(detections)

		labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
		
		annotated_frame = bounding_box_annotator.annotate(
			scene=frame.copy(),
			detections=detections
		)
		annotated_frame = bounding_box_annotator.annotate(
					scene=frame.copy(), detections=detections)
		# annotated_frame = label_annotator.annotate(
			#     scene=annotated_frame, detections=detections, labels=labels)

		annotated_frame = sv.draw_line(annotated_frame, start, end, sv.Color(0, 255, 50))

		crossed_in, crossed_out = line_zone.trigger(detections)
		if np.any(crossed_in):
			delivered.append(["returned", frame_count])
			total_in += 1

		if np.any(crossed_out):
			# register delivered event
			delivered.append(["delivered", frame_count])
			total_out += 1

		annotated_frame = cv2.putText(annotated_frame, "TOTAL DELIVERED:{}".format(total_out),
																									(5, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 255, 0), 2, cv2.LINE_AA)
		annotated_frame = cv2.putText(annotated_frame, "TOTAL RETURNED:{}".format(total_in),
																									(5, 70), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 255), 2, cv2.LINE_AA)
		annotated_frame = cv2.putText(annotated_frame, "TOTAL:{}".format(total_out-total_in),
																									(5, 90), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255, 0, 0), 2, cv2.LINE_AA)
							
		fps.update()
		fps.stop()
		print("FPS: {:.2f}".format(fps.fps()))

		fps_text = "{0:.2f}".format(fps.fps())
		cv2.putText(annotated_frame, "FPS: " + fps_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		video_result.write(annotated_frame) 

		frame_count += 1

		cv2.imshow('frame', annotated_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	df = pd.DataFrame(delivered, columns=["delivered_event", "delivered_at_frame"])
	df.to_csv(filename + "_delivered_results.csv", index=False)

	df2 = pd.DataFrame([frame_count], columns=["total_frames"])
	df2.to_csv(filename + "_total_frames.csv", index=False)

	cap.release()
	video_result.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument("-v", "--video", type=str, 
			required=True)
	
	args = parser.parse_args()

	process_video(args.video)