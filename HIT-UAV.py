import argparse
import os
import sys
import cv2
import time 
import glob
from ultralytics import YOLO
import numpy as np 
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack
from pymavlink import mavutil

parser = argparse.ArgumentParser()
parser.add_argument('--model', help = 'Path to your YOLO model', required=True )
parser.add_argument('--source', help= 'Source of camera', required=True)
parser.add_argument('--thresh', help= 'Minimum confidence threshold for displaying box', default=0.5)
parser.add_argument('--resolution', help='Default = 640x480 or match source resolution', default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.', action='store_true')

args = parser.parse_args()
model_path = args.model
src = args.source
min_thresh = args.thresh
user_res= args.resolution
record = args.record

#check if valid file path of model
if(not os.path.exists(model_path)):
    print("ERROR: File path is not valid")
    sys.exit(0)

model = YOLO(model_path)
labels = model.names

img_ext_list = ['.jpg','.jpeg','.png','.bmp',]
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

#Assign source type =[image, folder, video, usb or picamera]

if os.path.isdir(src): #check if directory
    source_type = 'folder'

elif os.path.isfile(src): #check if it is just a file
    _, ext = os.path.splitext(src)

    if(ext.lower() in img_ext_list):
        source_type = 'image'
    elif(ext in vid_ext_list):
        source_type='video'
    else:
        print("File extension-{ext} not supported")
        sys.exit(0)
    
elif 'usb' in src:
    source_type='usb'
    usb_idx= int(src[3:])
elif 'picamera' in src:
    source_type= 'picamera'
else:
    print(f'Input {src} is invalid. Please try again')
    sys.exit(0)


#Parse display resolution (specified by user)
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int (user_res.split('x')[1]) #if user has specified resolution, then parse that into resH, resW

#Set up recording (if user has agreed)
if record:
    if source_type not in ['video','usb']:
        print("Recording only works for video or camera sources")
        sys.exit(0)
    
    if not user_res: 
        print('Please specify resolution to record video at')
        sys.exit(0)

    record_name = 'demo1.avi'
    record_fps = 30 
    recorder= cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps , (resW, resH))


#Parse the input (diff actions for diff source types)
if source_type== 'image':
    imgs_list = [src]


elif source_type=='video' or source_type=='usb':
    cap_arg = src if source_type=='video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    ret = cap.set(3,resW)
    ret = cap.set(4,resH)

elif source_type=='picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()


elif source_type == 'folder':
    imgs_list = []
    vids_list=[]
    filelist = glob.glob(os.path.join(src, '*'))
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext.lower() in img_ext_list:
            imgs_list.append(file)
        elif file_ext.lower() in vid_ext_list:
            vids_list.append(file)

#Set colours for bounding boxes 
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate =0
frame_rate_buffer=[]
fps_avg_len=200
img_count=0
vid_count=0
tracker = ByteTrack(
    track_activation_threshold= float(min_thresh),
    lost_track_buffer=30,
    frame_rate=20 
)

master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)  #check port later
master.wait_heartbeat()
print("Connected to flight controller!")

master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

master.set_mode_apm('GUIDED')
print("Armed and in GUIDED mode")


k = 0.01  # Proportional gain (tune later)
selected_id= None
tracked_detections = []

def mouse_callback(event, x, y, flags, param):
    global selected_id, tracked_detections
    if event == cv2.EVENT_LBUTTONDOWN:
        for xyxy, _, _, _, tracker_id, _ in tracked_detections:
            xmin, ymin, xmax, ymax = map(int, xyxy)
            if xmin <= x <= xmax and ymin <= y <= ymax:
                selected_id = tracker_id
                print(f"Selected object ID: {selected_id}")
                break

cv2.namedWindow('YOLO Detection results')
cv2.setMouseCallback('YOLO Detection results', mouse_callback)

while True: #runs for each frame 
    t_start = time.perf_counter()
    
    #Load frames from image source 

    if source_type == 'folder':
        if img_count < len(imgs_list):
            frame = cv2.imread(imgs_list[img_count])
            img_count= img_count+1
            tracker = ByteTrack(
            track_activation_threshold= float(min_thresh),
            lost_track_buffer=30,
            frame_rate=20  # Match your camera's FPS
            )
            
            
        else:
            print("All images have been processed")
            break

        # Handle videos (separate loop for videos, once all images are processed)
        if vid_count < len(vids_list):
            cap = cv2.VideoCapture(vids_list[vid_count])
            ret, frame = cap.read()
            if not ret:
                print(f"End of this video. Moving to next.")
                vid_count += 1
                cap.release()

    elif source_type=='image':
        if(img_count>=len(imgs_list)):
            print("All images have been processed")
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        


    elif source_type=='video':
        ret, frame = cap.read()
        if not ret: 
            print('Reached end of the video file. Exiting program.')
            break

    elif source_type=='usb':
        ret,frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type=='picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    #Resize frames to desired displays
    if resize==True:
        frame = cv2.resize(frame, (resW,resH))

    #inference on each frame
    results = model(frame, verbose=False) #results composed of bounding boxes, confidence scores, class ids
    #detections = results[0] #access the bounding box cordinates 

    detections = sv.Detections.from_ultralytics(results[0])
    tracked_detections = tracker.update_with_detections(detections)

    for xyxy, _, confidence, class_id, tracker_id , _ in tracked_detections:
        xmin, ymin, xmax, ymax = map(int, xyxy)
        color = bbox_colors[tracker_id % 10]  # Use tracker_id for color
        
        label = f"ID:{tracker_id} {labels[class_id]}: {confidence:.2f}"
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) #labelSize = (width,height) in pixels
        label_width, label_height = labelSize

        if xmin+ label_width > frame.shape[1]:
            label_xmin = frame.shape[1]-label_width-1   
        else: 
            label_xmin = xmin
        
        label_xmax = label_xmin + label_width
        
        if ymin - label_height - 10 <0:
            label_ymin = ymax + 10
        else:
            label_ymin = ymin- label_height-10

        cv2.rectangle(frame, (label_xmin, label_ymin), (label_xmax, label_ymin + label_height + baseLine), color, cv2.FILLED) #draw label box
        cv2.putText(frame, label, (label_xmin, label_ymin + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1) #draw bounding box for object


        if tracker_id==selected_id:
            obj_cx= (xmin+xmax)//2
            obj_cy = (ymin+ymax)//2

            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape // 2
            
            err_x = obj_cx - frame_center_x
            err_y = obj_cy - frame_center_y
            
            
            vx = -err_y * k  # Forward/backward
            vy = -err_x * k  # Left/right
            vz = 0  

            master.mav.set_position_target_local_ned_send(
                0, master.target_system, master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED, 0b0000111111000111,
                0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
            
            print(f"Following ID:{tracker_id} - Error: ({err_x}, {err_y}) - Velocity: ({vx:.3f}, {vy:.3f})")

    #display detection results 
    if source_type=='video' or source_type=='usb' or source_type=='picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('YOLO Detection results', frame)
    if record: recorder.write(frame)
     
    if source_type=='image' or source_type=='folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)

    #calculate fps for this frame 

    t_stop = time.perf_counter() # the single frame's time is stopped
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)
    
    avg_frame_rate = np.mean(frame_rate_buffer)
    


print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()








