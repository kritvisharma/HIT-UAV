import argparse
import os
import sys
import cv2
import time 
import glob
from ultralytics import YOLO
import numpy as np 
from supervision.tracker.byte_tracker import ByteTrack



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
    picam_idx = int(src[8:])
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
    track_activation_threshold=0.5,
    lost_track_buffer=30,
    frame_rate=20  # Match your camera's FPS
)


while True: #runs for each frame 
    t_start = time.perf_counter()
    
    #Load frames from image source 

    if source_type == 'folder':
        if img_count < len(imgs_list):
            frame = cv2.imread(imgs_list[img_count])
            img_count += 1

        # Handle videos (separate loop for videos, once all images are processed)
        elif vid_count < len(vids_list):
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
        img_count= img_count+1


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
    detections = results[0] #access the bounding box cordinates 

    detections_list=[]
    for i in range(len(detections)): #loop for every object detected in the single frame
        #results use ultralytics (YOLO) and are in tensor form, need to be converted back to array 
        xyxy_tensor = detections[i].xyxy.cpu() #access tensor form of cordinates in CPU
        xyxy= xyxy_tensor.numpy().squeeze() #convert to numpy array
        xmin,ymin,xmax,ymax = xyxy.astype(int)

        class_id = int(detections[i].cls.item())
        classname = labels[class_id]
        confidence = float(detections[i].conf.item())

        if(confidence>0.5):
            detections_list.append([xmin, ymin, xmax, ymax, confidence])

    if (detections_list):
        detections_array = np.array(detections_list)
        tracked_objects= tracker.update(detections_array)
    else:
        tracked_objects= np.empty((0, 5))

    for obj in tracked_objects:
        xmin, ymin, xmax, ymax, obj_id = map(int, obj[:5])

        color= bbox_colors[class_id%10]
        cv2.rectangle(frame, (xmin,ymin),(xmax,ymax), color,2)
        
        label = f'{classname}: {int (confidence*100)}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

    #display detection results 
    if source_type=='video' or source_type=='usb' or source_type=='picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.pimshow('YOLO Detection results', frame)
    if record: recorder.write(frame)


    #wait for a user to press a key to go to next image or video 
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








