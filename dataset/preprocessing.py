'''
Code for preprocessing the video dataset 
Processes a video, generates face crop, and creates a constant crop for all face crops generated 
Writes the generated face crops (as frames) to a video
'''

import math
import os
import os.path as osp
from tqdm import tqdm
import gc
from glob import glob

import cv2
import matplotlib.pyplot as plt 
import mediapipe as mp

def display_image(image, requires_colorization=True):
    plt.figure()
    if requires_colorization:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    '''
    comptute the iou by taking the intersecting area over the sum of bounding boxes
    '''

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


'''
Function for writing frames to video 
'''
def write_frames_to_video(frames, current_frames, video_order_id, VIDEO_DIR, fps=30):
    height, width, _ = frames[0].shape
    # the frame coordinates have to be taken from the bounding box 
    video_path = osp.join(VIDEO_DIR, str(video_order_id).zfill(5) + '.mp4')
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_index in current_frames:
        video.write(frames[frame_index])
      
    cv2.destroyAllWindows() 
    video.release()
            
    print(f'Video {video_path} written successfully')
    
    
def crop_get_video(frames, current_indexes, bounding_box, VIDEO_DIR, video_order_id, fps=30):
    left, top, right, down = bounding_box['x1'], bounding_box['y1'], bounding_box['x2'], bounding_box['y2']
    width, height = right - left, down - top
    video_path = osp.join(VIDEO_DIR, str(video_order_id).zfill(5) + '.mp4')
    
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for index in current_indexes:
        current_frame = frames[index]
        cropped = current_frame[top : down, left : right]
        
        video.write(cropped) 

    print(f'Writing to file : {video_order_id}')
    cv2.destroyAllWindows()


'''
Function to return the bb coordinates of the cropped face 
'''
def crop_face_coordinates(image, x_px, y_px, width_px, height_px):
    # using different thresholds/bounds for the upper and lower faces 
    image_height, image_width, _ = image.shape
    lower_face_buffer, upper_face_buffer = 0.25, 0.65
    min_x, min_y, max_x, max_y = x_px, y_px, x_px + width_px, y_px + height_px

    x_left = max(0, int(min_x - (max_x - min_x) * lower_face_buffer))
    x_right = min(image_width, int(max_x + (max_x - min_x) * lower_face_buffer))
    y_top = max(0, int(min_y - (max_y - min_y) * upper_face_buffer))
    y_down = min(image_height, int(max_y + (max_y - min_y) * lower_face_buffer))

    size = max(x_right - x_left, y_down - y_top)
    sw = int((x_left + x_right)/2 - size // 2)
    
    return sw, y_top, sw+size, y_down


'''
function to return the bb coordinates given the image
'''
def bb_coordinates(image):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    result = results.detections
    image_height, image_width, _ = image.shape
    
    if result is None:
        return -1, -1, -1, -1
    
    bb_values = result[0].location_data.relative_bounding_box

    normalized_x, normalized_y, normalized_width, normalized_height = \
                bb_values.xmin, bb_values.ymin, bb_values.width, bb_values.height

    # the bounding box coordinates are given as normalized values, unnormalize them by multiplying by height and width
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    width_px = min(math.floor(normalized_width * image_width), image_width - 1)
    height_px = min(math.floor(normalized_height * image_height), image_height - 1)
    
    return x_px, y_px, width_px, height_px

'''
the formula for using the intersection over union works differently for different resolutions 
this is because it is dependent on the pixel information and not necessarily what's inside 
the frames are read till either i) end of the frames are reached, or ii) 4000 frames (cpu limit) are read
'''

def process_frames(frames, VIDEO_DIR):
    # declaring global scopre for the video_order_id variable
    global video_order_id
    
    current_frames = list()
    mean_bounding_box = dict()
    iou_threshold = 0.7
    frame_count = 0
    frame_writing_threshold = 30 # minimum number of frames to write
    bb_prev_mean = dict()

    for index, frame in tqdm(enumerate(frames)):
        image_height, image_width, _ = frame.shape
        x_px, y_px, width_px, height_px = bb_coordinates(frame)

        if x_px == -1:
            if len(current_frames) > frame_writing_threshold:
                crop_get_video(frames, current_frames, mean_bounding_box, VIDEO_DIR, video_order_id)
                video_order_id += 1

            # reset 
            current_frames = list()
            frame_count = 0
            mean_bounding_box = dict()
            bb_prev_mean = dict()

        else:
            left, top, right, bottom = crop_face_coordinates(frame, x_px, y_px, width_px, height_px)
            current_bounding_box = {'x1' : left, 'x2' : right, 'y1' : top, 'y2' : bottom}

            if len(mean_bounding_box) == 0:

                mean_bounding_box = current_bounding_box.copy()
                bb_prev_mean = current_bounding_box.copy()

                frame_count += 1
                current_frames.append(index)

            else:
                # UPDATE - compute the iou between the current bounding box and the mean bounding box
                iou = get_iou(bb_prev_mean, current_bounding_box)

                if iou < iou_threshold:
                    mean_left, mean_right, mean_top, mean_down = mean_bounding_box['x1'], mean_bounding_box['x2'], mean_bounding_box['y1'], mean_bounding_box['y2']

                    if len(current_frames) > frame_writing_threshold:
                        crop_get_video(frames, current_frames, mean_bounding_box, VIDEO_DIR, video_order_id)
                        video_order_id += 1

                    current_frames = list()
                    frame_count = 0
                    mean_bounding_box = dict()

                # Add the current bounding box to the list of bounding boxes and compute the mean
                else:
                    mean_bounding_box['x1'] = min(mean_bounding_box['x1'], current_bounding_box['x1'])
                    mean_bounding_box['y1'] = min(mean_bounding_box['y1'], current_bounding_box['y1'])
                    mean_bounding_box['x2'] = max(mean_bounding_box['x2'], current_bounding_box['x2'])
                    mean_bounding_box['y2'] = max(mean_bounding_box['y2'], current_bounding_box['y2'])

                    # update the coordinates of the mean bounding box 
                    for item in bb_prev_mean.keys():
                        bb_prev_mean[item] = int((bb_prev_mean[item] * frame_count + current_bounding_box[item])/(frame_count + 1))

                    frame_count += 1
                    current_frames.append(index)
                    
    if len(current_frames) > frame_writing_threshold:
        crop_get_video(frames, current_frames, mean_bounding_box, VIDEO_DIR, video_order_id)
        video_order_id += 1


'''
method for processing a single video
reads video frames and calls function to process the frames
'''
def process_video(video_file, processed_videos_dir):
    video_stream = cv2.VideoCapture(video_file)
    print(f'Total number of frames in the current video : {video_stream.get(cv2.CAP_PROP_FRAME_COUNT)}')
    print(f'Processing video file {video_file}')
    frames = list()

    # keep reading the frames if the processing of the current frames is over 
    frame_reading_threshold = 8000
    frames_processed = 0
    frames_processed_threshold = 2000
    
    video_file_name = osp.basename(video_file).split('.')[0]
    VIDEO_DIR = osp.join(processed_videos_dir, video_file_name)
    
    os.makedirs(VIDEO_DIR, exist_ok=True)

    ret, frame = video_stream.read()
    while ret:
        frames.append(frame)
        frames_processed += 1

        if frames_processed % frames_processed_threshold == 0:
            print(f'{frames_processed} frames read')

        if frames_processed%frame_reading_threshold == 0:
            # perform processing and generate frame videos 
            print(f'Processing the frames for the current batch')
            process_frames(frames, VIDEO_DIR)
            print(f'Done with processing of the current batch')

            del frames
            gc.collect()
            frames = list() # clear out the frames

        ret, frame = video_stream.read()
    
    # check if more frames that were read need to be processed
    if len(frames) != 0:
        print(f'Processing remaining frames')
        process_frames(frames, VIDEO_DIR)

    video_stream.release()


'''
method to process multiple videos
'''
def process_videos(video_dir):
    video_files = glob(video_dir + '/*.mp4')

    for video_file in tqdm(video_files):
        process_video(video_file)


if __name__ == '__main__':
    video_dir = 'videos_dir'
    process_videos(video_dir)