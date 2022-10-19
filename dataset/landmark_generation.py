# Multi GPU multi batch code for generating keypoints using the defined keypoint generator
import face_alignment
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import cv2
import os
import torch
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process
from itertools import product
import argparse
from glob import glob

ngpus = 1

# fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(id)) for id in range(ngpus)]
fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(0))]

def drawPolyline(image, landmarks, start, end, isClosed=False):
    points = []
    for i in range(start, end+1):
        point = [landmarks[i][0], landmarks[i][1]]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (0, 255, 255), 2, 16)

# Draw lines around landmarks corresponding to different facial regions
def drawPolylines(image, landmarks):
    drawPolyline(image, landmarks, 0, 16)           # Jaw line
    drawPolyline(image, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(image, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(image, landmarks, 27, 30)          # Nose bridge
    drawPolyline(image, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(image, landmarks, 36, 41, True)    # Left eye
    drawPolyline(image, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(image, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(image, landmarks, 60, 67, True)    # Inner lip

# Detect landmarks for the given batch
def batch_landmarks(batches, fa, gpu_id):
    landmarks_detected = 0
    batch_landmarks = list()

    for current_batch in batches:
        current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2).to('cuda:{}'.format(gpu_id))
        # current_batch = torch.from_numpy(np.asarray(current_batch)).permute(0, 3, 1, 2)
        landmarks = fa.get_landmarks_from_batch(current_batch)
        landmarks_detected += len(landmarks)
        batch_landmarks.extend(landmarks)

    return batch_landmarks, landmarks_detected

# def generate_landmarks_video(gpu_id_video_file, debug=False):

def generate_landmarks_video(video_file_gpu_id, debug=False):
    video_path, gpu_id = video_file_gpu_id
    batch_size = 32
    resize_dim = 256

    lower_face_buffer = 0.3
    upper_face_buffer = 0.8
    processed_folder = 'Processed'
    os.makedirs(processed_folder, exist_ok=True)

# for video_path in video_files:
    # video_path is like SpeakerVideos/AnfisaNava/video_id/XXXX.mp4
    # video_path, gpu_id = gpu_id_video_file
    speaker = video_path.split('/')[1]
    print(f'Processing video file : {video_path} for speaker : {speaker} using GPU : {gpu_id}', flush=True)

    bad_filepath = os.path.join(processed_folder, speaker + '_bad_files.txt')
    valid_filepath = os.path.join(processed_folder, speaker + '_landmark_files.txt')

    video_stream = cv2.VideoCapture(video_path)

    image_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Frames read : {total_frames}, image height and width: {image_height}, {image_width}', flush=True)
    # gpu_id = 0

    # NPZs go inside the folder 
    folder_name = '/'.join(video_path.split('/')[:-1])

    # If debug is enabled true, then image frames go inside the specific XXXX folder 
    image_folder_path = os.path.join(folder_name, os.path.basename(video_path).split('.')[0]) # SpeakerVideos/AnfisaNava/video_id/XXXX
    # out_folder_path = os.path.join('/home2/aditya1/text2face/temp_audio_files/', folder_name)
    # print(f'Image folder path : {image_folder_path}')
    # Create image folder only if images are to be saved
    if debug == True:
        os.makedirs(image_folder_path, exist_ok=True)

    frames = list()
    success, image = video_stream.read()
    while success:
        frames.append(image)
        success, image = video_stream.read()

    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    processed = False
    while not processed:
        try:
            if batch_size == 0:
                # Write this as a bad file
                with open(bad_filepath, 'a') as f:
                    f.write(video_path + '\n')
                continue
            landmarks, landmarks_detected = batch_landmarks(batches, fa[gpu_id], gpu_id)
            processed = True
        except Exception as e: # Exception arising out of CUDA memory unavailable
            print(e)
            batch_size = batch_size // 2
            print(f'Cuda memory unavailable, reducing batch size to : {batch_size}', flush=True)
            batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
            continue

    # print(f'Image frames generated : {len(landmarks)}')

    landmark_threshold = 68 # Ignore frames where landmarks detected is not equal to landmark_threshold
    frames_ignored = 0
    frame_ignore_threshold = 10 # reject video if more than 10% of frames are bad 
    
    resized_gt = list() # resized gt image
    # resized_image_landmarks = list() # resized gt image with landmarks drawn
    resized_landmarks = list() # resized image with just landmarks

    for i, landmark in enumerate(landmarks):
        image = frames[i]
        # print(f'{i}, landmark length : {np.asarray(landmark).shape}')
        # This is done to mostly prevent frames with more than one face from getting added 
        # Unfortunately, sometimes even with one face more than one set of landmarks are detected 
        # They can potentially be removed using common area over bounding boxes, ignored for now
        if (len(landmark) != landmark_threshold):
            frames_ignored += 1
            continue

        min_x, min_y, max_x, max_y = min(landmark[:, 0]), min(landmark[:, 1]), max(landmark[:, 0]), max(landmark[:, 1])

        # There is a possibility that the coordinates can exceed the bounds of the frame, modification of the coordinates
        x_left = max(0, int(min_x - (max_x - min_x) * lower_face_buffer))
        x_right = min(image_width, int(max_x + (max_x - min_x) * lower_face_buffer))
        y_top = max(0, int(min_y - (max_y - min_y) * upper_face_buffer))
        y_down = min(image_height, int(max_y + (max_y - min_y) * lower_face_buffer))

        # print(f'{i} Coordinates after modification : {x_left, x_right, y_top, y_down}')
        size = max(x_right - x_left, y_down - y_top)

        # add centering sceheme, the centering is done only on the width side 
        # the centering is done using the formula -> (x_left + x_right) / 2 - size // 2 : (x_left + x_right) / 2 + size // 2
        sw = int((x_left + x_right) / 2 - size // 2)

        # handling edge cases 
        if (sw < 0):
            sw = 0
        if (sw + size > image_width):
            frames_ignored += 1
            continue

        # generate the original image with just the crop according to the landmarks 
        original_cropped = image[y_top:y_down, sw:sw+size]
        resized_original = cv2.resize(original_cropped, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        resized_gt.append(resized_original)

        # draw the lines around the landmarks in the original image
        drawPolylines(image, landmark)

        # cropped image with the landmarks drawn
        # cropped_image_landmarks = image[y_top:y_down, sw:sw+size]
        # resized_im_landmarks = cv2.resize(cropped_image_landmarks, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        # resized_image_landmarks.append(resized_im_landmarks)

        # generate an image with just the face landmarks
        blank_image = np.ones((image_height, image_width), np.uint8)*255
        # draw the landmarks 
        drawPolylines(blank_image, landmark)

        # crop the landmark image and then resize 
        cropped_landmarks = blank_image[y_top:y_down, sw:sw+size]
        resized_cropped_landmarks = cv2.resize(cropped_landmarks, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        resized_landmarks.append(resized_cropped_landmarks)
        
    # check if we want to save the npz files for the videos 
    if (frames_ignored / total_frames) * 100 > frame_ignore_threshold:
        print(f'Bad video {video_path}, ignoring!, frames ignored : {frames_ignored}, total frames : {total_frames}', flush=True)
        with open(bad_filepath, 'a') as f:
            f.write(video_path + '\n')
        # continue
        return

    # save the npz files inside the folder, if debug is True, save the intermediate image files generated
    # files to save are the npz files for gt image, landmarks on gt image, raw landmarks (all images are cropped and resized)
    np.savez_compressed(os.path.join(folder_name, os.path.basename(video_path).split('.')[0] + '_gt.npz'), data=resized_gt)
    # np.savez_compressed(os.path.join(out_folder_path, 'image_landmarks.npz'), data=resized_image_landmarks)
    np.savez_compressed(os.path.join(folder_name, os.path.basename(video_path).split('.')[0] + '_landmarks.npz'), data=resized_landmarks)
    
    # write all valid videopaths to file
    with open(valid_filepath, 'a') as f:
        f.write(os.path.join(folder_name, os.path.basename(video_path).split('.')[0] + '_landmarks.npz' + '\n'))

    print(f'Video : {video_path}, Total frames : {total_frames}, gt frames : {len(resized_gt)}, landmark len : {len(resized_landmarks)}, using GPU : {gpu_id}', flush=True)

    # save image files only if debug is True
    if debug == True:
        for i in range(len(resized_gt)):
            gt_filepath = os.path.join(image_folder_path, 'gt_' + str(i+1).zfill(3) + '.jpg')
            # im_landmarks_filepath = os.path.join(out_folder_path, 'imlandmarks_' + str(i+1).zfill(3) + '.jpg')
            landmarks_filepath = os.path.join(image_folder_path, 'landmarks_' + str(i+1).zfill(3) + '.jpg')
            
            # write the images to disk if required
            cv2.imwrite(gt_filepath, resized_gt[i]) # ground truth
            # cv2.imwrite(im_landmarks_filepath, resized_image_landmarks[i])
            cv2.imwrite(landmarks_filepath, resized_landmarks[i]) # landmarks

# This code is used for detecting face along with generating landmarks for the face image 
def detect_face_generate_landmarks(vid_gpu, debug=False):
    video_path, gpu_id = vid_gpu
    batch_size = 32
    resize_dim = 256

    lower_face_buffer = 0.3
    upper_face_buffer = 0.8
    processed_folder = 'Processed'
    os.makedirs(processed_folder, exist_ok=True)

    speaker = video_path.split('/')[1]
    print(f'Processing video file : {video_path} for speaker : {speaker} using GPU : {gpu_id}', flush=True)

    bad_filepath = os.path.join(processed_folder, speaker + '_bad_files.txt')
    valid_filepath = os.path.join(processed_folder, speaker + '_landmark_files.txt')

    video_stream = cv2.VideoCapture(video_path)

    threshold_limit = 1000
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > threshold_limit:
        print(f'Execution failed for video : {video_file}, continuing')
        with open(bad_filepath, 'a') as f:
            f.write(video_path + '\n')
        return

    image_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video : {video_path}, Frames read : {total_frames}, image height and width: {image_height}, {image_width}', flush=True)

    # NPZs go inside the folder 
    folder_name = '/'.join(video_path.split('/')[:-1])

    # If debug is enabled true, then image frames go inside the specific XXXX folder 
    image_folder_path = os.path.join(folder_name, os.path.basename(video_path).split('.')[0]) # SpeakerVideos/AnfisaNava/video_id/XXXX
    if debug == True:
        os.makedirs(image_folder_path, exist_ok=True)

    frames = list()
    success, image = video_stream.read()
    while success:
        frames.append(image)
        success, image = video_stream.read()

    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    processed = False
    while not processed:
        try:
            if batch_size == 0:
                # Write this as a bad file
                with open(bad_filepath, 'a') as f:
                    f.write(video_path + '\n')
                continue
            landmarks, landmarks_detected = batch_landmarks(batches, fa[gpu_id], gpu_id)
            processed = True
        except Exception as e: # Exception arising out of CUDA memory unavailable
            print(e)
            batch_size = batch_size // 2
            print(f'Cuda memory unavailable, reducing batch size to : {batch_size}', flush=True)
            batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
            continue

    landmark_threshold = 68 # Ignore frames where landmarks detected is not equal to landmark_threshold
    frames_ignored = 0
    frame_ignore_threshold = 10 # reject video if more than 10% of frames are bad 
    
    resized_gt = list() # resized gt image
    resized_landmarks = list() # resized image with just landmarks

    for i, landmark in enumerate(landmarks):
        if (len(landmark) != landmark_threshold):
            frames_ignored += 1

    # check if the video file needs to be processed 
    if (frames_ignored/total_frames)*100 > frame_ignore_threshold:
        print(f'Bad video {video_path}, ignoring!, frames ignored : {frames_ignored}, total frames : {total_frames}', flush=True)
        with open(bad_filepath, 'a') as f:
            f.write(video_path + '\n')
        return

    # process the video file otherwise 
    for i, landmark in enumerate(landmarks):
        image = frames[i]

        if (len(landmark) != landmark_threshold):
            continue # This helps skip processing the current frame and skip to the next frame

        min_x, min_y, max_x, max_y = min(landmark[:, 0]), min(landmark[:, 1]), max(landmark[:, 0]), max(landmark[:, 1])

        # There is a possibility that the coordinates can exceed the bounds of the frame, modification of the coordinates
        x_left = max(0, int(min_x - (max_x - min_x) * lower_face_buffer))
        x_right = min(image_width, int(max_x + (max_x - min_x) * lower_face_buffer))
        y_top = max(0, int(min_y - (max_y - min_y) * upper_face_buffer))
        y_down = min(image_height, int(max_y + (max_y - min_y) * lower_face_buffer))

        size = max(x_right - x_left, y_down - y_top)

        sw = int((x_left + x_right) / 2 - size // 2)

        # handling edge cases 
        if (sw < 0):
            sw = 0
        if (sw + size > image_width):
            frames_ignored += 1
            continue

        # generate the original image with just the crop according to the landmarks 
        original_cropped = image[y_top:y_down, sw:sw+size]
        resized_original = cv2.resize(original_cropped, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        resized_gt.append(resized_original)

        # draw the lines around the landmarks in the original image
        drawPolylines(image, landmark)

        # generate an image with just the face landmarks
        blank_image = np.ones((image_height, image_width), np.uint8)*255
        # draw the landmarks 
        drawPolylines(blank_image, landmark)

        # crop the landmark image and then resize 
        cropped_landmarks = blank_image[y_top:y_down, sw:sw+size]
        resized_cropped_landmarks = cv2.resize(cropped_landmarks, (resize_dim, resize_dim), cv2.INTER_LINEAR)
        resized_landmarks.append(resized_cropped_landmarks)
    
    # save the npz files, if debug is True, save the intermediate files generated 
    # files save are gt image, and landmarks 
    np.savez_compressed(os.path.join(folder_name, os.path.basename(video_path).split('.')[0] + '_gt.npz'), data=resized_gt)
    # np.savez_compressed(os.path.join(out_folder_path, 'image_landmarks.npz'), data=resized_image_landmarks)
    np.savez_compressed(os.path.join(folder_name, os.path.basename(video_path).split('.')[0] + '_landmarks.npz'), data=resized_landmarks)
    
    # write valid video to valid filepath
    with open(valid_filepath, 'a') as f:
        f.write(os.path.join(folder_name, os.path.basename(video_path).split('.')[0] + '_landmarks.npz' + '\n'))

    print(f'Video : {video_path}, Total frames : {total_frames}, gt frames : {len(resized_gt)}, landmark len : {len(resized_landmarks)}, using GPU : {gpu_id}', flush=True)

    # save image files only if debug is True
    if debug == True:
        for i in range(len(resized_gt)):
            gt_filepath = os.path.join(image_folder_path, 'gt_' + str(i+1).zfill(3) + '.jpg')
            landmarks_filepath = os.path.join(image_folder_path, 'landmarks_' + str(i+1).zfill(3) + '.jpg')            
            cv2.imwrite(gt_filepath, resized_gt[i]) # ground truth
            cv2.imwrite(landmarks_filepath, resized_landmarks[i]) # landmarks

if __name__ == '__main__':
    p = ThreadPoolExecutor(ngpus)

    default_batch_size = 64
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--speakers', help='Speakers')
    parser.add_argument('-b', '--batch_size', help='Batch size', default=default_batch_size)
    args = parser.parse_args()

    files_to_process = list()

    dirname = '/ssd_scratch/cvit/aditya1/rebuttal_scores_validation/source_gt'
    files_to_process = glob(dirname + '/*.mp4')
    
    jobs = [(video_file, job_id%ngpus) for job_id, video_file in enumerate(files_to_process)]
    futures = [p.submit(detect_face_generate_landmarks, job) for job in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]