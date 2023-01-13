import sys
import random
from glob import glob
import os
import json
import os.path as osp

from skimage import io
from skimage.transform import resize
from scipy.ndimage import laplace
import numpy as np
import cv2
from skimage import transform as tf

from scipy.ndimage import laplace

target_without_face_apply = True

def resize_frame(frame, resize_dim=256):
    h, w, _ = frame.shape

    if h > w:
        padw, padh = (h-w)//2, 0
    else:
        padw, padh = 0, (w-h)//2

    padded = cv2.copyMakeBorder(frame, padh, padh, padw, padw, cv2.BORDER_CONSTANT, value=0)
    padded = cv2.resize(padded, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

    return padded

def readPoints(nparray) :
    points = []
    
    for row in nparray:
        x, y = row[0], row[1]
        points.append((int(x), int(y)))
    
    return points

def generate_convex_hull(img, points):
    # points = np.load(landmark_path, allow_pickle=True)['landmark'].astype(np.uint8)
    points = readPoints(points)

    hull = []
    hullIndex = cv2.convexHull(np.array(points), returnPoints = False)

    for i in range(0, len(hullIndex)):
        hull.append(points[int(hullIndex[i])])

    sizeImg = img.shape   
    rect = (0, 0, sizeImg[1], sizeImg[0])

    hull8U = []
    for i in range(0, len(hull)):
        hull8U.append((hull[i][0], hull[i][1]))

    mask = np.zeros(img.shape, dtype = img.dtype)  

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    # convex_face = ((mask/255.) * img).astype(np.uint8)

    return mask

def enlarge_mask(img_mask, enlargement=5):
    img1 = img_mask.copy()
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(img,50,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        cv2.drawContours(img1, [hull], -1, (255, 255, 255), enlargement)

    return img1
        
def poisson_blend(target_img, src_img, mask_img, iter: int = 1024):
    for _ in range(iter):
        target_img = target_img + 0.25 * mask_img * laplace(target_img - src_img)
    return target_img.clip(0, 1)

# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped

# Method to combine the face segmented with the face segmentation mask
def combine_images(face_mask, face_image, generate_mask=True):
    image_masked = face_mask.copy()
    if generate_mask:
        mask = face_image[..., 0] != 0
        image_masked[mask] = 0
    
    combined_image = image_masked + face_image
    
    return combined_image

# computes the rotation of the face using the angle of the line connecting the eye centroids 
def compute_rotation(shape):
    # landmark coordinates corresponding to the eyes 
    lStart, lEnd = 36, 41
    rStart, rEnd = 42, 47

    # landmarks for the left and right eyes 
    leftEyePoints = shape[lStart:lEnd]
    rightEyePoints = shape[rStart:rEnd]

    # compute the center of mass for each of the eyes 
    leftEyeCenter = leftEyePoints.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePoints.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) 
    
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
    
    dist = np.sqrt((dX ** 2) + (dY ** 2)) # this indicates the distance between the two eyes 
    
    return angle, eyesCenter, dist

def apply_mask(mask, image):
    return ((mask / 255.) * image).astype(np.uint8)

# code to generate the alignment between the source and the target image 
def generate_warped_image(source_landmark_npz, target_landmark_npz, 
                            source_image_path, target_image_path,
                            poisson_blend_required = False,
                         require_full_mask = False):
    
    stablePoints = [33, 36, 39, 42, 45]
    std_size = (256, 256)
    
    source_image = resize_frame(io.imread(source_image_path))
    target_image = resize_frame(io.imread(target_image_path))
    
    source_landmarks = np.load(source_landmark_npz)['landmark']

    target_landmarks = np.load(target_landmark_npz)['landmark']
    
    if require_full_mask:
        source_convex_mask = generate_convex_hull(source_image, source_landmarks)
        source_convex_mask_no_enlargement = source_convex_mask.copy()
    else:
        source_convex_mask = generate_convex_hull(source_image, source_landmarks[17:])
        # enlarge the convex mask
        source_convex_mask_no_enlargement = source_convex_mask.copy()
        source_convex_mask = enlarge_mask(source_convex_mask, enlargement=10)
        
    # apply the convex mask to the face 
    source_face_segmented = apply_mask(source_convex_mask, source_image)
    source_face_transformed, transformation = warp_img(source_landmarks[stablePoints, :],
                             target_landmarks[stablePoints, :],
                             source_face_segmented, 
                             std_size)
    
    source_convex_mask_transformed = apply_transform(transformation, source_convex_mask, std_size)
    source_convex_mask_no_enlargement_transformed = apply_transform(transformation, source_convex_mask_no_enlargement, std_size)
    source_image_transformed = apply_transform(transformation, source_image, std_size)

    target_convex_mask = np.invert(generate_convex_hull(target_image, target_landmarks))
    # target_background = apply_mask(target_convex_mask, target_image)
    
    target_convex_mask_without_jaw = generate_convex_hull(target_image, target_landmarks[17:])
    target_convex_mask_without_jaw = enlarge_mask(target_convex_mask_without_jaw, enlargement=10)
    
    target_convex_mask_without_jaw = np.invert(target_convex_mask_without_jaw)
    target_without_face_features = apply_mask(target_convex_mask_without_jaw, target_image)
    target_without_face = apply_mask(target_convex_mask, target_image)

    if poisson_blend_required:
        combined_image = poisson_blend(target_image/255., source_image/255., source_face_transformed/255.)
    else:
        if target_without_face_apply:
            combined_image = combine_images(target_without_face, source_face_transformed)
        else:
            combined_image = combine_images(target_image, source_face_transformed)
    # apply the transformed convex mask to the target face for sanity
    # target_masked = apply_mask(source_convex_mask_transformed, target_image)
    
    return source_face_transformed, source_convex_mask_transformed, source_image_transformed, source_convex_mask_no_enlargement, target_image, target_convex_mask, combined_image, target_without_face_features, source_image

# code to generate the alignment between the source and the target image 
def generate_aligned_image(source_landmark_npz, target_landmark_npz, 
                            source_image_path, target_image_path,
                            poisson_blend_required = False,
                          require_full_mask = False):
    
    source_image = resize_frame(io.imread(source_image_path))
    target_image = resize_frame(io.imread(target_image_path))
    
    source_landmarks = np.load(source_landmark_npz)['landmark']
    source_rotation, source_center, source_distance = compute_rotation(source_landmarks)

    target_landmarks = np.load(target_landmark_npz)['landmark']
    target_rotation, target_center, target_distance = compute_rotation(target_landmarks)

    # rotation of the target conditioned on the source orientation 
    target_conditioned_source_rotation = source_rotation - target_rotation
    
    # calculate the scaling that needs to be applied on the source image 
    scaling = target_distance / source_distance

    # apply the rotation on the source image
    height, width = 256, 256
    # print(f'Angle of rotation is : {target_conditioned_source_rotation}')
    rotate_matrix = cv2.getRotationMatrix2D(center=source_center, angle=target_conditioned_source_rotation, scale=scaling)

    # calculate the translation component of the matrix M 
    rotate_matrix[0, 2] += (target_center[0] - source_center[0])
    rotate_matrix[1, 2] += (target_center[1] - source_center[1])
    
    if require_full_mask:
        source_convex_mask = generate_convex_hull(source_image, source_landmarks)
    else:
        source_convex_mask = generate_convex_hull(source_image, source_landmarks[17:])
        # enlarge the convex mask using the enlargement
        source_convex_mask = enlarge_mask(source_convex_mask, enlargement=5)
        
    # apply the convex mask to the face 
    source_face_segmented = apply_mask(source_convex_mask, source_image)
    source_face_transformed = cv2.warpAffine(source_face_segmented, rotate_matrix, (width, height), flags=cv2.INTER_CUBIC)
    source_convex_mask_transformed = cv2.warpAffine(source_convex_mask, rotate_matrix, (width, height), flags=cv2.INTER_CUBIC)
    source_image_transformed = cv2.warpAffine(source_image, rotate_matrix, (width, height), flags=cv2.INTER_CUBIC)

    # used for computing the target background
    target_convex_mask = np.invert(generate_convex_hull(target_image, target_landmarks))
    # target_background = ((target_convex_mask/255.)*target_image).astype(np.uint8)
    target_convex_mask_without_jaw = np.invert(generate_convex_hull(target_image, target_landmarks[17:]))
    target_without_face_features = apply_mask(target_convex_mask_without_jaw, target_image)
    target_without_face = apply_mask(target_convex_mask, target_image)

    if poisson_blend_required:
        combined_image = poisson_blend(target_image/255., source_image/255., source_face_transformed/255.)
    else:
        if target_without_face_apply:
            combined_image = combine_images(target_without_face, source_face_transformed)
        else:
            combined_image = combine_images(target_image, source_face_transformed)

    return source_face_transformed, source_convex_mask_transformed, source_image_transformed, target_image, target_convex_mask, combined_image
