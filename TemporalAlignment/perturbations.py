'''
This file generates the paired data by introducing motion errors (aka perturbations) in the source face
The imperfectly blended face is a 6 channel image which the model has to generate to denoise in a temporal fashion
'''

import sys
import os
import os.path as osp
import random
from glob import glob
from tqdm import tqdm
from enum import Enum

from TemporalAlignment.ranges import *

import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image
from wand.image import Image as WandImage

import cv2
import numpy as np

'''
The following perturbation functions are added -- 
1. Affine transformations - translation, rotation, resize
2. Color transformations - change hue, saturation etc
3. Non-linear transformations aka distortions - arc, barrel, barrel_inverse
'''

'''
types of distortions
'''
class Distortion(Enum):
    ARC = 1,
    BARREL = 2,
    BARREL_INVERSE = 3,


'''
horizontal image translation -- affine transformation
'''
def translate_horizontal(x, image):
    M = np.float32([
        [1, 0, x],
        [0, 1, 0]
    ])
    
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

'''
vertical image translation -- affine transformation
'''
def translate_vertical(y, image):
    M = np.float32([
        [1, 0, 0],
        [0, 1, y]
    ])
    
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return shifted

'''
image rotation (clockwise or anti-clockwise) -- affine transformation 
'''
def rotate_image(rotation, image, center=None):
    # rotates the image by the center of the image
    h, w = image.shape[:2]

    if center is None:
        cX, cY = (w//2, h//2)
        M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
    else:
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
    
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

'''
image resize operations (zoom in and zoom out) -- not used currently
'''
def resize_image(magnification, image):
    res = cv2.resize(image, None, fx=magnification, fy=magnification, interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    
    if magnification >= 1:
        cX, cY = res.shape[1] // 2, res.shape[0] // 2
        left_index = cX - w // 2
        upper_index = cY - h // 2
        modified_image = res[upper_index : upper_index + h, left_index : left_index + w]
    else:
        modified_image = np.zeros((image.shape), dtype=np.uint8)
        hs, ws = res.shape[:2]
        difference_h = h - hs
        difference_w = w - ws
        left_index = difference_w // 2
        upper_index = difference_h // 2
        modified_image[upper_index : upper_index + hs, left_index : left_index + ws] = res
        
    return modified_image

'''
applies shear transformation along both the axes -- not used currently
'''
def shear_image(shear, image):
    shear_x, shear_y = shear, shear
    M = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    sheared = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return sheared

'''
flips the imagein the horizontal direction -- not used currently
'''
def image_flip(flip_code, image):
    flipped_image = cv2.flip(image, int(flip_code))
    return flipped_image

'''
applies non-linear transformation (distortion) to the image
'''
def distort_image(distortion_type, image):
    img_file = np.array(image)
    with WandImage.from_array(img_file) as img:
        '''
        three type of image distortions are supported - i) arc, ii) barrel, and iii) inverse-barrel
        '''
        if distortion_type == Distortion.ARC.value:
            distortion_angle = random.randint(0, 30)
            img.distort('arc', (distortion_angle,))
            img.resize(image.shape[0], image.shape[1])

            opencv_image = np.array(img)

        elif distortion_type == Distortion.BARREL.value:
            a = random.randint(0, 10)/10
            b = random.randint(2, 7)/10
            c = random.randint(0, 5)/10
            d = random.randint(10, 10)/10
            args = (a, b, c, d)
            img.distort('barrel', (args))
            img.resize(image.shape[0], image.shape[1])

            opencv_image = np.array(img)

        else:
            b = random.randint(0, 2)/10
            c = random.randint(-5, 0)/10
            d = random.randint(10, 10)/10
            args = (0.0, b, c, d)
            img.distort('barrel_inverse', (args))
            img.resize(image.shape[0], image.shape[1])

            opencv_image = np.array(img)

    return opencv_image

'''
blend the perturbed image and the masked image
'''
def combine_images(face_mask, perturbed_image, generate_mask=True):
    image_masked = face_mask.copy()
    if generate_mask:
        mask = perturbed_image[..., 0] != 0
        image_masked[mask] = 0
    
    combined_image = image_masked + perturbed_image
    
    return combined_image

'''
finds the center of the eyes from the face landmarks
'''
def find_eye_center(shape):
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

'''
applies composite perturbations to a single image 
the perturbations to apply are selected randomly
'''
def perturb_image_composite(face_image, landmark):
    perturbation_functions = [
        translate_horizontal,
        translate_vertical,
        rotate_image,
        resize_image,
        # shear_image,
        # image_flip,
        distort_image,
    ]

    perturbation_function_map = {
        translate_horizontal : [-translation_range, translation_range, 1],
        translate_vertical : [-translation_range, translation_range, 1],
        rotate_image : [-rotation_range, rotation_range, 1],
        resize_image : [scale_ranges[0], scale_ranges[1], 100],
        # shear_image : [-10, 10, 100],
        # image_flip : [1, 1, 1],
        distort_image : [0, len(Distortion), 1],
    }

    gt_transformations = {
        'translate_horizontal': 0, 
        'translate_vertical': 0, 
        'rotate_image': 0
    }

    eyes_center = find_eye_center(landmark)

    # maintains the perturbations to apply to the image
    composite_perturbations = list()
    # ensures atleast one perturbation is produced
    while len(composite_perturbations) == 0:
        for i, perturbation_function in enumerate(perturbation_functions):
            if random.randint(0, 1):
                composite_perturbations.append(perturbation_function)

    # print(f'Perturbations applied : {composite_perturbations}', flush=True)

    for perturbation_function in composite_perturbations:
        perturbation_map = perturbation_function_map[perturbation_function]
        perturbation_value = random.randint(perturbation_map[0], perturbation_map[1])/perturbation_map[2]
        normalized_value = perturbation_value/perturbation_map[1]

        if perturbation_function == translate_horizontal:
            gt_transformations['translate_horizontal'] = perturbation_value
        elif perturbation_function == translate_vertical:
            gt_transformations['translate_vertical'] = perturbation_value
        else:
            gt_transformations['rotate_image'] = perturbation_value

        if perturbation_function == rotate_image:
            face_image = perturbation_function(perturbation_value, face_image, center=eyes_center)
        else:
            face_image = perturbation_function(perturbation_value, face_image)

    return face_image, gt_transformations

'''
the function specifies the perturbation functions and the amounts by which corresponding perturbations are applied
the perturbed image (foreground) is combined with the face_mask (background) to reproduce blending seen due to two different flows
applies multiple perturbations (composite perturbations) to generate complex perturbations
'''
def perturb_image(face_image):
    perturbation_functions = [
        translate_horizontal,
        translate_vertical,
        rotate_image,
        resize_image
    ]

    perturbation_function_map = {
        translate_horizontal : [-20, 20, 1],
        translate_vertical : [-20, 20, 1],
        rotate_image : [-25, 25, 1],
        resize_image : [90, 110, 100]
    }

    random_perturbation_index = random.randint(0, len(perturbation_functions)-1)
    # random_perturbation_index = 0 # used for debugging
    perturbation_function = perturbation_functions[random_perturbation_index]
    perturbation_map = perturbation_function_map[perturbation_function]
    perturbation_value = random.randint(perturbation_map[0], perturbation_map[1])/perturbation_map[2]
    # print(f'Using perturbation : {random_perturbation_index}, with value : {perturbation_value}', flush=True)
    intermediate_perturbed_image = perturbation_function(perturbation_value, face_image)
    # perturbed_image = combine_images(face_mask, intermediate_perturbed_image)

    return intermediate_perturbed_image

'''
def test_sample():
    file = '/ssd_scratch/cvit/aditya1/CelebA-HQ-img/13842.jpg'
    gpu_id = 0
    parsing, image = generate_segmentation(file, gpu_id)
    for i in range(PERTURBATIONS_PER_IDENTITY):
        face_image, background_image = generate_segmented_face(parsing, image)
        perturbed_image = perturb_image_composite(face_image, background_image)
        perturbed_filename, extension = osp.basename(file).split('.')
        perturbed_image_path = osp.join(perturbed_image_dir, perturbed_filename + '_' + str(i) + '.' + extension)
        save_image(perturbed_image_path, perturbed_image)
'''