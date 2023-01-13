import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import utils


def save_frames_as_video(frames, video_path, fps=30):
    height, width, layers = frames[0].shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames: 
        video.write(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR)) 
      
    cv2.destroyAllWindows() 
    video.release()

def save_image(data, saveas, video=False):
    utils.save_image(
        data,
        saveas,
        nrow=data.shape[0]//2,
        normalize=True,
        range=(-1, 1),
    )
 

def process_data(data, device, dataset):
    source, target, background, source_images, source_images_original = data
    
    img = torch.cat([source, background], axis=2).squeeze(0).to(device)
    source = source.squeeze(0)
    ground_truth = source_images.squeeze(0).to(device)

    S = source.shape[0]

    return img, S, ground_truth, source_images_original.squeeze(0).to(device)


'''
Conv3D based temporal module is added before the quantization step
LPIPS based perceptual loss is added
'''
def get_facetranslation_latent_conv_perceptual(args, device):
    from TemporalAlignment.dataset import TemporalAlignmentDataset 
    from models.vqvae_conv3d_latent import VQVAE
    from loss import VQLPIPS

    print(f'Inside conv3d applied before quantization along with perceptual loss')

    model = VQVAE(in_channel=3*2).to(device)
    vqlpips = VQLPIPS().to(device)

    train_dataset = TemporalAlignmentDataset(
        'train', 30, 
        color_jitter_type=args.colorjit,
        grayscale_required=args.gray)

    val_dataset = TemporalAlignmentDataset(
        'val', 50, 
        color_jitter_type=args.colorjit,
        cross_identity_required=args.crossid,
        grayscale_required=args.gray,
        custom_validation_required=args.custom_validation,
        validation_datapoints=args.validation_folder)

    try:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=1,
            shuffle=True, 
            num_workers=2)
    except:
        train_loader = None
        
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        batch_size=1,
        num_workers=2)

    return train_loader, val_loader, model, vqlpips


'''
get the loaders and the models
'''
def get_loaders_and_models(args, device):
    return get_facetranslation_latent_conv_perceptual(args, device)