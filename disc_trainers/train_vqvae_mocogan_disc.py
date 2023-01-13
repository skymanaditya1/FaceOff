# This code uses the mocogan content and temporal discriminators

import argparse
import sys
import os
import random
import os.path as osp

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms, utils

from TemporalAlignment.models import mocogan_discriminator

from tqdm import tqdm

from scheduler import CycleScheduler

from utils import *
from config import DATASET, LATENT_LOSS_WEIGHT, image_disc_weight, video_disc_weight, SAMPLE_SIZE_FOR_VISUALIZATION

criterion = nn.MSELoss()

gan_criterion = nn.BCEWithLogitsLoss()

global_step = 0

sample_size = SAMPLE_SIZE_FOR_VISUALIZATION

dataset = DATASET

CONST_FRAMES_TO_CHECK = 16

BASE = '/ssd_scratch/cvit/aditya1/video_vqvae2_results'
# sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/{}'


def run_step(model, data, device, run='train'):
    img, S, ground_truth, source_image = process_data(data, device, dataset)

    out, latent_loss = model(img)

    out = out[:, :3]
    
    recon_loss = criterion(out, ground_truth)
    latent_loss = latent_loss.mean()
    
    if run == 'train':
        return recon_loss, latent_loss, S, out, ground_truth
    else:
        return ground_truth, img, out, source_image

def run_step_custom(model, data, device, run='train'):
    img, S, ground_truth = process_data(data, device, dataset)

    out, latent_loss = model(img)

    out = out[:, :3] # first 3 channels of the prediction

    return out, ground_truth


def jitter_validation(model, val_loader, device, epoch, i, run_type, sample_folder):
    for i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            source_images, input, prediction, source_images_original = run_step(model, data, device, run='val')
            
        source_hulls = input[:, :3]
        background = input[:, 3:]

        saves = {
            'source': source_hulls,
            'background': background,
            'prediction': prediction,
            'source_images': source_images,
            'source_original': source_images_original
        }

        # if i % (len(val_loader) // 10) == 0 or run_type != 'train':
        if True:
            def denormalize(x):
                return (x.clamp(min=-1.0, max=1.0) + 1)/2

            for name in saves:
                saveas = f"{sample_folder}/{epoch + 1}_{global_step}_{i}_{name}.mp4"
                frames = saves[name].detach().cpu()
                frames = [denormalize(x).permute(1, 2, 0).numpy() for x in frames]

                # os.makedirs(sample_folder, exist_ok=True)
                save_frames_as_video(frames, saveas, fps=25)

def base_validation(model, val_loader, device, epoch, i, run_type, sample_folder):
    def get_proper_shape(x):
        shape = x.shape
        return x.view(shape[0], -1, 3, shape[2], shape[3]).view(-1, 3, shape[2], shape[3])

    for val_i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            sample, _, out, source_img = run_step(model, data, device, 'val')

        if val_i % (len(val_loader)//10) == 0: # save 10 results

            def denormalize(x):
                return (x.clamp(min=-1.0, max=1.0) + 1)/2
            
            save_as_sample = f"{sample_folder}/{val_i}_sample.mp4"
            save_as_out = f"{sample_folder}/{val_i}_out.mp4"

            sample = sample.detach().cpu()
            sample = [denormalize(x).permute(1, 2, 0).numpy() for x in sample]

            out = out.detach().cpu()
            out = [denormalize(x).permute(1, 2, 0).numpy() for x in out]

            save_frames_as_video(sample, save_as_sample, fps=25)
            save_frames_as_video(out, save_as_out, fps=25)

def validation(model, val_loader, device, epoch, i, sample_folder, run_type='train'):
    if dataset >= 6:
        jitter_validation(model, val_loader, device, epoch, i, run_type, sample_folder)
    else:
        base_validation(model, val_loader, device, epoch, i, run_type, sample_folder)

        
def flip_video(x):
    num = random.randint(0, 1)
    if num == 0:
        return torch.flip(x, [2])
    else:
        return x


# inside the train discriminator - disc, real, fake is required
def train_discriminator(opt, discriminator, real_batch, fake_batch):
    opt.zero_grad()

    real_disc_preds, _ = discriminator(real_batch)
    fake_disc_preds, _ = discriminator(fake_batch.detach())

    ones = torch.ones_like(real_disc_preds)
    zeros = torch.zeros_like(fake_disc_preds)

    l_discriminator = (gan_criterion(real_disc_preds, ones) + gan_criterion(fake_disc_preds, zeros))/2

    l_discriminator.backward()
    opt.step()

    return l_discriminator

def train_generator(opt, image_discriminator, video_discriminator, 
                    fake_batch, recon_loss, latent_loss):
    opt.zero_grad()

    # image disc predictions 
    fake_image_disc_preds, _ = image_discriminator(fake_batch)
    all_ones = torch.ones_like(fake_image_disc_preds)
    fake_image_disc_loss = gan_criterion(fake_image_disc_preds, all_ones)
    
    # video disc predictions 
    fake_video_batch = fake_batch.unsqueeze(0).permute(0, 2, 1, 3, 4)
    fake_video_disc_preds, _ = video_discriminator(fake_video_batch)
    all_ones = torch.ones_like(fake_video_disc_preds)
    fake_video_disc_loss = gan_criterion(fake_video_disc_preds, all_ones)

    gen_loss = recon_loss \
        + LATENT_LOSS_WEIGHT * latent_loss \
        + image_disc_weight * fake_image_disc_loss \
        + video_disc_weight * fake_video_disc_loss
    gen_loss.backward(retain_graph=True)
    opt.step()

    return gen_loss

# training is done in an alternating fashion 
# disc and gen alternate their training
def train(model, patch_image_disc, patch_video_disc, 
        gen_optim, image_disc_optim, video_disc_optim, 
        loader, val_loader, scheduler, device, 
        epoch, validate_at, checkpoint_dir, sample_folder):

    SAMPLE_FRAMES = 16 # sample 16 frames for the discriminator

    for i, data in enumerate(loader):

        global global_step

        global_step += 1

        model.train()
        patch_image_disc.train()
        patch_video_disc.train()

        model.zero_grad()
        patch_image_disc.zero_grad()
        patch_video_disc.zero_grad()

        # train video generator
        recon_loss, latent_loss, S, out, ground_truth = run_step(model, data, device)

        # print(f'Frames : {out.shape[0]}')

        # skip if the number of frames is less than SAMPLE_FRAMES 
        if out.shape[0] < SAMPLE_FRAMES:
            print(f'Encountered {out.shape[0]} frames which is less than {SAMPLE_FRAMES}. Continuing ...')
            continue

        # sample SAMPLE_FRAMES frames from out
        fake_sampled = out[:SAMPLE_FRAMES] # dim -> SAMPLE_FRAMES x 3 x 256 x 256
        real_sampled = ground_truth[:SAMPLE_FRAMES] # dim -> SAMPLE_FRAMES x 3 x 256 x 256
        
        # print(f'fake sampled : {fake_sampled.shape}')
        gen_loss = train_generator(gen_optim, patch_image_disc, patch_video_disc, fake_sampled, recon_loss, latent_loss)

        # train image discriminator
        l_image_dis = train_discriminator(image_disc_optim, patch_image_disc, real_sampled, fake_sampled)
    
        # train video discriminator - adding the batch dimension
        l_video_dis = train_discriminator(video_disc_optim, patch_video_disc, 
                        real_sampled.unsqueeze(0).permute(0, 2, 1, 3, 4),
                        fake_sampled.unsqueeze(0).permute(0, 2, 1, 3, 4))

        lr = gen_optim.param_groups[0]["lr"]

        # indicates that both the generator and discriminator steps would have been performed
        print(f'Epoch : {epoch+1}, step : {global_step}, gen loss : {gen_loss.item():.5f}, image disc loss : {l_image_dis.item():.5f}, video disc loss : {l_video_dis.item():.5f}, lr : {lr:.5f}')

        # check if validation is required 
        if i%validate_at == 0:
            # set the model to eval and generate the predictions
            model.eval()

            validation(model, val_loader, device, epoch, i, sample_folder)

            os.makedirs(checkpoint_dir, exist_ok=True)

            # save the vqvae2 generator weights 
            torch.save(model.state_dict(), f"{checkpoint_dir}/vqvae_{epoch+1}_{str(global_step).zfill(4)}.pt")

            # save the discriminator weights 
            # save the video disc weights
            torch.save(patch_video_disc.state_dict(), f"{checkpoint_dir}/patch_video_disc_{epoch+1}_{str(global_step).zfill(4)}.pt")

            # save the image/content disc weights  
            torch.save(patch_image_disc.state_dict(), f"{checkpoint_dir}/patch_image_disc_{epoch+1}_{str(global_step).zfill(4)}.pt")

            # reverse the training state of the model
            model.train()


def main(args):
    device = "cuda"

    default_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_loader, val_loader, model, patch_image_disc, patch_video_disc, \
        image_disc_optim, video_disc_optim = \
            get_loaders_and_models(args, dataset, default_transform, device, test=args.test)


    # load the models on the gpu 
    model = model.to(device)
    patch_image_disc = patch_image_disc.to(device)
    patch_video_disc = patch_video_disc.to(device)

    # loading the pretrained generator (vqvae2) model weights
    if args.ckpt:
        print(f'Loading pretrained generator model : {args.ckpt}')
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        try:
            model.module.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict)

        # load the image and video disc models if required 
        if args.load_disc:
            # image_disc_path = 'patch_image_disc_' + args.ckpt.split('_', 1)[1]
            # video_disc_path = 'patch_video_disc_' + args.ckpt.split('_', 1)[1]

            image_disc_path = osp.join(osp.dirname(args.ckpt), 'patch_image_disc_' + osp.basename(args.ckpt).split('_', 1)[1])
            video_disc_path = osp.join(osp.dirname(args.ckpt), 'patch_video_disc_' + osp.basename(args.ckpt).split('_', 1)[1])

            print(f'Loading pretrained disc models : {image_disc_path}, {video_disc_path}')

            image_disc_state_dict = torch.load(image_disc_path)
            image_disc_state_dict = { k.replace('module.', ''): v for k, v in image_disc_state_dict.items() }  
            
            video_disc_state_dict = torch.load(video_disc_path)
            video_disc_state_dict = { k.replace('module.', ''): v for k, v in video_disc_state_dict.items() }

            try:
                patch_image_disc.module.load_state_dict(image_disc_state_dict)
            except:
                patch_image_disc.load_state_dict(image_disc_state_dict)

            try:
                patch_video_disc.module.load_state_dict(video_disc_state_dict)
            except:
                patch_video_disc.load_state_dict(video_disc_state_dict)

    if args.test:
        validation(model, val_loader, device, 0, 0, args.sample_folder, 'val')
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        scheduler = None
        
        if args.sched == "cycle":
            scheduler = CycleScheduler(
                optimizer,
                args.lr,
                n_iter=len(train_loader) * args.epoch,
                momentum=None,
                warmup_proportion=0.05,
            )

        for i in range(args.epoch):
            
            train(model, patch_image_disc, patch_video_disc, 
                optimizer, image_disc_optim, video_disc_optim,
                train_loader, val_loader, scheduler, device, i, 
                args.validate_at, args.checkpoint_dir, args.sample_folder)


def get_random_name(cipher_length=5):
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join([chars[random.randint(0, len(chars)-1)] for i in range(cipher_length)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = random.randint(51000, 52000)

    # parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--checkpoint_suffix", type=str, default='')
    parser.add_argument("--validate_at", type=int, default=1024)
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)
    parser.add_argument("--gray", action='store_true', required=False)
    parser.add_argument("--colorjit", type=str, default='', help='const or random or empty')
    parser.add_argument("--crossid", action='store_true', required=False)
    parser.add_argument("--sample_folder", type=str, default='samples')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoint')
    parser.add_argument("--validation_folder", type=str, default=None)
    parser.add_argument("--custom_validation", action='store_true', required=False)
    parser.add_argument("--load_disc", action='store_true', required=False)

    args = parser.parse_args()

    # args.n_gpu = torch.cuda.device_count()
    current_run = get_random_name()

    # sample_folder = sample_folder.format(args.checkpoint_suffix)
    args.sample_folder = osp.join(BASE, args.sample_folder + '_' + current_run)
    os.makedirs(args.sample_folder, exist_ok=True)

    args.checkpoint_dir = osp.join(BASE, args.checkpoint_dir + '_' + current_run)
    # os.makedirs(args.checkpoint_dir, exist_ok=True)

    # checkpoint_dir = checkpoint_dir.format(args.checkpoint_suffix)

    print(args, flush=True)

    print(f'Weight configuration used, latent loss : {LATENT_LOSS_WEIGHT}, image disc weight : {image_disc_weight}, video disc weight : {video_disc_weight}')

    # print(f'Weight configuration used : + \
    #         {LATENT_LOSS_WEIGHT}, 2d disc weight : {G_LOSS_2D_WEIGHT}, + \
    #         temporal disc weight : {G_LOSS_3D_WEIGHT}')

    # dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
    main(args)