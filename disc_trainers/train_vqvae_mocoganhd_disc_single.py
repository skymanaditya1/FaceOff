# This is the code that has been modified for running on a single GPU 

import argparse
import sys
import os
import random
import os.path as osp

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms, utils

from TemporalAlignment.models import mocoganhd_losses

from tqdm import tqdm

from scheduler import CycleScheduler
import distributed as dist

from utils import *
from config import DATASET, LATENT_LOSS_WEIGHT, G_LOSS_2D_WEIGHT, G_LOSS_3D_WEIGHT, SAMPLE_SIZE_FOR_VISUALIZATION

criterion = nn.MSELoss()

bce_loss = nn.BCEWithLogitsLoss()

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

def blob2full_validation(model, img):
    face, rhand, lhand = img

    face = face[:sample_size] 
    rhand = rhand[:sample_size]
    lhand = lhand[:sample_size]
    sample = face, rhand, lhand

    gt = gt[:sample_size]

    with torch.no_grad():
        out, _ = model(sample)
    
    save_image(torch.cat([face, rhand, lhand, out, gt], 0), 
        f"sample/{epoch + 1}_{i}.png")

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
                
            # save_image(
            #     torch.cat([sample[:3*3], out[:3*3]], 0), 
            #     f"{sample_folder}/{epoch + 1}_{i}_{val_i}.png")

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

# training is done in two steps
# modelG would be the vqvae2 model used for generating the imags
def train(model, loader, val_loader, scheduler, device, 
            epoch, validate_at, checkpoint_dir, sample_folder,
            modelD_img, modelD_3d, gen_optim):

    SAMPLE_FRAMES = 16 # sample 16 frames for the discriminator

    for i, data in enumerate(loader):
        global global_step
        global_step += 1
        
        recon_loss, latent_loss, S, out, ground_truth = run_step(model, data, device)

        # print(out.shape, ground_truth.shape)
        num_frames = len(out)

        # add the batch dimension 
        x_fake, x = out.unsqueeze(0), ground_truth.unsqueeze(0)

        # generate a random idx 
        # check if there are atleast SAMPLE_FRAMES in the input
        if num_frames < SAMPLE_FRAMES:
            print(f'Frames found {num_frames} less than minimum {SAMPLE_FRAMES}')
            continue

        random_idx = random.randint(0, num_frames - SAMPLE_FRAMES)

        x_fake = x_fake[:, random_idx:random_idx+SAMPLE_FRAMES]
        x = x[:, random_idx:random_idx+SAMPLE_FRAMES]

        # additionally the generator can be optimized for a few more number of steps than the discriminator
        if i%2 == 0:
            step = 'gen'
        else:
            step = 'disc'

        # alternate between the generator and discriminator 
        if step == 'gen':
            kernel_size = 1
            x_fake = F.avg_pool3d(x_fake, (1, kernel_size, kernel_size))

            # sample a random frame for contrastive loss
            frame_id = random.randint(1, SAMPLE_FRAMES-1)

            # generate discriminator predictions on the fake images, img disc
            D_fake = modelD_img(torch.cat((x_fake[:, 0], x_fake[:, frame_id]), dim=1))

            # generate discriminator predictions on the real images
            # we don't want generator to be trained for disc's predictions on real images
            D_real = modelD_img(torch.cat((x[:, 0], x[:, frame_id]), dim=1).detach())

            criterionGAN = mocoganhd_losses.Relativistic_Average_LSGAN()

            # compute the 2d image loss 
            G_loss_2d = (criterionGAN(D_fake, D_real, True) + 
                        criterionGAN(D_real, D_fake, False)) * 0.5
            
            # pair 0th frame with every other frame in the input

            x_in = torch.cat((x[:, 0].unsqueeze(1).repeat(1, SAMPLE_FRAMES-1, 1, 1, 1), x[:, 1:]), dim=2)
            x_fake_in = torch.cat((x_fake[:, 0].unsqueeze(1).repeat(1, SAMPLE_FRAMES-1, 1, 1, 1), x_fake[:, 1:]), dim=2)

            # temporal discriminator predictions 
            D_real_3d = modelD_3d(flip_video(torch.transpose(x_in, 1, 2)))
            D_fake_3d = modelD_3d(flip_video(torch.transpose(x_fake_in, 1, 2)))

            # compute the 3d loss (temporal disc loss)
            # only the gradients of the gen are computed and params updated
            G_loss_3d = (criterionGAN(D_fake_3d, D_real_3d, True) + 
                        criterionGAN(D_real_3d, D_fake_3d, False)) * 0.5

            G_loss = recon_loss \
                    + LATENT_LOSS_WEIGHT * latent_loss \
                    + G_LOSS_2D_WEIGHT * G_loss_2d \
                    + G_LOSS_3D_WEIGHT * G_loss_3d

            # backpropagate and compute the gradients for the generator
            model.zero_grad()
            # zero out the gradients of the image and temporal discriminator
            modelD_3d.optim.zero_grad()
            modelD_img.optim.zero_grad()

            G_loss.backward()

            if scheduler is not None:
                scheduler.step()

            gen_optim.step()

        else: # disc step 
            kernel_size = 1
            x_fake = F.avg_pool3d(x_fake, (1, kernel_size, kernel_size))

            # concatenate the 0th frame with the other frames in the input
            x_in = torch.cat((x[:, 0].unsqueeze(1).repeat(1, SAMPLE_FRAMES - 1, 1, 1, 1), x[:, 1:]), dim=2)
            x_fake_in = torch.cat((x_fake[:, 0].unsqueeze(1).repeat(1, SAMPLE_FRAMES - 1, 1, 1, 1), x_fake[:, 1:]), dim=2)

            # compute the temporal disc predictions on real and fake 
            # detach required because the generator need not be trained in this step
            D_fake_3d = modelD_3d(flip_video(torch.transpose(x_fake_in, 1, 2).detach()))
            D_real_3d = modelD_3d(flip_video(torch.transpose(x_in, 1, 2)))

            criterionGAN = mocoganhd_losses.Relativistic_Average_LSGAN()
            D_loss_real_3d = criterionGAN(D_real_3d, D_fake_3d, True)
            D_loss_fake_3d = criterionGAN(D_fake_3d, D_real_3d, False)

            # compute the temporal disc loss
            D_loss_3d = (D_loss_real_3d + D_loss_fake_3d) * 0.5

            # trying to access the models using the module in distributed training setup
            # modelD_3d.module.optim.zero_grad()
            # D_loss_3d.backward(retain_graph=True)
            # modelD_3d.module.optim.step()

            # retain graph is required for generating disc's predictions on real and fake data
            modelD_3d.optim.zero_grad()
            D_loss_3d.backward(retain_graph=True)
            modelD_3d.optim.step()

            # -- image disc loss for the discriminator
            # sample a random frame_id
            frame_id = random.randint(1, SAMPLE_FRAMES-1)

            # discriminator's predictions on real and fake data
            D_real = modelD_img(torch.cat((x[:, 0], x[:, frame_id]), dim=1).detach())
            D_fake = modelD_img(torch.cat((x_fake[:, 0], x_fake[:, frame_id]), dim=1).detach())

            # disc losses for fake and real images 
            D_loss_real = criterionGAN(D_real, D_fake, True)
            D_loss_fake = criterionGAN(D_fake, D_real, False)

            # disc image loss - modification required at a later stage
            D_loss = (D_loss_real + D_loss_fake) * 0.5

            # backpropagate and compute the gradients and optimize the disc
            # modelD_img.module.optim.zero_grad()
            # D_loss.backward()
            # modelD_img.module.optim.step()

            modelD_img.optim.zero_grad()
            D_loss.backward()
            modelD_img.optim.step()

        lr = gen_optim.param_groups[0]["lr"]
        # loader.set_description(
        #     (
        #         f"epoch: {epoch + 1}; gen loss : {G_loss.item():.5f}; disc loss: {D_loss.item():.5f}; "
        #                 f"lr: {lr:.5f}"
        #     )
        # )

        # indicates that both the generator and discriminator steps would have been performed
        if (i+1)%2 == 0:
            print(f"epoch: {epoch + 1}; gen loss : {G_loss.item():.5f}; disc loss: {D_loss.item():.5f}; lr: {lr:.5f}")

        # check if validation is required 
        if i%validate_at == 0:
            # set the model to eval and generate the predictions
            model.eval()

            validation(model, val_loader, device, epoch, i, sample_folder)

            os.makedirs(checkpoint_dir, exist_ok=True)

            # save the vqvae2 generator weights 
            torch.save(model.state_dict(), f"{checkpoint_dir}/vqvae_{epoch+1}_{str(i + 1).zfill(4)}.pt")

            # save the discriminator weights
            # save the temporal disc weights
            torch.save(modelD_3d.state_dict(), f"{checkpoint_dir}/modelD_3d_{epoch+1}_{str(i+1).zfill(4)}.pt")

            # save the image/content disc weights  
            torch.save(modelD_img.state_dict(), f"{checkpoint_dir}/modelD_img_{epoch+1}_{str(i+1).zfill(4)}.pt")

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

    train_loader, val_loader, model, modelD_img, modelD_3d = \
        get_loaders_and_models(args, dataset, default_transform, device, test=args.test)


    # load the models on the gpu 
    model = model.to(device)
    modelD_img = modelD_img.to(device)
    modelD_3d = modelD_3d.to(device)

    # loading the pretrained generator weights
    if args.ckpt:
        print(f'Loading pretrained generator model : {args.ckpt}')
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        try:
            model.module.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict)

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
            
            train(model, train_loader, val_loader, scheduler, device, i, 
                args.validate_at, args.checkpoint_dir, args.sample_folder,
                modelD_img, modelD_3d, optimizer)


def get_random_name(cipher_length=5):
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join([chars[random.randint(0, len(chars)-1)] for i in range(cipher_length)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    # port = (
    #     2 ** 15
    #     + 2 ** 14
    #     + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    # )

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

    print(f'Weight configuration used : + \
            {LATENT_LOSS_WEIGHT}, 2d disc weight : {G_LOSS_2D_WEIGHT}, + \
            temporal disc weight : {G_LOSS_3D_WEIGHT}')

    # dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
    main(args)