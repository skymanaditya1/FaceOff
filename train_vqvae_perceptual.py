import argparse
import sys
import os
import random
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from scheduler import CycleScheduler
import distributed as dist

from utils import *
from config import DATASET, LATENT_LOSS_WEIGHT, PERCEPTUAL_LOSS_WEIGHT, SAMPLE_SIZE_FOR_VISUALIZATION


criterion = nn.MSELoss()

global_step = 0

sample_size = SAMPLE_SIZE_FOR_VISUALIZATION

dataset = DATASET

BASE = '/ssd_scratch/cvit/aditya1/video_vqvae2_results'


def run_step(model, vqlpips, data, device, run='train'):
    img, S, ground_truth, source_image = process_data(data, device, dataset)

    out, latent_loss = model(img)

    out = out[:, :3]
    
    recon_loss = criterion(out, ground_truth)
    latent_loss = latent_loss.mean()

    perceptual_loss = vqlpips(ground_truth, out)
    
    if run == 'train':
        return recon_loss, latent_loss, perceptual_loss, S
    else:
        return ground_truth, img, out, source_image


'''
runs the validation step, saves the blended output video
'''
def validation(model, vqlpips, val_loader, device, epoch, i, sample_folder, run_type='train'):
    print(f'Inside jitter validation')
    for i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            source_images, input, prediction, source_images_original = run_step(model, vqlpips, data, device, run='val')
            
        source_hulls = input[:, :3]
        background = input[:, 3:]

        saves = {
            'source': source_hulls,
            'background': background,
            'prediction': prediction,
            'source_images': source_images,
            'source_original': source_images_original,
        }

        if True:
            def denormalize(x):
                return (x.clamp(min=-1.0, max=1.0) + 1)/2

            for name in saves:
                saveas = f"{sample_folder}/{epoch + 1}_{global_step}_{i}_{name}.mp4"
                frames = saves[name].detach().cpu()
                frames = [denormalize(x).permute(1, 2, 0).numpy() for x in frames]

                save_frames_as_video(frames, saveas, fps=25)

'''
trainer code -- loss is computed as the weighted sum of reconstruction loss, latent loss, and perceptual loss
'''
def train(model, vqlpips, loader, val_loader, optimizer, scheduler, device, epoch, validate_at, checkpoint_dir, sample_folder):
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    mse_sum = 0
    mse_n = 0
    perceptual_losses = []

    for i, data in enumerate(loader):
        model.zero_grad()

        recon_loss, latent_loss, perceptual_loss, S = run_step(model, vqlpips, data, device)

        # loss is a weighted sum of i) reconstruction loss, ii) latent loss, and iii) perceptual loss
        loss = recon_loss + LATENT_LOSS_WEIGHT * latent_loss + PERCEPTUAL_LOSS_WEIGHT * perceptual_loss

        loss.backward()

        perceptual_losses.append(perceptual_loss.item())

        if scheduler is not None:
            scheduler.step()

        optimizer.step()

        global global_step

        global_step += 1

        part_mse_sum = recon_loss.item() * S
        part_mse_n = S

        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"perceptual: {np.array(perceptual_losses).mean():.3f} "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

        if i % validate_at == 0:
            model.eval()

            validation(model, vqlpips, val_loader, device, epoch, i, sample_folder)

            if dist.is_primary():
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), f"{checkpoint_dir}/vqvae_{epoch+1}_{str(i + 1).zfill(4)}.pt")

            model.train()

def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    default_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    loader, val_loader, model, vqlpips = get_loaders_and_models(
        args, device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

        vqlpips = nn.parallel.DistributedDataParallel(
            vqlpips,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    # load the pretrained checkpoints if available
    if args.ckpt:
        print(f'Loading pretrained checkpoint - {args.ckpt}')
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        try:
            model.module.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict)

    if args.test:
        validation(model, vqlpips, val_loader, device, 0, 0, args.sample_folder, 'val')
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        scheduler = None
        
        if args.sched == "cycle":
            scheduler = CycleScheduler(
                optimizer,
                args.lr,
                n_iter=len(loader) * args.epoch,
                momentum=None,
                warmup_proportion=0.05,
            )

        for i in range(args.epoch):
            train(model, vqlpips, loader, val_loader, optimizer, scheduler, device, i, args.validate_at, args.checkpoint_dir, args.sample_folder)

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

    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, default=256, help='image resolution')
    parser.add_argument("--epoch", type=int, default=560, help='total epochs')
    parser.add_argument("--lr", type=float, default=3e-4, help='learning rate')
    parser.add_argument("--sched", type=str)
    parser.add_argument("--checkpoint_suffix", type=str, default='')
    parser.add_argument("--validate_at", type=int, default=1024, help='validate after (number of steps)')
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--test", action='store_true', required=False)
    parser.add_argument("--gray", action='store_true', required=False)
    parser.add_argument("--colorjit", type=str, default='', help='const or random or empty')
    parser.add_argument("--crossid", action='store_true', required=False)
    parser.add_argument("--custom_validation", action='store_true', required=False, help='perform custom validation')
    parser.add_argument("--sample_folder", type=str, default='samples')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoint', help='dir path for saving the checkpoints')
    parser.add_argument("--validation_folder", type=str, default=None, help='dir path for saving the validated samples')

    args = parser.parse_args()

    # args.n_gpu = torch.cuda.device_count()
    current_run = get_random_name()

    args.sample_folder = osp.join(BASE, args.sample_folder + '_' + current_run)
    os.makedirs(args.sample_folder, exist_ok=True)

    args.checkpoint_dir = osp.join(BASE, args.checkpoint_dir + '_' + current_run)

    print(args, flush=True)

    print(f'Weight configuration used - perceptual loss weight : {PERCEPTUAL_LOSS_WEIGHT}, latent loss weight : {LATENT_LOSS_WEIGHT}')

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))