# This model uses the temporal discriminator based on mocogan-hd

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
from config import DATASET, LATENT_LOSS_WEIGHT, DISC_LOSS_WEIGHT, SAMPLE_SIZE_FOR_VISUALIZATION

criterion = nn.MSELoss()

bce_loss = nn.BCEWithLogitsLoss()

global_step = 0

sample_size = SAMPLE_SIZE_FOR_VISUALIZATION

dataset = DATASET

CONST_FRAMES_TO_CHECK = 16

BASE = '/ssd_scratch/cvit/aditya1/video_vqvae2_results'
# sample_folder = '/home2/bipasha31/python_scripts/CurrentWork/samples/{}'

# checkpoint_dir = 'checkpoint_{}'

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
            source_images, input, prediction = run_step(model, data, device, run='val')
            
        source_hulls = input[:, :3]
        background = input[:, 3:]

        saves = {
            'source': source_hulls,
            'background': background,
            'prediction': prediction,
            'source_images': source_images
        }

        if i % (len(val_loader) // 10) == 0 or run_type != 'train':
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
            sample, _, out = run_step(model, data, device, 'val')

        # print(f'sample shape : {sample.shape}, out shape : {out.shape}')
        # reshape the sample and the output
        # sample = sample.permute(0, 2, 3, 1)
        # out = out.permute(0, 2, 3, 1)

        # if sample.shape[1] != 3:
        #     sample = get_proper_shape(sample[:sample_size])
        #     out = get_proper_shape(out[:sample_size])

        # if i % (len(val_loader) // 10) == 0 or run_type != 'train':
        #     def denormalize(x):
        #         return (x.clamp(min=-1.0, max=1.0) + 1)/2

        #     for name in saves:
        #         saveas = f"{sample_folder}/{epoch + 1}_{global_step}_{i}_{name}.mp4"
        #         frames = saves[name].detach().cpu()
        #         frames = [denormalize(x).permute(1, 2, 0).numpy() for x in frames]

        #         # os.makedirs(sample_folder, exist_ok=True)
        #         save_frames_as_video(frames, saveas, fps=25)


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
    if dataset == 6:
        jitter_validation(model, val_loader, device, epoch, i, run_type, sample_folder)
    else:
        base_validation(model, val_loader, device, epoch, i, run_type, sample_folder)

        
def flip_video(x):
    num = random.randint(0, 1)
    if num == 0:
        return torch.flip(x, [2])
    else:
        return x

# insidie the disc step, there are two discs - i) image disc, ii) motion disc
def disc_step(x_fake, x, modelD_3d, modelD_img):
    # x_fake -> batch_size x num_frames x channels x img_dim x img_dim
    kernel_size = 1
    n_frames_G = 1
    x_fake = F.avg_pool3d(x_fake, (1, kernel_size, kernel_size))

    # concatenating first frame with every other frame for real images
    # dimension -> batch_size x num_frames-1 x channels x size x size
    x_in = torch.cat((x[:, 0].unsqueeze(1).repeat(1, n_frames_G - 1, 1, 1, 1), x[:, 1:]), dim=2)

    # concatenating first frame with every frame for fake images
    x_fake_in = torch.cat((x_fake[:, 0].unsqueeze(1).repeat(1, n_frames_G - 1, 1, 1, 1), x_fake[:, 1:]), dim=2)

    # temporal discriminator's predictions on fake images 
    # detach() because generator training is not required during discriminator
    D_fake_3d = modelD_3d(flip_video(torch.transpose(x_fake_in, 1, 2).detach()))

    # temporal discriminator's predictions on the real images
    D_real_3d = modelD_3d(flip_video(torch.transpose(x_in, 1, 2)))

    # criterionGAN = losses.Relativistic_Average_LSGAN()
    criterionGAN = mocoganhd_losses.Relativistic_Average_LSGAN()

    # compute the real and fake temporal discriminator losses
    D_loss_real_3d = criterionGAN(D_real_3d, D_fake_3d, True)
    D_loss_fake_3d = criterionGAN(D_fake_3d, D_real_3d, False)

    # disc loss is the combination of the real and fake losses
    D_loss_3d = (D_loss_real_3d + D_loss_fake_3d) * 0.5 

    print(f'3D discriminator loss : {D_loss_3d}')

    # skipping computing gradient penalty for now
    # loss_GP_3d = mocoganhd_losses.compute_gradient_penalty_T(
    #     torch.transpose(x_in, 1, 2), torch.transpose(x_fake_in, 1, 2),
    #     modelD_3d, opt)

    # setting the gradients to 0 and performing the backward step by updating the gradients
    modelD_3d.module.optim.zero_grad()
    D_loss_3d.backward(retain_graph=True)
    modelD_3d.module.optim.step()

    # ------ image based discriminator losses
    # take a random frame id and compute the image disc loss between real-fake and real-real 
    frame_id = random.randint(1, n_frames_G - 1)

    # image based predictions between real frames
    D_real = modelD_img(torch.cat((x[:, 0], x[:, frame_id]), dim=1).detach())

    D_fake = modelD_img(torch.cat((x_fake[:, 0], x_fake[:, frame_id]), dim=1).detach())

    # compute the real and the fake losses
    D_loss_real = criterionGAN(D_real, D_fake, True)
    D_loss_fake = criterionGAN(D_fake, D_real, False)

    # compute the image disc loss
    D_loss = (D_loss_real + D_loss_fake) * 0.5 

    modelD_img.module.optim.zero_grad()
    D_loss.backward()
    modelD_img.module.optim.step()

    # what we have is D_loss_real, D_loss_fake, D_loss_real_3d, D_loss_fake_3d
    return D_loss_real.item(), D_loss_fake.item(), D_loss_real_3d.item(), D_loss_fake_3d.item() 

# this is the generator step 
# takes in the temporal and image discriminator
def gen_step(x, x_fake, modelD_3d, modelD_img, G_optim):
    n_frames_G = 16
    # x.shape -> batch_size x n_frames x channels x img_dim x img_dim
    kernel_size = 1
    x_fake = F.avg_pool3d(x_fake, (1, kernel_size, kernel_size))
    
    # compute the discriminator's prediction on the real images and the images generated by the generator
    frame_id = random.randint(1, n_frames_G - 1)
    D_fake = modelD_img(torch.cat((x_fake[:, 0], x_fake[:, frame_id]), dim=1))
    D_real = modelD_img(torch.cat((x[:, 0], x[:, frame_id]), dim=1))

    criterionGAN = mocoganhd_losses.Relativistic_Average_LSGAN()

    # the image discriminator is trained in a constrastive loss setup 

    # compute the 2d loss 
    G_loss_2d = (criterionGAN(D_fake, D_real, True) + criterionGAN(D_real, D_fake,False)) * 0.5

    # repeat the 0th input frame a num_frames number of times to create x_in 
    x_in = torch.cat((x[:, 0].unsqueeze(1).repeat(1, n_frames_G - 1, 1, 1, 1), x[:, 1:]), dim=2)

    # repeat the 0th fake frame a num_frames number of times to create x_fake_in
    x_fake_in = torch.cat((x_fake[:, 0].unsqueeze(1).repeat(1, n_frames_G - 1, 1, 1, 1), x_fake[:, 1:]), dim=2)

    # compute the temporal model loss 
    D_real_3d = modelD_3d(flip_video(torch.transpose(x_in, 1, 2)))
    D_fake_3d = modelD_3d(flip_video(torch.transpose(x_fake_in, 1, 2)))

    # temporal loss on the generator 
    G_loss_3d = (criterionGAN(D_fake_3d, D_real_3d, True) + criterionGAN(D_real_3d, D_fake_3d, False)) * 0.5

    G_loss = G_loss_3d + G_loss_2d

    # backpropagate, compute the gradients, and update the parameters 
    # modelG.module.modelR.optim.zero_grad()
    # G_loss.backward()
    # modelG.module.modelR.optim.step()

    # optimizer is defined separately that optimizes the weights of the generator
    G_optim.zero_grad()
    G_loss.backward()
    G_optim.step()

    # the parameters of the vqvae2 model needs to be updated at this step 

    return G_loss_2d.item(), G_loss_3d.item()

# one generator discriminator step needs to be carried out
# G_optim is the optimizer for the vqvae model
def GD_step(modelG, modelD_img, modelD_3d, G_optim, x, x_fake):
    # carries one step of the generator discriminator setup 
    G_loss, G_loss_3d = gen_step(x_fake, x, modelD_3d, modelD_img, G_optim)

    D_loss_real, D_loss_fake, D_loss_3d_real, D_loss_3d_fake = disc_step(x_fake, x, modelD_3d, modelD_img)

    # these are all the losses

# training is done in two steps
# modelG would be the vqvae2 model used for generating the imags
def train(model, loader, val_loader, scheduler, device, 
            epoch, validate_at, checkpoint_dir, sample_folder,
            modelD_img, modelD_3d, gen_optim):
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    mse_sum = 0
    mse_n = 0
    disc_sum = 0
    disc_n = 0

    SAMPLE_FRAMES = 16 # sample 16 frames for the discriminator

    global global_step

    for i, data in enumerate(loader):
        # run the predictions using the model 
        # prediction.shape -> batch_size x num_frames x channels x img_dim x img_dim
        # prediction, ground_truth = run_step_custom(model, data, device)
        # predictions from the vqvae2 network (generator)
        recon_loss, latent_loss, S, out, ground_truth = run_step(model, data, device)

        print(out.shape, ground_truth.shape)
        num_frames = len(out)

        # add the batch dimension 
        x_fake, x = out.unsqueeze(0), ground_truth.unsqueeze(0)

        # generate a random idx 
        random_idx = random.randint(0, num_frames - SAMPLE_FRAMES)

        x_fake = x_fake[:, random_idx:random_idx+SAMPLE_FRAMES]
        x = x[:, random_idx:random_idx+SAMPLE_FRAMES]

        # additionally the generator can be optimized for a few more number of steps
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
            D_real = modelD_img(torch.cat((x[:, 0], x[:, frame_id]), dim=1).detach())

            criterionGAN = mocoganhd_losses.Relativistic_Average_LSGAN()

            # compute the 2d image loss 
            G_loss_2d = (criterionGAN(D_fake, D_real, True) + 
                        criterionGAN(D_real, D_fake, False)) * 0.5
            
            # pair 0th frame with every other frame in the input
            x_in = torch.cat((x[:, 0].unsqueeze(1).repeat(1, SAMPLE_FRAMES-1 - 1, 1, 1, 1), x[:, 1:]), dim=2)
            x_fake_in = torch.cat((x_fake[:, 0].unsqueeze(1).repeat(1, SAMPLE_FRAMES-1, 1, 1, 1), x_fake[:, 1:]), dim=2)

            # temporal discriminator predictions 
            D_real_3d = modelD_3d(flip_video(torch.transpose(x_in, 1, 2)))
            D_fake_3d = modelD_3d(flip_video(torch.transpose(x_fake_in, 1, 2)))

            # compute the 3d loss (temporal disc loss)
            # only the gradients of the gen are computed and params updated
            G_loss_3d = (criterionGAN(D_fake_3d, D_real_3d, True) + 
                        criterionGAN(D_real_3d, D_fake_3d, False)) * 0.5

            G_loss = recon_loss + latent_loss + G_loss_2d + G_loss_3d

            # backpropagate and compute the gradients for the generator
            model.zero_grad()
            G_loss.backward()

            if scheduler is not None:
                scheduler.step()

            gen_optim.step()

        else: # disc step 
            kernel_size = 1
            x_fake = F.avg_pool3d(x_fake, (1, kernel_size, kernel_size))

            kernel_size = 1
            x_fake = F.avg_pool3d(x_fake, (1, kernel_size, kernel_size))

            # concatenate the 0th frame with the other frames in the input
            x_in = torch.cat((x[:, 0].unsqueeze(1).repeat(1, SAMPLE_FRAMES - 1, 1, 1, 1), x[:, 1:]), dim=2)
            x_fake_in = torch.cat((x_fake[:, 0].unsqueeze(1).repeat(1, SAMPLE_FRAMES - 1, 1, 1, 1), x_fake[:, 1:]), dim=2)

            # compute the temporal disc predictions on real and fake 
            # detach required because the generator need not be trained in this step
            D_fake_3d = modelD_3d(flip_video(torch.transpose(x_fake_in, 1, 2).detach()))
            D_real_3d = modelD_3d(flip_video(torch.transpose(x_in, 1, 2).detach()))

            criterionGAN = mocoganhd_losses.Relativistic_Average_LSGAN()
            D_loss_real_3d = criterionGAN(D_real_3d, D_fake_3d, True)
            D_loss_fake_3d = criterionGAN(D_fake_3d, D_real_3d, False)

            # compute the temporal disc loss
            D_loss_3d = (D_loss_real_3d + D_loss_fake_3d) * 0.5

            modelD_3d.module.optim.zero_grad()
            D_loss_3d.backward(retain_graph=True)
            modelD_3d.module.optim.step()

            # -- image disc loss for the discriminator
            # sample a random frame_id
            frame_id = random.randint(1, SAMPLE_FRAMES-1)

            # discriminator's predictions on real and fake data
            D_real = modelD_img(torch.cat((x[:, 0], x[:, frame_id]), dim=1).detach())
            D_fake = modelD_img(torch.cat((x_fake[:, 0], x_fake[:, frame_id]), dim=1).detach())

            # disc losses for fake and real images 
            D_loss_real = criterionGAN(D_real, D_fake, True)
            D_loss_fake = criterionGAN(D_fake, D_real, False)

            # disc image loss
            D_loss = (D_loss_real + D_loss_fake) * 0.5

            # backpropagate and compute the gradients and optimize the disc
            modelD_img.module.optim.zero_grad()
            D_loss.backward()
            modelD_img.module.optim.step()

def train1(model, disc, loader, val_loader, optimizer, disc_optimizer, scheduler, device, epoch, validate_at, checkpoint_dir, sample_folder):
    if dist.is_primary():
        loader = tqdm(loader, file=sys.stdout)

    mse_sum = 0
    mse_n = 0
    disc_sum = 0
    disc_n = 0

    global global_step

    for i, data in enumerate(loader):

        # disc step
        if global_step%2:
            model.zero_grad()
            disc.zero_grad()

            # generator step 
            prediction, ground_truth = run_step_custom(model, data, device)

            # generate the disc predictions 
            # print(f'Ground truth shape : {ground_truth.shape}, prediction shape : {prediction.shape}')

            # test the flow for any random sequence of frames
            random_index = random.randint(0, ground_truth.shape[0] - CONST_FRAMES_TO_CHECK - 1)

            ground_truth = ground_truth[random_index : random_index + CONST_FRAMES_TO_CHECK].unsqueeze(0).permute(0, 2, 1, 3, 4)
            prediction = prediction[random_index : random_index + CONST_FRAMES_TO_CHECK].unsqueeze(0).permute(0, 2, 1, 3, 4)

            disc_real_pred = disc(ground_truth)
            disc_fake_pred = disc(prediction.detach()) 

            disc_real_loss = bce_loss(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_fake_loss = bce_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            # backprop the disc loss
            disc_loss.backward()
            disc_optimizer.step()

            # print(f'Disc real prediction : {disc_real_pred.shape}, disc fake : {disc_fake_pred.shape}')
            part_disc_sum = disc_loss.item() * S
            part_disc_n = S

            comm = {"disc_sum": part_disc_sum, "disc_n": part_disc_n}
            comm = dist.all_gather(comm)

            for part in comm:
                disc_sum += part["disc_sum"]
                disc_n += part["disc_n"]

            if dist.is_primary():
                lr = optimizer.param_groups[0]["lr"]

                loader.set_description(
                    (
                        f"epoch: {epoch + 1}; disc step; disc loss: {disc_loss.item():.5f}; "
                        f"avg disc loss: {disc_sum / disc_n:.5f}; "
                        f"lr: {lr:.5f}"
                    )
                )
            
        # gen step
        else:
            model.zero_grad()
            disc.zero_grad()

            # train the generator 
            recon_loss, latent_loss, S, prediction, ground_truth = \
                        run_step(model, data, device)

            # generate the discriminator predictions on the generator output
            random_index = random.randint(0, ground_truth.shape[0] - CONST_FRAMES_TO_CHECK - 1)

            prediction = prediction[random_index : random_index + CONST_FRAMES_TO_CHECK].unsqueeze(0).permute(0, 2, 1, 3, 4)

            disc_fake_pred = disc(prediction)

            disc_fake_loss = bce_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))

            gen_loss = recon_loss + LATENT_LOSS_WEIGHT * latent_loss + DISC_LOSS_WEIGHT * disc_fake_loss

            gen_loss.backward()

            if scheduler is not None:
                scheduler.step()

            optimizer.step()

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
                        f"epoch: {epoch + 1}; gen step; mse: {recon_loss.item():.5f}; "
                        f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                        f"lr: {lr:.5f}"
                    )
                )

        global_step += 1

        if i % validate_at == 0:
            model.eval()

            validation(model, val_loader, device, epoch, i, sample_folder)

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

    # loader, val_loader, model, disc = get_loaders_and_models(
    #     args, dataset, default_transform, device, test=args.test)
    # train_loader, val_loader, model, vqlpips = get_loaders_and_models(
    #     args, dataset, default_transform, device, test=args.test)
    train_loader, val_loader, model, modelD_img, modelD_3d = \
        get_loaders_and_models(args, dataset, default_transform, device, test=args.test)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

        modelD_img = nn.parallel.DistributedDataParallel(
            modelD_img,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

        modelD_3d = nn.parallel.DistributedDataParallel(
            modelD_3d,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        state_dict = { k.replace('module.', ''): v for k, v in state_dict.items() }  
        try:
            model.module.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict)

    if args.test:
        # test(loader, model, device)
        validation(model, val_loader, device, 0, 0, args.sample_folder, 'val')
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # disc_optimizer = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999),
        #                                      weight_decay=0.00001)
        
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
            # train(model, disc, train_loader, val_loader, optimizer, disc_optimizer, scheduler, device, i, args.validate_at, args.checkpoint_dir, args.sample_folder)
            # train(model, train_loader, val_loader, scheduler, device, i, args.validate_at, args.checkpoint_dir, args.sample_folder)
            
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

    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--checkpoint_suffix", type=str, default='')
    parser.add_argument("--validate_at", type=int, default=512)
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

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
