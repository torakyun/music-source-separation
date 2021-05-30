# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import tqdm
#########################
import torch
#########################
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import apply_model, average_metric, center_trim

############################################################


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def forward(netG, real_A):
    """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    fake_B = netG(real_A)  # G(A)
    return fake_B


def backward_D(netD, input_D, real_A, real_B, fake_B, device, criterionGAN):
    """Calculate GAN loss for the discriminator"""
    real_A = center_trim(real_A, fake_B)
    real_B = center_trim(real_B, fake_B)
    if 'outputs' in input_D:
        real_B = real_B.view(real_B.size(0), real_B.size(
            1) * real_B.size(2), real_B.size(-1))
        fake_B = fake_B.view(fake_B.size(0), fake_B.size(
            1) * fake_B.size(2), fake_B.size(-1))
        if input_D == 'outputs+mix':
            # Fake; stop backprop to the generator by detaching fake_B
            # we use conditional GANs; we need to feed both input and output to the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB.detach())
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
        elif input_D == 'outputs':
            pred_fake = netD(fake_B.detach())
            pred_real = netD(real_B)
        loss_D_fake = criterionGAN(pred_fake, False)
        loss_D_real = criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        return {'D_real': loss_D_real.item(), 'D_fake': loss_D_fake.item()}
    else:
        log = {}
        loss_D = 0
        for i, key in enumerate(['drums', 'bass', 'other', 'vocals']):
            real_B_i = real_B[:, i, :, :]
            fake_B_i = fake_B[:, i, :, :]
            if input_D == 'output+mix':
                # Fake; stop backprop to the generator by detaching fake_B
                label = torch.eye(4, device=device)[i].view(1, 4, 1)
                label = label.expand(real_A.size(0), 4, real_A.size(-1))
                # we use conditional GANs; we need to feed both input and output to the discriminator
                fake_AB = torch.cat((real_A, fake_B_i, label), 1)
                pred_fake = netD(fake_AB.detach())
                # Real
                real_AB = torch.cat((real_A, real_B_i, label), 1)
                pred_real = netD(real_AB)
            elif input_D == 'output':
                label = torch.eye(4, device=device)[i].view(1, 4, 1)
                label = label.expand(real_A.size(0), 4, real_A.size(-1))
                fake_B_i = torch.cat((fake_B_i, label), 1)
                real_B_i = torch.cat((real_B_i, label), 1)
                pred_fake = netD(fake_B_i.detach())
                pred_real = netD(real_B_i)
            elif input_D == 'output+mix(separated)':
                real_AB = torch.cat((real_A, real_B_i), 1)
                pred_real = netD[key](real_AB)
                fake_AB = torch.cat((real_A, fake_B_i), 1)
                pred_fake = netD[key](fake_AB.detach())
            elif input_D == 'output(separated)':
                pred_real = netD[key](real_B_i)
                pred_fake = netD[key](fake_B_i.detach())
            loss_D_real = criterionGAN(pred_real, True)
            log['D_'+key+'_real'] = loss_D_real.item()
            loss_D_fake = criterionGAN(pred_fake, False)
            log['D_'+key+'_fake'] = loss_D_fake.item()
            if 'separated' not in input_D:
                loss_D += (loss_D_fake + loss_D_real) / 8.
            else:
                loss_D += (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        return log


def backward_G(netD, input_D, real_A, real_B, fake_B, device, criterionGAN, criterionL1, lambda_L1):
    """Calculate GAN and L1 loss for the generator"""
    real_A = center_trim(real_A, fake_B)
    real_B = center_trim(real_B, fake_B)
    if 'outputs' in input_D:
        real_B = real_B.view(real_B.size(0), real_B.size(
            1) * real_B.size(2), real_B.size(-1))
        fake_B = fake_B.view(fake_B.size(0), fake_B.size(
            1) * fake_B.size(2), fake_B.size(-1))
        if input_D == 'outputs+mix':
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = netD(fake_AB)
        elif input_D == 'outputs':
            pred_fake = netD(fake_B)
        loss_G_GAN = criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = criterionL1(fake_B, real_B)
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1 * lambda_L1
        loss_G.backward()
        return {'train': loss_G_L1.item(), 'G': loss_G_GAN.item()}
    else:
        loss_G_L1 = criterionL1(fake_B, real_B)
        log = {'train': loss_G_L1.item()}
        loss_G = loss_G_L1 * lambda_L1
        for i, key in enumerate(['drums', 'bass', 'other', 'vocals']):
            fake_B_i = fake_B[:, i, :, :]
            if input_D == 'output+mix':
                label = torch.eye(4, device=device)[i].view(1, 4, 1)
                label = label.expand(real_A.size(0), 4, real_A.size(-1))
                fake_AB = torch.cat((real_A, fake_B_i, label), 1)
                pred_fake = netD(fake_AB)
            elif input_D == 'output':
                label = torch.eye(4, device=device)[i].view(1, 4, 1)
                label = label.expand(real_A.size(0), 4, real_A.size(-1))
                fake_B_i = torch.cat((fake_B_i, label), 1)
                pred_fake = netD(fake_B_i)
            elif input_D == 'output+mix(separated)':
                fake_AB = torch.cat((real_A, fake_B_i), 1)
                pred_fake = netD[key](fake_AB)
            elif input_D == 'output(separated)':
                pred_fake = netD[key](fake_B_i)
            loss_G_GAN = criterionGAN(pred_fake, True)
            log['G_'+key] = loss_G_GAN.item()
            loss_G += loss_G_GAN / 4.
        loss_G.backward()
        return log


def optimize_parameters(netG, netD, input_D, real_A, real_B, device, criterionGAN, criterionL1, lambda_L1, optimizer_G, optimizer_D):
    fake_B = netG(real_A)                   # compute fake images: G(A)
    if 'separated' not in input_D:
        # update D
        set_requires_grad(netD, True)  # enable backprop for D
        optimizer_D.zero_grad()     # set D's gradients to zero
        # calculate gradients for D
        loss_D = backward_D(netD, input_D, real_A, real_B,
                            fake_B, device, criterionGAN)
        optimizer_D.step()          # update D's weights
        # update G
        # D requires no gradients when optimizing G
        set_requires_grad(netD, False)
        optimizer_G.zero_grad()        # set G's gradients to zero
        loss_G = backward_G(netD, input_D, real_A, real_B, fake_B, device, criterionGAN,
                            criterionL1, lambda_L1)                   # calculate graidents for G
        optimizer_G.step()             # udpate G's weights
        return dict(loss_G, **loss_D)
    else:
        # update D
        for key in netD:
            set_requires_grad(netD[key], True)  # enable backprop for D
            optimizer_D[key].zero_grad()     # set D's gradients to zero
        # calculate gradients for D
        loss_D = backward_D(netD, input_D, real_A, real_B,
                            fake_B, device, criterionGAN)
        for value in optimizer_D.values():
            value.step()          # update D's weights
        # update G
        for key in netD:
            # D requires no gradients when optimizing G
            set_requires_grad(netD[key], False)
            optimizer_G[key].zero_grad()        # set G's gradients to zero
        loss_G = backward_G(netD, input_D, real_A, real_B, fake_B, device, criterionGAN,
                            criterionL1, lambda_L1)                   # calculate graidents for G
        for value in optimizer_G.values():
            value.step()             # udpate G's weights
        return dict(loss_G, **loss_D)
############################################################


def train_model(epoch,
                dataset,
                model,
                criterion,
                optimizer,
                augment,
                use_gan=False,
                netD=None,
                input_D=None,
                criterionGAN=None,
                lambda_L1=None,
                optimizer_D=None,
                repeat=1,
                device="cpu",
                seed=None,
                workers=4,
                world_size=1,
                batch_size=16):

    if world_size > 1:
        sampler = DistributedSampler(dataset)
        sampler_epoch = epoch * repeat
        if seed is not None:
            sampler_epoch += seed * 1000
        sampler.set_epoch(sampler_epoch)
        batch_size //= world_size
        loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=workers, shuffle=True)
    current_loss = {'train': 0}
    #######################
    if use_gan:
        if 'separated' not in input_D:
            current_loss['G'] = 0
            current_loss['D_real'] = 0
            current_loss['D_fake'] = 0
        else:
            current_loss['G_drums'] = 0
            current_loss['G_bass'] = 0
            current_loss['G_other'] = 0
            current_loss['G_vocals'] = 0
            current_loss['D_drums_real'] = 0
            current_loss['D_drums_fake'] = 0
            current_loss['D_bass_real'] = 0
            current_loss['D_bass_fake'] = 0
            current_loss['D_other_real'] = 0
            current_loss['D_other_fake'] = 0
            current_loss['D_vocals_real'] = 0
            current_loss['D_vocals_fake'] = 0
    #######################
    for repetition in range(repeat):
        tq = tqdm.tqdm(loader,
                       ncols=100,
                       desc=f"[{epoch:03d}] train ({repetition + 1}/{repeat})",
                       leave=False,
                       file=sys.stdout,
                       unit=" batch")
        total_loss = {'train': 0}
        ######################
        if use_gan:
            if 'separated' not in input_D:
                total_loss['G'] = 0
                total_loss['D_real'] = 0
                total_loss['D_fake'] = 0
            else:
                total_loss['G_drums'] = 0
                total_loss['G_bass'] = 0
                total_loss['G_other'] = 0
                total_loss['G_vocals'] = 0
                total_loss['D_drums_real'] = 0
                total_loss['D_drums_fake'] = 0
                total_loss['D_bass_real'] = 0
                total_loss['D_bass_fake'] = 0
                total_loss['D_other_real'] = 0
                total_loss['D_other_fake'] = 0
                total_loss['D_vocals_real'] = 0
                total_loss['D_vocals_fake'] = 0
        ######################
        for idx, streams in enumerate(tq):
            if len(streams) < batch_size:
                # skip uncomplete batch for augment.Remix to work properly
                continue
            streams = streams.to(device)
            #######################################################################
            if use_gan:
                real_B = streams[:, 1:]
                real_B = augment(real_B)
                real_A = real_B.sum(dim=1)

                loss = optimize_parameters(model, netD, input_D, real_A, real_B, device, criterionGAN,
                                           criterion, lambda_L1, optimizer, optimizer_D)

                for key, value in loss:
                    total_loss[key] += value
                    current_loss[key] = total_loss[key] / (1 + idx)
                tq.set_postfix(loss=f"{current_loss['train']:.4f}")

                # free some space before next round
                del streams, real_B, real_A
            else:
                ########################################################################
                sources = streams[:, 1:]
                sources = augment(sources)
                mix = sources.sum(dim=1)

                estimates = model(mix)
                sources = center_trim(sources, estimates)
                loss = criterion(estimates, sources)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss['train'] += loss.item()
                current_loss['train'] = total_loss['train'] / (1 + idx)
                tq.set_postfix(loss=f"{current_loss['train']:.4f}")

                # free some space before next round
                del streams, sources, mix, estimates, loss

        if world_size > 1:
            sampler.epoch += 1

    if world_size > 1:
        #######################################################
        for key, value in loss:
            current_loss[key] = average_metric(current_loss[key])
        #######################################################

    return current_loss


def validate_model(epoch,
                   dataset,
                   model,
                   criterion,
                   device="cpu",
                   rank=0,
                   world_size=1,
                   shifts=0,
                   split=False):
    indexes = range(rank, len(dataset), world_size)
    tq = tqdm.tqdm(indexes,
                   ncols=100,
                   desc=f"[{epoch:03d}] valid",
                   leave=False,
                   file=sys.stdout,
                   unit=" track")
    current_loss = 0
    for index in tq:
        streams = dataset[index]
        # first five minutes to avoid OOM on --upsample models
        streams = streams[..., :15_000_000]
        streams = streams.to(device)
        sources = streams[1:]
        mix = streams[0]
        estimates = apply_model(model, mix, shifts=shifts, split=split)
        loss = criterion(estimates, sources)
        current_loss += loss.item() / len(indexes)
        del estimates, streams, sources

    if world_size > 1:
        current_loss = average_metric(current_loss, len(indexes))
    return current_loss
