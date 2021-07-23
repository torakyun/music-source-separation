# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import time
#from pytorch_memlab import profile, MemReporter

import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import apply_model, average_metric, center_trim


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


# @profile
def backward_D(model, netD, input_D, real_A, real_B, device, criterionGAN, batch_divide):
    """Calculate GAN loss for the discriminator"""

    log = {}
    if 'outputs' in input_D:
        log['D_real'] = 0
        log['D_fake'] = 0
    else:
        instruments = ['drums', 'bass', 'other', 'vocals']
        for instrument in instruments:
            log['D_'+instrument+'_real'] = 0
            log['D_'+instrument+'_fake'] = 0
            log['auxillary_'+instrument+'_real'] = 0
            log['auxillary_'+instrument+'_fake'] = 0

    # fake_B = fake_B.detach()
    for start in range(batch_divide):
        divided_real_A = real_A[start::batch_divide]
        divided_real_B = real_B[start::batch_divide]
        divided_fake_B = model(divided_real_A).detach()
        if 'mix' in input_D:
            divided_real_A = center_trim(divided_real_A, divided_fake_B)
        else:
            del divided_real_A
        divided_real_B = center_trim(divided_real_B, divided_fake_B)

        loss_D = 0

        if 'outputs' in input_D:
            divided_real_B = divided_real_B.view(divided_real_B.size(0), divided_real_B.size(
                1) * divided_real_B.size(2), divided_real_B.size(-1))
            divided_fake_B = divided_fake_B.view(divided_fake_B.size(0), divided_fake_B.size(
                1) * divided_fake_B.size(2), divided_fake_B.size(-1))

            if input_D == 'outputs+mix':
                # Fake; stop backprop to the generator by detaching fake_B
                # we use conditional GANs; we need to feed both input and output to the discriminator
                fake_AB = torch.cat((divided_real_A, divided_fake_B), 1)
                pred_fake = netD(fake_AB)
            elif input_D == 'outputs':
                pred_fake = netD(divided_fake_B)
            del divided_fake_B
            loss_D_fake = criterionGAN(pred_fake, False) / batch_divide
            del pred_fake
            log['D_fake'] += loss_D_fake.item()
            loss_D += loss_D_fake
            del loss_D_fake

            if input_D == 'outputs+mix':
                # Real
                real_AB = torch.cat((divided_real_A, divided_real_B), 1)
                pred_real = netD(real_AB)
            elif input_D == 'outputs':
                pred_real = netD(divided_real_B)
            del divided_real_B
            loss_D_real = criterionGAN(pred_real, True) / batch_divide
            del pred_real
            log['D_real'] += loss_D_real.item()
            loss_D += loss_D_real
            del loss_D_real
        else:
            criterionCE = torch.nn.CrossEntropyLoss()
            for i, instrument in enumerate(instruments):
                divided_real_B_i = divided_real_B[:, i, :, :]
                divided_fake_B_i = divided_fake_B[:, i, :, :]

                if input_D == 'output+mix':
                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB = torch.cat(
                        (divided_real_A, divided_fake_B_i), 1)
                    pred_fake, pred_label = netD(fake_AB)
                elif input_D == 'output':
                    pred_fake, pred_label = netD(divided_fake_B_i)
                elif input_D == 'output+mix+label':
                    # Fake; stop backprop to the generator by detaching fake_B
                    label = torch.eye(4, device=device)[i].view(1, 4, 1)
                    label = label.expand(divided_real_A.size(
                        0), 4, divided_real_A.size(-1))
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB = torch.cat(
                        (divided_real_A, divided_fake_B_i, label), 1)
                    pred_fake, pred_label = netD(fake_AB)
                elif input_D == 'output+label':
                    label = torch.eye(4, device=device)[i].view(1, 4, 1)
                    label = label.expand(divided_fake_B_i.size(
                        0), 4, divided_fake_B_i.size(-1))
                    divided_fake_B_i = torch.cat((divided_fake_B_i, label), 1)
                    pred_fake, pred_label = netD(divided_fake_B_i)
                del divided_fake_B_i
                loss_D_fake = criterionGAN(
                    pred_fake, False) / batch_divide / 4.
                del pred_fake
                loss_auxillary_fake = criterionCE(pred_label, torch.tensor(
                    i, device=device).unsqueeze(0)) / batch_divide / 4.
                del pred_label
                log['D_'+instrument+'_fake'] += loss_D_fake.item()
                log['auxillary_'+instrument+'_fake'] += loss_auxillary_fake.item()
                loss_D += loss_D_fake + loss_auxillary_fake
                del loss_D_fake, loss_auxillary_fake

                if input_D == 'output+mix':
                    # Real
                    real_AB = torch.cat(
                        (divided_real_A, divided_real_B_i), 1)
                    pred_real, pred_label = netD(real_AB)
                elif input_D == 'output':
                    pred_real, pred_label = netD(divided_real_B_i)
                if input_D == 'output+mix+label':
                    # Real
                    real_AB = torch.cat(
                        (divided_real_A, divided_real_B_i, label), 1)
                    pred_real, pred_label = netD(real_AB)
                elif input_D == 'output+label':
                    divided_real_B_i = torch.cat((divided_real_B_i, label), 1)
                    pred_real, pred_label = netD(divided_real_B_i)
                del divided_real_B_i
                loss_D_real = criterionGAN(pred_real, True) / batch_divide / 4.
                loss_auxillary_real = criterionCE(pred_label, torch.tensor(
                    i, device=device).unsqueeze(0)) / batch_divide / 4.
                del pred_real
                log['D_'+instrument+'_real'] += loss_D_real.item()
                log['auxillary_'+instrument+'_real'] += loss_auxillary_real.item()
                loss_D += loss_D_real + loss_auxillary_real
                del loss_D_real, loss_auxillary_real
        loss_D.backward()
    return log


# @profile
def backward_G(model, netD, input_D, real_A, real_B, device, criterionGAN, criterionL1, lambda_L1, batch_divide):
    """Calculate GAN and L1 loss for the generator"""

    log = {'train': 0}
    if 'outputs' in input_D:
        log['G'] = 0
    else:
        instruments = ['drums', 'bass', 'other', 'vocals']
        for instrument in instruments:
            log['G_'+instrument] = 0

    for start in range(batch_divide):
        divided_real_A = real_A[start::batch_divide]
        divided_real_B = real_B[start::batch_divide]
        divided_fake_B = model(divided_real_A)
        if 'mix' in input_D:
            divided_real_A = center_trim(divided_real_A, divided_fake_B)
        else:
            del divided_real_A
        divided_real_B = center_trim(divided_real_B, divided_fake_B)

        loss_G_L1 = criterionL1(divided_fake_B, divided_real_B) / batch_divide
        del divided_real_B
        log['train'] += loss_G_L1.item()
        loss_G = loss_G_L1 * lambda_L1
        del loss_G_L1

        if 'outputs' in input_D:
            divided_fake_B = divided_fake_B.view(divided_fake_B.size(0), divided_fake_B.size(
                1) * divided_fake_B.size(2), divided_fake_B.size(-1))

            # First, G(A) should fake the discriminator
            if input_D == 'outputs+mix':
                fake_AB = torch.cat((divided_real_A, divided_fake_B), 1)
                pred_fake = netD(fake_AB)
            elif input_D == 'outputs':
                pred_fake = netD(divided_fake_B)
            del divided_fake_B
            loss_G_GAN = criterionGAN(pred_fake, True) / batch_divide
            del pred_fake
            log['G'] += loss_G_GAN.item()
            loss_G += loss_G_GAN
            del loss_G_GAN
        else:
            for i, instrument in enumerate(instruments):
                divided_fake_B_i = divided_fake_B[:, i, :, :]

                if input_D == 'output+mix':
                    fake_AB = torch.cat(
                        (divided_real_A, divided_fake_B_i), 1)
                    pred_fake, _ = netD(fake_AB)
                elif input_D == 'output':
                    pred_fake, _ = netD(divided_fake_B_i)
                if input_D == 'output+mix+label':
                    label = torch.eye(4, device=device)[i].view(1, 4, 1)
                    label = label.expand(divided_real_A.size(
                        0), 4, divided_real_A.size(-1))
                    fake_AB = torch.cat(
                        (divided_real_A, divided_fake_B_i, label), 1)
                    pred_fake, _ = netD(fake_AB)
                elif input_D == 'output+label':
                    label = torch.eye(4, device=device)[i].view(1, 4, 1)
                    label = label.expand(divided_fake_B_i.size(
                        0), 4, divided_fake_B_i.size(-1))
                    divided_fake_B_i = torch.cat((divided_fake_B_i, label), 1)
                    pred_fake, _ = netD(divided_fake_B_i)
                del divided_fake_B_i
                loss_G_GAN = criterionGAN(pred_fake, True) / batch_divide / 4.
                del pred_fake
                log['G_'+instrument] += loss_G_GAN.item()
                loss_G += loss_G_GAN
                del loss_G_GAN
        loss_G.backward()
        del loss_G
    return log


# @profile
def optimize_parameters(model, netD, input_D, real_A, real_B, device, criterionGAN, criterionL1, lambda_L1, optimizer_G, optimizer_D, batch_divide):
    if 'separated' not in input_D:
        # update D
        set_requires_grad(model, False)
        set_requires_grad(netD, True)  # enable backprop for D
        # calculate gradients for D
        loss_D = backward_D(model, netD, input_D, real_A, real_B,
                            device, criterionGAN, batch_divide)
        optimizer_D.step()          # update D's weights
        optimizer_D.zero_grad()     # set D's gradients to zero
        # update G
        # D requires no gradients when optimizing G
        set_requires_grad(model, True)
        set_requires_grad(netD, False)
        loss_G = backward_G(model, netD, input_D, real_A, real_B, device, criterionGAN,
                            criterionL1, lambda_L1, batch_divide)                   # calculate graidents for G
        optimizer_G.step()             # udpate G's weights
        optimizer_G.zero_grad()        # set G's gradients to zero
        return dict(loss_G, **loss_D)
    else:
        # update D
        for key in netD:
            set_requires_grad(netD[key], True)  # enable backprop for D
            optimizer_D[key].zero_grad()     # set D's gradients to zero
        # calculate gradients for D
        loss_D = backward_D(model, netD, input_D, real_A, real_B,
                            device, criterionGAN, batch_divide)
        for value in optimizer_D.values():
            value.step()          # update D's weights
        # update G
        for key in netD:
            # D requires no gradients when optimizing G
            set_requires_grad(netD[key], False)
            optimizer_G[key].zero_grad()        # set G's gradients to zero
        loss_G = backward_G(model, netD, input_D, real_A, real_B, device, criterionGAN,
                            criterionL1, lambda_L1, batch_divide)                   # calculate graidents for G
        for value in optimizer_G.values():
            value.step()             # udpate G's weights
        return dict(loss_G, **loss_D)


# @profile
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
                batch_size=16,
                batch_divide=1):

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
    if use_gan:
        if 'outputs' in input_D:
            current_loss['G'] = 0
            current_loss['D_real'] = 0
            current_loss['D_fake'] = 0
        else:
            instruments = ['drums', 'bass', 'other', 'vocals']
            for instrument in instruments:
                current_loss['G_'+instrument] = 0
                current_loss['D_'+instrument+'_real'] = 0
                current_loss['D_'+instrument+'_fake'] = 0
                current_loss['auxillary_'+instrument+'_real'] = 0
                current_loss['auxillary_'+instrument+'_fake'] = 0
    for repetition in range(repeat):
        tq = tqdm.tqdm(loader,
                       ncols=100,
                       desc=f"[{epoch:03d}] train ({repetition + 1}/{repeat})",
                       leave=False,
                       file=sys.stdout,
                       unit=" batch")
        total_loss = {'train': 0}
        if use_gan:
            if 'outputs' in input_D:
                total_loss['G'] = 0
                total_loss['D_real'] = 0
                total_loss['D_fake'] = 0
            else:
                for instrument in instruments:
                    total_loss['G_'+instrument] = 0
                    total_loss['D_'+instrument+'_real'] = 0
                    total_loss['D_'+instrument+'_fake'] = 0
                    total_loss['auxillary_'+instrument+'_real'] = 0
                    total_loss['auxillary_'+instrument+'_fake'] = 0
        for idx, streams in enumerate(tq):
            # if idx == 2:break
            if len(streams) < batch_size:
                # skip uncomplete batch for augment.Remix to work properly
                continue
            streams = streams.to(device)
            if use_gan:
                real_B = streams[:, 1:]
                real_B = augment(real_B)
                del streams
                real_A = real_B.sum(dim=1)

                # reporter_G = MemReporter(model)
                # reporter_D = MemReporter(netD)
                # reporter_G.report()
                # reporter_D.report()
                loss = optimize_parameters(model, netD, input_D, real_A, real_B, device, criterionGAN,
                                           criterion, lambda_L1, optimizer, optimizer_D, batch_divide)
                # reporter_G.report()
                # reporter_D.report()

                for key, value in loss.items():
                    total_loss[key] += value
                    current_loss[key] = total_loss[key] / (1 + idx)
                tq.set_postfix(loss=f"{current_loss['train']:.4f}")

                # free some space before next round
                del real_A, real_B, loss
            else:
                sources = streams[:, 1:]
                sources = augment(sources)
                del streams
                mix = sources.sum(dim=1)

                for start in range(batch_divide):
                    divided_sources = sources[start::batch_divide]
                    divided_mix = mix[start::batch_divide]
                    divided_estimates = model(divided_mix)
                    del divided_mix
                    divided_sources = center_trim(
                        divided_sources, divided_estimates)

                    loss = criterion(divided_estimates,
                                     divided_sources) / batch_divide
                    del divided_estimates, divided_sources
                    loss.backward()
                    total_loss['train'] += loss.item()
                    del loss, divided_sources, divided_estimates

                del sources, mix
                optimizer.step()
                optimizer.zero_grad()

                current_loss['train'] = total_loss['train'] / (1 + idx)
                tq.set_postfix(loss=f"{current_loss['train']:.4f}")

        if world_size > 1:
            sampler.epoch += 1

    if world_size > 1:
        for key, value in current_loss.items():
            current_loss[key] = average_metric(value)

    return current_loss


# @profile
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
