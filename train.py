import argparse
import os
from math import log10
import time
import psutil
import GPUtil
import json
from collections import defaultdict

import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
from torch.utils.data import Subset
import random


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--train_sample_count', default=10000, type=int, help='number of training samples to use')

def get_performance_metrics():
    gpus = GPUtil.getGPUs()
    gpu_util = gpus[0].load * 100 if gpus else 0
    gpu_mem_mb = gpus[0].memoryUsed if gpus else 0
    gpu_mem_gb = gpu_mem_mb / 1024 if gpu_mem_mb else 0
    cpu_util = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_util = ram.percent
    ram_used_mb = ram.used / (1024 * 1024)  # Convert bytes to MB
    ram_used_gb = ram_used_mb / 1024  # Convert MB to GB
    return {
        'gpu_util': gpu_util,
        'gpu_mem_mb': gpu_mem_mb,
        'gpu_mem_gb': gpu_mem_gb,
        'cpu_util': cpu_util,
        'ram_util': ram_util,
        'ram_used_mb': ram_used_mb,
        'ram_used_gb': ram_used_gb,
    }

if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    
    # Add this line to set the desired number of training samples
    TRAIN_SAMPLE_COUNT = 100  # Change this to your desired number
    
    train_set = TrainDatasetFromFolder('data/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    
    # Limit the number of training samples
    if len(train_set) > TRAIN_SAMPLE_COUNT:
        indices = random.sample(range(len(train_set)), TRAIN_SAMPLE_COUNT)
        train_set = Subset(train_set, indices)
    
    val_set = ValDatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    performance_metrics = defaultdict(list)
    
    out_path = 'statistics/'
    os.makedirs(out_path, exist_ok=True)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        
        batch_metrics = []
        
        for batch_num, (data, target) in enumerate(train_bar):
            batch_start_time = time.time()
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data
            if torch.cuda.is_available():
                z = z.float().cuda()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            ############################
            # (2) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img.detach()).mean()
            d_loss = 1 - real_out + fake_out

            optimizerD.zero_grad()
            d_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerD.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
            
            # Record batch metrics
            batch_metrics.append({
                'batch_num': batch_num,
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item(),
                'd_score': real_out.item(),
                'g_score': fake_out.item(),
                'batch_time': time.time() - batch_start_time,
                **get_performance_metrics()
            })
    
        netG.eval()
        # ... (validation code remains the same)
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.float().cuda()
                    hr = hr.float().cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            
            # Save only 10 validation image results for every 20 epochs
            if epoch % 10 == 0:
                val_images = torch.stack(val_images[:30])  # Take only first 10 sets (30 images)
                val_images = torch.chunk(val_images, val_images.size(0) // 3)
                val_save_bar = tqdm(val_images, desc='[saving validation results]')
                for i, image in enumerate(val_save_bar):
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, out_path + 'epoch_%d_img_%d.png' % (epoch, i+1), padding=5)



        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        # Save batch metrics
        with open(f'{out_path}epoch_{epoch}_batch_metrics.json', 'w') as f:
            json.dump(batch_metrics, f)
        
        # Calculate and save epoch metrics
        epoch_time = time.time() - epoch_start_time
        epoch_metrics = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'd_loss': results['d_loss'][-1],
            'g_loss': results['g_loss'][-1],
            'd_score': results['d_score'][-1],
            'g_score': results['g_score'][-1],
            'psnr': results['psnr'][-1],
            'ssim': results['ssim'][-1],
            **{k: sum(d[k] for d in batch_metrics) / len(batch_metrics) for k in ['gpu_util', 'gpu_mem_mb', 'gpu_mem_gb', 'cpu_util', 'ram_util', 'ram_used_mb', 'ram_used_gb']}
        }
        performance_metrics['epoch'].append(epoch)
        for k, v in epoch_metrics.items():
            if k != 'epoch':
                performance_metrics[k].append(v)
        
        # Save performance metrics every epoch
        perf_df = pd.DataFrame(performance_metrics)
        perf_df.to_csv(f'{out_path}performance_metrics.csv', index=False)
        
        # Save training results every 10 epochs
        if epoch % 10 == 0:
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

        print(f"Epoch {epoch} completed. Time taken: {epoch_time:.2f}s")
        print(f"Average metrics - GPU Util: {epoch_metrics['gpu_util']:.2f}%, "
              f"GPU Mem: {epoch_metrics['gpu_mem_mb']:.2f}MB ({epoch_metrics['gpu_mem_gb']:.2f}GB), "
              f"CPU Util: {epoch_metrics['cpu_util']:.2f}%, "
              f"RAM Util: {epoch_metrics['ram_util']:.2f}%, "
              f"RAM Used: {epoch_metrics['ram_used_mb']:.2f}MB ({epoch_metrics['ram_used_gb']:.2f}GB)")