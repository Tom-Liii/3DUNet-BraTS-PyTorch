import os
import time
import warnings
from copy import deepcopy
from os.path import join

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast

import utils.metrics as metrics
from configs import parse_seg_args
from dataset import brats2021
from models import get_unet
from utils.loss import SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, CaseSegMetricsMeterBraTS, ProgressMeter, LeaderboardBraTS,
                        brats_post_processing, initialization, load_cases_split, save_brats_nifti)
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler


def infer(args, epoch, model:nn.Module, infer_loader, writer, logger, mode:str, save_pred:bool=False):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    case_metrics_meter = CaseSegMetricsMeterBraTS()
    
    # make save epoch folder
    folder_dir = mode if epoch is None else f"{mode}_epoch_{epoch:02d}"
    save_path = join(args.exp_dir, folder_dir)
    if not os.path.exists(save_path):
        os.system(f"mkdir -p {save_path}")

    with torch.no_grad():
        end = time.time()
        for i, (image, label, _, brats_names) in enumerate(infer_loader):
            # get data
            image, label = image.cuda(), label.bool().cuda()
            bsz = image.size(0)

            # get seg map
            seg_map = sliding_window_inference(
                inputs=image, 
                predictor=model,
                roi_size=args.patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.patch_overlap,
                mode=args.sliding_window_mode
            )

            # discrete
            seg_map = torch.where(seg_map > 0.5, True, False)

            # post-processing
            seg_map = brats_post_processing(seg_map)

            # print the info of seg_map and label
            # print(f'seg_map ({type(seg_map)}): ', seg_map.shape)
            # print(f'label ({type(label)}): ', label.shape)
            # print(f'brats names: {brats_names}')

            # convert label to one hot embedding
            label = label.squeeze(1)
            one_hot_label = F.one_hot(label.long(), num_classes=5).float()
            one_hot_label = one_hot_label.permute(0, 4, 1, 2, 3)

            # calc metric 
            dice = metrics.dice(seg_map, one_hot_label)
            hd95 = metrics.hd95(seg_map, one_hot_label)

            # print(dice)
            # print(hd95)

            # output seg map
            if save_pred:
                save_brats_nifti(seg_map, brats_names, mode, args.data_root, save_path)

            # logging
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            case_metrics_meter.update(dice, hd95, brats_names, bsz)

            # monitor training progress
            # TODO: modify the metrics to fit in our 4-labels dataset 
            # ET (label 3): enhancing tumor
            # TC (label 1): tumor core
            # WT: whole tumor
            # CA (label 4): cavity
            # ED (label 2): edema
            if (i == 0) or (i + 1) % args.print_freq == 0:
                mean_metrics = case_metrics_meter.mean()
                logger.info("\t".join([
                    f'{mode.capitalize()}: [{epoch}][{i+1}/{len(infer_loader)}]', str(batch_time), 
                    f"Dice_BG {dice[:, 0].mean():.3f} ({mean_metrics['Dice_BG']:.3f})", # background
                    f"Dice_TC {dice[:, 1].mean():.3f} ({mean_metrics['Dice_TC']:.3f})", # tumor core
                    f"Dice_ED {dice[:, 2].mean():.3f} ({mean_metrics['Dice_ET']:.3f})", # edema
                    f"Dice_ET {dice[:, 3].mean():.3f} ({mean_metrics['Dice_ET']:.3f})", # enhancing tumor
                    f"Dice_RC {dice[:, 4].mean():.3f} ({mean_metrics['Dice_ET']:.3f})", # resection cavity
                    f"HD95_BG {hd95[:, 0].mean():7.3f} ({mean_metrics['HD95_BG']:7.3f})", # background
                    f"HD95_TC {hd95[:, 1].mean():7.3f} ({mean_metrics['HD95_TC']:7.3f})", # tumor core
                    f"HD95_ED {hd95[:, 2].mean():7.3f} ({mean_metrics['HD95_ET']:7.3f})", # edema
                    f"HD95_ET {hd95[:, 3].mean():7.3f} ({mean_metrics['HD95_ET']:7.3f})", # enhancing tumor
                    f"HD95_RC {hd95[:, 4].mean():7.3f} ({mean_metrics['HD95_ET']:7.3f})", # resection cavity
                ]))

            end = time.time()

        # output case metric csv
        case_metrics_meter.output(save_path)

    # get validation metrics and log to tensorboard
    infer_metrics = case_metrics_meter.mean()
    for key, value in infer_metrics.items():
        writer.add_scalar(f"{mode}/{key}", value, epoch)
    
    return infer_metrics


def main():
    args = parse_seg_args()
    logger, writer = initialization(args)

    # dataloaders
    train_cases, val_cases, test_cases = load_cases_split(args.cases_split)
    train_loader = brats2021.get_infer_loader(args, train_cases)
    val_loader   = brats2021.get_infer_loader(args, val_cases)
    test_loader  = brats2021.get_infer_loader(args, test_cases)

    # model & stuff
    model = get_unet(args).cuda()
    


    # load model
    if args.weight_path is not None:
        logger.info("==> Loading pretrain model...")
        assert args.weight_path.endswith(".pth")
        model_state = torch.load(args.weight_path)['model']
        model.load_state_dict(model_state)

    # test
    logger.info("==> Testing starts...")
    # best_model = model.state_dict()
    # model.load_state_dict(best_model)
    infer(args, 0, model, train_loader, writer, logger, mode='test', save_pred=args.save_pred)

    
    logger.info("==> Testing ends...")


if __name__ == '__main__':
    main()