"""Fine-tune a HCF ViT backbone for the eye modality.

This script loads a pretrained HCF checkpoint and fine-tunes it as a classifier
on retinal images (fundus or OCTA). It is the entry point used by the shell
scripts in `scripts/train_eye_*.sh`.

"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit
from engine_finetune import train_one_epoch, evaluate


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "HCF fine-tuning for eye-modality classification", add_help=True
    )

    # Training hyper-parameters
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Per-GPU batch size.")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="Gradient accumulation steps.")

    # Model
    parser.add_argument("--model", default="vit_large_patch16", type=str,
                        metavar="MODEL", help="Name of model to train.")
    parser.add_argument("--checkpoint_key", default="model", type=str,
                        help="Key to use in the checkpoint dict.")
    parser.add_argument("--input_size", default=224, type=int,
                        help="Input image size.")
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT",
                        help="Drop path rate.")

    # Optimizer
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR",
                        help="Absolute learning rate (overrides --blr).")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR",
                        help="Base learning rate: lr = blr * total_batch_size / 256.")
    parser.add_argument("--layer_decay", type=float, default=0.75,
                        help="Layer-wise LR decay (ELECTRA / BEiT).")
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N")

    # Augmentation
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", metavar="NAME",
                        help="AutoAugment policy.")
    parser.add_argument("--smoothing", type=float, default=0.1,
                        help="Label smoothing factor.")
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT",
                        help="Random-erase probability.")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)

    # Mixup / CutMix
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # Fine-tuning
    parser.add_argument("--finetune", default="", type=str,
                        help="Path to pretrained HCF checkpoint.")
    parser.add_argument("--task", default="", type=str,
                        help="Task subdirectory under output_dir.")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool",
                        help="Use the CLS token instead of global pooling.")

    # Data
    parser.add_argument("--data_path", default="./data", type=str,
                        help="Root of ImageFolder-style dataset with train/val/test.")
    parser.add_argument("--nb_classes", default=4, type=int)
    parser.add_argument("--output_dir", default="./output_dir",
                        help="Directory for checkpoints.")
    parser.add_argument("--log_dir", default="./output_dir",
                        help="Directory for TensorBoard logs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="Resume from checkpoint.")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluation-only mode.")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


def main(args: argparse.Namespace) -> None:
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_dataset(is_train="train", args=args)
    dataset_val = build_dataset(is_train="val", args=args)
    dataset_test = build_dataset(is_train="test", args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print("Warning: validation set not divisible by process number; "
                  "duplicate entries will be added.")
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    mixup_fn = None
    mixup_active = (args.mixup > 0 or args.cutmix > 0.0
                    or args.cutmix_minmax is not None)
    if mixup_active:
        print("Mixup is activated.")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode, label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint[args.checkpoint_key]

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(
        model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        evaluate(data_loader_test, model, device, args,
                 epoch=0, mode="test", num_class=args.nb_classes,
                 log_writer=log_writer)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_auc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer, args=args,
        )

        val_stats, val_auc_roc = evaluate(
            data_loader_val, model, device, args, epoch,
            mode="val", num_class=args.nb_classes, log_writer=log_writer,
        )

        if max_auc < val_auc_roc:
            max_auc = val_auc_roc
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler,
                    epoch=epoch, mode="best",
                )

        if log_writer is not None:
            log_writer.add_scalar("loss/val", val_stats["loss"], epoch)
            log_writer.flush()

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch, "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.task, "log.txt"),
                      mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # Final evaluation on the test set using the best checkpoint
    ckpt_path = os.path.join(args.output_dir, args.task, "checkpoint-best.pth")
    state_dict_best = torch.load(ckpt_path, map_location="cpu")
    model_without_ddp.load_state_dict(state_dict_best["model"], strict=False)
    evaluate(data_loader_test, model_without_ddp, device, args,
             epoch=0, mode="test", num_class=args.nb_classes, log_writer=None)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
