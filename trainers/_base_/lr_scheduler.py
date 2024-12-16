import math
import torch
from dassl.optim.lr_scheduler import ConstantWarmupScheduler, LinearWarmupScheduler


AVAI_SCHEDS = ["single_step", "multi_step", "cosine", 
               "cosine_restart", "cosine_norestart", "plateau"]


def cosine_norestart(epoch, max_epoch):
    epoch = min(epoch, max_epoch)
    lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / max_epoch))
    return lr_factor


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch)
    
    elif lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=stepsize, verbose=True)

    if optim_cfg.WARMUP_EPOCH > 0:
        if not optim_cfg.WARMUP_RECOUNT:
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )

        elif optim_cfg.WARMUP_TYPE == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_MIN_LR
            )

        else:
            raise ValueError

    return scheduler

