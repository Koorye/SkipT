"""
Modified from Dassl
"""
import warnings
import torch
import torch.nn as nn

from dassl.optim.radam import RAdam

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]


def build_optimizer(model, optim_cfg, param_groups=None):
    """ add new_lr_mult and print info """
    
    optim = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    base_lr_mult = optim_cfg.BASE_LR_MULT
    # new_lr_mult = optim_cfg.NEW_LR_MULT
    new_lr = optim_cfg.NEW_LR

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
        )

    if param_groups is not None and staged_lr:
        warnings.warn(
            "staged_lr will be ignored, if you need to use staged_lr, "
            "please bind it with param_groups yourself."
        )

    if param_groups is None:
        if staged_lr:
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "When staged_lr is True, model given to "
                    "build_optimizer() must be an instance of nn.Module"
                )

            if isinstance(model, nn.DataParallel):
                model = model.module

            if isinstance(new_layers, str):
                if new_layers is None:
                    warnings.warn("new_layers is empty (staged_lr is useless)")
                new_layers = [new_layers]

            base_params, new_params = [], []
            base_modules, new_modules = [], []

            for name, param in model.named_parameters():
                is_new = False
                for layer in new_layers:
                    if layer in name:
                        is_new = True
                        break
                
                if is_new:
                    new_modules.append(name)
                    new_params.append(param)
                else:
                    base_modules.append(name)
                    base_params.append(param)
            
            if len(new_modules) > 0:
                param_groups = [
                    {
                        "name": "base",
                        "params": base_params,
                        "lr": lr
                    },
                    {
                        "name": "new",
                        "params": new_params,
                        "lr": new_lr
                    },
                ]
                print('Use staged learning rate!')
                print(f'Base modules (lr={lr}):')
                # [print(f'  - {module}') for module in base_modules]
                print_string = '\n'.join([f'  - {module}' for module in base_modules])
                print(print_string)
                print(f'New modules (lr={new_lr}):')
                # [print(f'  - {module}') for module in new_modules]
                print_string = '\n'.join([f'  - {module}' for module in new_modules])
                print(print_string)
            else:
                print('Warning! Use staged learning rate but no new modules find, staged_lr will be ignored!')
                if isinstance(model, nn.Module):
                    param_groups = model.parameters()
                else:
                    param_groups = model

        else:
            if isinstance(model, nn.Module):
                param_groups = model.parameters()
            else:
                param_groups = model

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=1e-4,
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optim == "radam":
        optimizer = RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    return optimizer

