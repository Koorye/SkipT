import datetime
import numpy as np
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from dassl.engine import TrainerX
from dassl.utils import AverageMeter, MetricMeter, load_checkpoint, load_pretrained_weights

from clip import clip
from clip_maple import clip as clip_maple
from torchinfo import summary

from .optim import build_optimizer
from .lr_scheduler import build_lr_scheduler


def load_clip_to_cpu(cfg):
    """ load clip for coop, cocoop and kgcoop """
    
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def load_clip_to_cpu_maple(cfg):
    """ load clip for maple """
    
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip_maple._MODELS[backbone_name]
    model_path = clip_maple._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, 
                      "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip_maple.build_model(state_dict or model.state_dict(), design_details)

    return model


class ParameterWrapper(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
    
    def forward(self):
        return self.param

class BaseCustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.learnable_params = nn.ModuleDict()
        self.excluded_params = nn.ModuleDict()

    def forward(self, image):
        raise NotImplementedError
    
    def add_learnable_parameter(self, name, param):
        if issubclass(type(param), nn.Module):
            self.learnable_params[name] = param
        else:
            self.learnable_params[name] = ParameterWrapper(param)
    
    def remove_learnable_parameter(self, name):
        if name in self.learnable_params:
            del self.learnable_params[name]
    
    def clear_learnable_parameters(self):
        self.learnable_params.clear()
    
    def add_excluded_parameter(self, name, param):
        if issubclass(type(param), nn.Module):
            self.excluded_params[name] = param
        else:
            self.excluded_params[name] = ParameterWrapper(param)


class BaseTrainer(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ['fp16', 'fp32', 'amp']

    def build_model(self):
        """ modify parameters which need to update and save """
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = self.load_clip_model(cfg)

        if cfg.TRAINER.PREC == 'fp32' or cfg.TRAINER.PREC == 'amp':
            clip_model.float()

        print('Building custom CLIP')
        self.build_custom_clip(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        self.model.requires_grad_(False)
        self.model.learnable_params.requires_grad_(True)
        self.model.excluded_params.requires_grad_(False)

        # Double check
        enabled = []
        for name, param in self.model.learnable_params.named_parameters():
            if param.requires_grad:
                enabled.append((name, param))
                
        enabled = list(sorted(enabled, key=lambda x: x[0]))
        print(f'Parameters to be updated:')
        print_string = '\n'.join([f'  - {p[0]}: {tuple(p[1].shape)}' for p in enabled])
        print(print_string)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        optim_cfg = cfg.OPTIM
        self.optim = build_optimizer(self.model.learnable_params, optim_cfg)
        self.sched = build_lr_scheduler(self.optim, optim_cfg)
        self.register_model('learnable_params', self.model.learnable_params, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PREC == 'amp' else None

        device_count = torch.cuda.device_count()
        assert device_count == 1, 'Multiple GPUs are not supported!'
        return clip_model
    
    def load_clip_model(self, cfg):
        trainer = cfg.TRAINER.NAME.lower()
        if 'maple' in trainer:
            return load_clip_to_cpu_maple(cfg)
        else:
            return load_clip_to_cpu(cfg)

    def build_custom_clip(self, cfg, classnames, clip_model):
        self.model = BaseCustomCLIP(cfg, classnames, clip_model)
    
    def load_model(self, directory, epoch=None):
        """ get last epoch when epoch < 0, delete unuseful tokens, and customize state dict """
        
        if not directory:
            print('Note that load_model() is skipped as no pretrained model is given')
            return

        names = self.get_model_names()

        model_file = 'model-best.pth.tar'

        if epoch is not None:
            epoch = self._get_last_epoch(directory, names[0]) if epoch < 0 else epoch
            model_file = 'model.pth.tar-' + str(epoch)
        else:
            epoch = self._get_last_epoch(directory, names[0]) 
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            state_dict = self._delete_tokens(state_dict)
            state_dict = self.custom_state_dict(state_dict)
            
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def custom_state_dict(self, state_dict):
        return state_dict
    
    def before_epoch(self):
        super().before_epoch()
        self.epoch_start_time = time.time()

    def after_epoch(self):
        """ show training time """
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        
        epoch_time = time.time() - self.epoch_start_time
        speed = self.num_batches / epoch_time
        print(f"Epoch time: {epoch_time:.2f}s, Speed: {speed:.2f} batch/s")

        if last_epoch:
            training_time = time.time() - self.time_start
            speed = self.num_batches * self.max_epoch / training_time
            print(f"Training time: {training_time:.2f}s, Speed: {speed:.2f} batch/s")

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        """ show inference time """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        self.before_test()
        print(f"Evaluate on the *{split}* set")
        start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
            
        end_time = time.time()
        time_delta = end_time - start_time
        speed = len(data_loader.dataset) / time_delta
        print(f"Test time: {time_delta:.2f}s, Speed: {speed:.2f} img/s")
        self.after_test()

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    
    @torch.no_grad()
    def test_ood(self, data_loader, T):
        """Test-time OOD detection pipeline."""
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()
        self.before_test()

        glmcm_score = []
        mcm_score = []
        for batch_idx, (images, labels, *id_flag) in enumerate(tqdm(data_loader)):
            images = images.to(self.device)
            # output, output_local = self.model_inference(images)
            output = self.model_inference(images)
            output /= 100.0
            # output_local /= 100.0
            smax_global = to_np(F.softmax(output / T, dim=-1))
            # smax_local = to_np(F.softmax(output_local / T, dim=-1))
            mcm_global_score = -np.max(smax_global, axis=1)
            # mcm_local_score = -np.max(smax_local, axis=(1, 2))
            mcm_score.append(mcm_global_score)
            # glmcm_score.append(mcm_global_score + mcm_local_score)
            glmcm_score.append(mcm_global_score)

        self.after_test()
        return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[:len(data_loader.dataset)].copy()
    
    def before_train(self):
        super().before_train()
        # do not change the random state
        rng_state = torch.get_rng_state()
        self.model_summary()
        torch.set_rng_state(rng_state)
        self.time_start = time.time()
    
    def after_train(self):
        super().after_train()

    def before_test(self):
        pass
    
    def after_test(self):
        pass
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        return input, label
        
    def run_epoch(self):
        """ reset print function """
        
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if self.cfg.TRAIN.PRINT_FREQ > 0:
                meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
                only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += self.num_batches - self.batch_idx - 1
                    nb_remain += (
                        self.max_epoch - self.epoch - 1
                    ) * self.num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"{losses}"]
                    info += [f"lr {self.get_current_lr():.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    
    def model_summary(self):
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        input_size = self.cfg.INPUT.SIZE
        summary(self.model, input_size=[(batch_size, 3, *input_size), (batch_size, )], 
                device=self.device, mode='train', dtypes=[torch.float, torch.long])
    
    def _get_last_epoch(self, directory, name):
        filenames = os.listdir(osp.join(directory, name))
        filenames = [filename for filename in filenames if '.tar' in filename]
        epochs = [int(filename.split('-')[-1]) for filename in filenames]
        return max(epochs)

    def _delete_tokens(self, state_dict):
        tokens = ['token_prefix', 'token_suffix', 'token_midfix']

        for token in tokens:
            for key in list(state_dict.keys()):
                if token in key:
                    print(f'Delete key {key} from checkpoint')
                    del state_dict[key]

        return state_dict
