import datetime
import math
import random
import time
import torch
import torch.nn.functional as F
from scipy.stats import expon
from torch import nn
from tqdm import tqdm

from dassl.data.data_manager import build_data_loader, build_transform
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import AverageMeter, MetricMeter

from torchinfo import summary

from .baselines.ft_clip import CustomCLIP, FinetuneCLIP
from .feature_loader import FeatureLoader


def keep_rng_state(fn):
    def wrapper(*args, **kwargs):
        state = torch.get_rng_state()
        result = fn(*args, **kwargs)
        torch.set_rng_state(state)
        return result
    return wrapper


class TransformerWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.width = transformer.width
        self.layers = transformer.layers
        self.resblocks = nn.ModuleList(list(transformer.resblocks.children()))

    def forward(self, x, start_layer=None, end_layer=None):
        if start_layer is None:
            start_layer = 0
        if end_layer is None:
            end_layer = len(self.resblocks) - 1
        if end_layer < start_layer:
            return x
        for block in self.resblocks[start_layer:end_layer + 1]:
            x = block(x)
        return x


class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.transformer = TransformerWrapper(text_encoder.transformer)
        self.token_embedding = text_encoder.token_embedding if hasattr(text_encoder, 'token_embedding') else None
        self.positional_embedding = text_encoder.positional_embedding
        self.ln_final = text_encoder.ln_final
        self.text_projection = text_encoder.text_projection
        self.dtype = text_encoder.dtype
    
    def forward(self, x=None, start_layer=None, end_layer=None, tokenized_texts=None):
        if start_layer is None:
            x = self.pre_forward(tokenized_texts)
        
        x = self.transformer(x, start_layer, end_layer)

        if end_layer is not None:
            return x

        return self.post_forward(x, tokenized_texts)
    
    def pre_forward(self, tokenized_texts):
        x = self.token_embedding(tokenized_texts).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        return x.permute(1, 0, 2)
    
    def post_forward(self, x, tokenized_texts):
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection


class ImageEncoderWrapper(nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.input_resolution = image_encoder.input_resolution
        self.output_dim = image_encoder.output_dim
        self.conv1 = image_encoder.conv1
        self.class_embedding = image_encoder.class_embedding
        self.positional_embedding = image_encoder.positional_embedding
        self.ln_pre = image_encoder.ln_pre
        self.transformer = TransformerWrapper(image_encoder.transformer)
        self.ln_post = image_encoder.ln_post
        self.proj = image_encoder.proj
    
    def forward(self, x, start_layer=None, end_layer=None):
        if start_layer is None:
            x = self.pre_forward(x)
        
        x = self.transformer(x, start_layer, end_layer)

        if end_layer is not None:
            return x
        
        return self.post_forward(x)
    
    def pre_forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.type(x.dtype)
        x = self.ln_pre(x)
        return x.permute(1, 0, 2)
    
    def post_forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj


class SkipCustomCLIP(CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        self.start_layer = cfg.TRAINER.SKIP.START_LAYER
        
        super().__init__(cfg, classnames, clip_model)
        self.num_classes = len(classnames)
        self.top_ratio = cfg.TRAINER.SKIP.TOP_RATIO
        self.max_top = cfg.TRAINER.SKIP.MAX_TOP
        
        self.text_encoder = TextEncoderWrapper(self.text_encoder)
        self.image_encoder = ImageEncoderWrapper(self.image_encoder)

        self.add_learnable_parameter('text_encoder', self.text_encoder)
        self.add_learnable_parameter('image_encoder', self.image_encoder)
        self.set_exclude_parameters()

        self.text_features = None
        self.class_prototypes = None
        self.reserve_mask = None
        self.tokenized_texts = self.prompt_learner()

    def forward(self, images, labels=None):
        tokenized_texts = self.tokenized_texts

        if labels is not None:
            images = self.drop_tokens(images.type(self.dtype))
            image_features = self.image_encoder(images, start_layer=self.start_layer)
            text_features = self.text_features
            tokenized_texts, text_features, labels = self.select_classes(image_features, tokenized_texts, text_features, labels)
            text_features = self.text_encoder(text_features, start_layer=self.start_layer, tokenized_texts=tokenized_texts)
        else:
            image_features = self.image_encoder(images.type(self.dtype))
            text_features = self.text_encoder(tokenized_texts=tokenized_texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ text_features.t()

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            acc = compute_accuracy(logits, labels)[0]
            return loss, {'loss': loss, 'acc': acc}
        
        return logits
    
    @torch.no_grad()
    def select_classes(self, image_features, tokenized_texts, text_features, labels):
        num_tops = math.ceil(self.top_ratio * self.num_classes) \
                   if self.top_ratio <= 1.0 else math.ceil(min(self.top_ratio, self.num_classes))
        num_tops = min(num_tops, self.max_top)
        
        if self.top_ratio == 1.0 or num_tops == self.num_classes:
            return tokenized_texts, text_features, labels
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        class_prototypes = self.class_prototypes
        # (B, D) @ (D, C) -> (B, C)
        similarity = image_features @ class_prototypes.t()
        # make sure similarity of label is largest
        similarity[torch.arange(similarity.shape[0]), labels] = 1e4
        max_similarity, _ = similarity.max(dim=0)
        _, inds = max_similarity.sort(descending=True)

        if self.reserve_mask is None:
            x = torch.linspace(0.0, 5.0, steps=self.num_classes - num_tops)
            
            assert self.cfg.TRAINER.SKIP.LAMBDA > 0
            pdf = expon.pdf(x, scale=self.cfg.TRAINER.SKIP.LAMBDA)
            pdf = (pdf - pdf.min()) / (pdf.max() - pdf.min())
            reserve_ratios = [1.0] * num_tops + pdf.tolist()
            reserve_ratios = torch.tensor(reserve_ratios).to(max_similarity.device)
            self.reserve_mask = torch.rand_like(max_similarity) < reserve_ratios

        inds, _ = inds[self.reserve_mask].sort()

        # select text features
        # (C, L) -> (K', L)
        tokenized_texts = tokenized_texts[inds]
        # (L, C, D) -> (L, K', D)
        text_features = text_features[:, inds]

        # select labels
        # (B, ) -> (B, C) -> (B, K') -> (B, )
        labels = F.one_hot(labels, self.num_classes)
        labels = labels[:, inds].argmax(dim=1)
        return tokenized_texts, text_features, labels
    
    def forward_image_features(self, images):
        images = self.image_encoder.pre_forward(images.type(self.dtype))
        image_features_medium = self.image_encoder(images, start_layer=0, end_layer=self.start_layer - 1)
        image_features_out = self.image_encoder(image_features_medium, start_layer=self.start_layer)
        return image_features_medium, image_features_out
        
    def forward_text_features(self):
        text_features_in = self.text_encoder.pre_forward(self.tokenized_texts)
        text_features_medium = self.text_encoder(text_features_in, start_layer=0, end_layer=self.start_layer - 1)
        text_features_out = self.text_encoder(text_features_medium, start_layer=self.start_layer, tokenized_texts=self.tokenized_texts)
        return text_features_medium, text_features_out

    def drop_tokens(self, features):
        drop_ratio = random.uniform(0.0, self.cfg.TRAINER.SKIP.DROP_RATIO)
        num_drop = math.ceil(features.shape[0] * drop_ratio)
        drop_indices = torch.randperm(features.shape[0])[:num_drop]
        mask = torch.ones(features.shape[0], dtype=torch.bool)
        mask[drop_indices] = False
        return features[mask]
    
    def drop_layers_before_start(self):
        for layer in range(self.start_layer):
            print(f'Dropping transformer layer {layer}...')
            self.text_encoder.transformer.resblocks[layer] = nn.Identity()
            self.image_encoder.transformer.resblocks[layer] = nn.Identity()
        
        print('Dropping positional/token embeddings...')
        del self.text_encoder.positional_embedding
        del self.text_encoder.token_embedding
        del self.image_encoder.positional_embedding
        del self.image_encoder.class_embedding

        print('Dropping conv1...')
        del self.image_encoder.conv1
    
    def set_exclude_parameters(self):
        self.add_excluded_parameter('token_embedding', self.text_encoder.token_embedding)
        self.add_excluded_parameter('text_positional_embedding', self.text_encoder.positional_embedding)
        self.add_excluded_parameter('class_embedding', self.image_encoder.class_embedding)
        self.add_excluded_parameter('conv1', self.image_encoder.conv1)
        self.add_excluded_parameter('ln_pre', self.image_encoder.ln_pre)
        self.add_excluded_parameter('image_positional_embedding', self.image_encoder.positional_embedding)

        for layer in range(self.start_layer):
            self.add_excluded_parameter(f'text_encoder_resblocks{layer}', self.text_encoder.transformer.resblocks[layer])
            self.add_excluded_parameter(f'image_encoder_resblocks{layer}', self.image_encoder.transformer.resblocks[layer])


@TRAINER_REGISTRY.register()
class SkipTuning(FinetuneCLIP):
    def __init__(self, cfg):
        if cfg.DATASET.NAME in ['ImageNet', 'SUN397', 'Food101']:
            cfg.defrost()
            cfg.OPTIM.MAX_EPOCH = int(cfg.OPTIM.MAX_EPOCH / 4)
            cfg.freeze()
        super().__init__(cfg)

    def build_custom_clip(self, cfg, classnames, clip_model):
        self.model = SkipCustomCLIP(cfg, classnames, clip_model)
        self.num_classes = len(classnames)

    @torch.no_grad()
    def before_train(self):
        self.model.tokenized_texts = self.model.tokenized_texts.to(self.device)
        super().before_train()
        
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        mode = 'batch_random' if self.cfg.DATASET.NUM_SHOTS >= 4 else 'random'
        self.feature_loader = FeatureLoader(batch_size, mode)

        with torch.no_grad():
            print('Extracting text features...')
            text_features_medium, text_features_out = self.extract_text_features()

            print('Extracting image features...')
            image_features_medium, _, labels = self.extract_image_features(0)
            self.feature_loader.set_features(image_features_medium, labels)
           
        self.model.text_features = text_features_medium.to(self.device)
        class_prototypes = text_features_out / text_features_out.norm(dim=-1, keepdim=True)
        self.model.class_prototypes = class_prototypes.to(self.device)
        
        torch.cuda.empty_cache()
        self.time_start = time.time()
    
    @torch.no_grad()
    def before_test(self):
        super().before_test()
        self.model.tokenized_texts = self.model.tokenized_texts.to(self.device)
        text_features, _ = self.extract_text_features()
        self.model.text_features = text_features.to(self.device)
        
    @torch.no_grad()
    def before_epoch(self):
        super().before_epoch()
        
        if self.epoch != 0:
            text_features_medium, _ = self.extract_text_features()
            image_features, _, labels = self.extract_image_features(self.epoch)
            self.model.text_features = text_features_medium.to(self.device)
            self.feature_loader.set_features(image_features, labels)
                
        self.epoch_start_time = time.time()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.feature_loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.feature_loader.get_features()):
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
    
    def forward_backward(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        prec = self.cfg.TRAINER.PREC
        if prec == "amp":
            raise NotImplementedError
        else:
            loss, loss_summary = self.model(images, labels)
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    @torch.no_grad()
    def extract_text_features(self):
        text_features_medium, text_features_out = self.model.forward_text_features()
        return text_features_medium.cpu(), text_features_out.cpu()
    
    @torch.no_grad()
    def extract_image_features(self, epoch):
        tfm_train = build_transform(self.cfg, is_train=True)
        quick_loader = build_data_loader(
            self.cfg, sampler_type='SequentialSampler', data_source=self.dm.dataset.train_x, batch_size=self.cfg.TRAINER.SKIP.QUICK_BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN, n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train, is_train=False, dataset_wrapper=None)

        image_features_medium, image_features_out, labels = [], [], []
        for batch in tqdm(quick_loader, desc=f'Extracting image features for epoch {epoch}'):
            images, labels_ = self.parse_batch_train(batch)
            image_features_medium_, image_features_out_ = self.model.forward_image_features(images)
            image_features_medium.append(image_features_medium_.cpu())
            image_features_out.append(image_features_out_.cpu())
            labels.append(labels_.cpu())

        image_features_medium = torch.cat(image_features_medium, dim=1)
        image_features_out = torch.cat(image_features_out, dim=0)
        labels = torch.cat(labels, dim=0)
        return image_features_medium, image_features_out, labels

    def model_summary(self):
        vision_dim, text_dim = 768, 512
        vision_len, text_len = 1 + 14 * 14, 77
        self.model.text_features = torch.randn(text_len, self.num_classes, text_dim).to(self.device).type(self.model.dtype)
        self.model.class_prototypes = torch.randn(self.num_classes, text_dim).to(self.device).type(self.model.dtype)
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        summary(self.model, input_size=[(vision_len, batch_size, vision_dim), (batch_size, )], 
                device=self.device, mode='train', dtypes=[torch.float, torch.long])
