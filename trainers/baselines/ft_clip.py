import clip
import torch
import torch.nn.functional as F
from torch import nn

from dassl.engine import TRAINER_REGISTRY

from .._base_ import BaseCustomCLIP, BaseTrainer


_CUSTOM_TEMPLATES = {
    "ImageNet": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetSketch": "a sketch of a {}.",
    "ImageNetA": "a bad photo of a {}.",
    "ImageNetR": "a rendition of a {}.",
    'Caltech101': 'itap of a {}.',
    'DescribableTextures': 'itap of a {}.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'FGVCAircraft': 'a bright photo of a {}, a type of aircraft.',
    'Food101': 'a photo of the nice {}.',
    'ImageNet': 'a photo of a {}.',
    'OxfordFlowers': 'a close-up photo of a {}.',
    'OxfordPets': 'a bad photo of a {}, a type of pet.',
    'SUN397': 'a photo of a {}.',
    'StanfordCars': 'a photo of the {}, a type of car.',
    'UCF101': 'a photo of a {}, a type of sport.',
}


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        template = _CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        texts = [template.format(c.replace("_", " ")) for c in classnames]
        print('Prompts:')
        print('\n'.join(texts))
        self.tokenized_texts = clip.tokenize(texts)
    
    def forward(self):
        return self.tokenized_texts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_texts):
        x = self.token_embedding(tokenized_texts).type(self.dtype)
        
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        return x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection


class CustomCLIP(BaseCustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames)
        self.prompt_learner.tokenized_texts = self.prompt_learner.tokenized_texts.cuda()
        self.text_encoder = TextEncoder(clip_model)
        
        trainer_cfg = cfg.TRAINER.FINETUNE
        
        if trainer_cfg.LEARNABLE_TEXT:
            self.add_learnable_parameter('text_encoder', self.text_encoder)
        else:
            device = next(self.text_encoder.parameters()).device
            self.text_encoder.cuda()

            with torch.no_grad():
                tokenized_texts = self.prompt_learner().cuda()
                text_features = self.text_encoder(tokenized_texts)

            self.text_encoder.to(device)
            self.text_features = text_features.to(device)
            
        if trainer_cfg.LEARNABLE_IMAGE:
            self.add_learnable_parameter('image_encoder', self.image_encoder)
        
        self.set_exclude_parameters()

    def forward(self, images, labels=None):
        image_features = self.image_encoder(images.type(self.dtype))
        
        if self.cfg.TRAINER.FINETUNE.LEARNABLE_TEXT:
            tokenized_texts = self.prompt_learner()
            text_features = self.text_encoder(tokenized_texts)
        else:
            text_features = self.text_features.to(images.device)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ text_features.t()

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            # acc = compute_accuracy(logits, labels)[0]
            # return loss, {'loss': loss, 'acc': acc}
            return loss, {'loss': loss}
        
        return logits
    
    def set_exclude_parameters(self):
        trainer_cfg = self.cfg.TRAINER.FINETUNE

        learnable_modules = self.cfg.TRAINER.FINETUNE.LEARNABLE_PREFIX_MODULES.split('+')
        
        if 'te' not in learnable_modules:
            self.add_excluded_parameter('token_embedding', self.text_encoder.token_embedding)
        if 'tpe' not in learnable_modules:
            self.add_excluded_parameter('text_positional_embedding', self.text_encoder.positional_embedding)
        if 'ce' not in learnable_modules:
            self.add_excluded_parameter('class_embedding', self.image_encoder.class_embedding)
        if 'conv' not in learnable_modules:
            self.add_excluded_parameter('conv1', self.image_encoder.conv1)
        if 'ln' not in learnable_modules:
            self.add_excluded_parameter('ln_pre', self.image_encoder.ln_pre)
        if 'ipe' not in learnable_modules:
            self.add_excluded_parameter('image_positional_embedding', self.image_encoder.positional_embedding)
        
        for layer in range(trainer_cfg.START_LAYER):
            self.add_excluded_parameter(f'text_encoder_resblocks{layer}', self.text_encoder.transformer.resblocks[layer])
            self.add_excluded_parameter(f'image_encoder_resblocks{layer}', self.image_encoder.transformer.resblocks[layer])


@TRAINER_REGISTRY.register()
class FinetuneCLIP(BaseTrainer):
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.PREC
        if prec == "amp":
            raise NotImplementedError
        else:
            loss, loss_summary = self.model(image, label)
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
