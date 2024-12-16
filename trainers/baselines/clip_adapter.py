import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy

from .._base_ import BaseCustomCLIP, BaseTrainer


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
   
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.text_features = self.clip_model.encode_text(prompts.cuda())
        return self.text_features


class CustomCLIP(BaseCustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        text_encoder = TextEncoder(cfg, classnames, clip_model.cuda())
        with torch.no_grad():
            self.text_features = text_encoder()
        self.adapter = Adapter(512, 4).to(clip_model.dtype)
        self.learnable_params['adapter'] = self.adapter
            
    def forward(self, image, labels=None):
        image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.text_features.detach()
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            acc = compute_accuracy(logits, labels)[0]
            return loss, {'loss': loss.item(), 'acc': acc}

        return logits


@TRAINER_REGISTRY.register()
class CLIPAdapter(BaseTrainer):
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        loss, loss_summary = self.model(image, label)
        self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
