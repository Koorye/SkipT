import os.path as osp
import torch
import torch.nn.functional as F
from torch import nn

from dassl.engine import TRAINER_REGISTRY

from datasets.imagenet import ImageNet
from dassl.utils import listdir_nohidden

from .kgcoop import CustomCLIP as CustomCLIP_, KgCoOp


class FiLM(nn.Module):
    def __init__(self, 
                 dim, 
                 bias=True, 
                 use_sigmoid=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.has_bias = bias
        self.use_sigmoid = use_sigmoid
    
    def forward(self, x):
        scale = self.scale.unsqueeze(0).type(x.dtype)
        bias = self.bias.unsqueeze(0).type(x.dtype) if self.has_bias else None
        
        x = scale * x
        if bias is not None:
            x = x + bias
        
        if self.use_sigmoid:
            return x.sigmoid()
        
        return x


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        self.dataset = cfg.DATASET.NAME
        self.trainer_cfg = cfg.TRAINER.DEPT

        clip_dim = clip_model.text_projection.size(1)
        self.film_img = FiLM(clip_dim)
        self.film_text = FiLM(clip_dim)
        
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            self.cls_head = nn.Linear(clip_dim, len(classnames)).type(self.dtype)
        else:
            self.cls_head = nn.Identity()
        
    def forward(self, img, labels=None):
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            return self._forward_base(img, labels)
        else:
            return self._forward_new(img)

    def _forward_base(self, img, labels=None):
        text_feats, img_feats = self._forward_feats(img)
        logits = self._forward_logits_sim(text_feats, img_feats)
        logits_cls, labels_cls = self._forward_logits_cls(text_feats, img_feats, labels)
        
        if labels is not None:
            return self._loss(logits, labels, logits_cls, labels_cls, text_feats)
        else:
            sim_weight = self.trainer_cfg.SIM_WEIGHT
            cls_weight = self.trainer_cfg.CLS_WEIGHT
            logits = sim_weight * logits + cls_weight * logits_cls
            return logits
    
    def _forward_new(self, img):
        text_feats, img_feats = self._forward_feats(img)
        logits = self._forward_logits_sim(text_feats, img_feats)
        return logits
    
    def _forward_feats(self, img):
        prompts = self.learnable_params['prompt_learner']()
        tokenized_prompts = self.tokenized_prompts
        text_feats = self.text_encoder(prompts, tokenized_prompts)
        img_feats = self.image_encoder(img.type(self.dtype))
        return text_feats, img_feats
    
    def _forward_logits_sim(self, text_feats, img_feats):
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feats.t()
        return logits
    
    def _forward_logits_cls(self, text_feats, img_feats, labels):
        text_feats = self.film_text(text_feats)
        img_feats = self.film_img(img_feats)

        if labels is None:
            all_feats = img_feats
            all_labels = labels
        else:
            text_feats = text_feats[labels]
            all_feats = torch.cat([text_feats, img_feats])
            all_labels = torch.cat([labels, labels])

        all_logits = self.cls_head(all_feats)
        return all_logits, all_labels
    
    def _loss(self, logits, labels, logits_cls, labels_cls, text_feats):
        loss_sim = F.cross_entropy(logits, labels)
        loss_cls = F.cross_entropy(logits_cls, labels_cls)

        text_feats_old = self.ori_embedding
        text_feats_old = text_feats_old / text_feats_old.norm(dim=-1, keepdim=True)

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        score = cos(text_feats, text_feats_old)
        loss_kg = 1.0 - torch.mean(score)

        sim_weight = self.trainer_cfg.SIM_WEIGHT
        cls_weight = self.trainer_cfg.CLS_WEIGHT
        loss = sim_weight * loss_sim + cls_weight * loss_cls + self.w * loss_kg
        return loss


@TRAINER_REGISTRY.register()
class KgDePT(KgCoOp):
    def build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
    
    def model_inference(self, input):
        return self.model(input)
    
    def custom_state_dict(self, state_dict):
        if self.cfg.DATASET.NAME in ['ImageNetA', 'ImageNetR']:
            dataset = self.dm.dataset
            text_file = osp.join(dataset.dataset_dir, "classnames.txt")
            all_folders = ImageNet.read_classnames(text_file).keys()

            folders = [f for f in listdir_nohidden(dataset.image_dir, sort=True)
                       if f not in 'README.txt']
            is_reserves = [f in folders for f in all_folders]

            print(f'State dict is CLIPPED to match the shape of target dataset {self.cfg.DATASET.NAME}!')
            if 'cls_head.weight' in state_dict:
                state_dict['classifier.weight'] = state_dict['classifier.weight'][is_reserves]
            if 'cls_head.bias' in state_dict:
                state_dict['classifier.bias'] = state_dict['classifier.bias'][is_reserves]
        return state_dict
