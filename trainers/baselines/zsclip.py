import time
import torch
from clip import clip

from dassl.engine import TRAINER_REGISTRY

from .._base_ import BaseTrainer, load_clip_to_cpu


_CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroShotCLIP(BaseTrainer):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = _CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_feats = clip_model.encode_text(prompts)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        self.text_feats = text_feats
        self.clip_model = clip_model
        
    def load_model(self, directory, epoch=None):
        print('Zero shot CLIP does not need pretrained model! Skipping to load model.')
        return
    
    def train(self):
        ''' skip loading and training step '''
        self.time_start = time.time()
        self.after_train()

    def model_inference(self, image):
        img_feats = self.clip_model.encode_image(image)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * img_feats @ self.text_feats.t()

        return logits
