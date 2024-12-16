import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .._base_ import load_clip_to_cpu, BaseCustomCLIP, BaseTrainer
from .coop import TextEncoder


_tokenizer = _Tokenizer()


_CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
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


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.KGCOOP.N_CTX
        ctx_init = cfg.TRAINER.KGCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if cfg.TRAINER.KGCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        
        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()
        
        temp = _CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512)),
            ("relu", nn.ReLU(inplace=True))
            #("linear2", nn.Linear(128, 512))
        ]))

        if cfg.TRAINER.PREC == "fp16":
            self.meta_net.half()

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        return torch.cat([
            prefix,  # (n_cls, 1, dim)
            ctx,
            suffix,  # (n_cls, *, dim)
        ], dim=1)


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


class CustomCLIP(BaseCustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)

        prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.learnable_params['prompt_learner'] = prompt_learner
        self.tokenized_prompts = prompt_learner.tokenized_prompts
        self.ori_embedding = prompt_learner.text_features
        self.text_encoder = TextEncoder(clip_model)
        
        self.meta_net = prompt_learner.meta_net
        self.adapter = Adapter(512, 4).to(clip_model.dtype)

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompt_learner = self.learnable_params['prompt_learner']
        prompts = prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts) 
        text_features_old = self.ori_embedding

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        score = cos(text_features,text_features_old)
        score = 1.0 - torch.mean(score)

        if label is not None:
            loss = F.cross_entropy(logits, label) + self.w * score
            return loss

        return logits, score


@TRAINER_REGISTRY.register()
class KgCoOp(BaseTrainer):
    def build_model(self):
        self.w = self.cfg.TRAINER.KGCOOP.W
        clip_model = super().build_model()
        self.model.w = self.w
        return clip_model
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PREC
        
        if prec == "amp":
            raise NotImplementedError
        else:
            loss = self.model(image, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.sched.step()

        return loss_summary

    def model_inference(self, input):
        return self.model(input)[0]

    def build_custom_clip(self, cfg, classnames, clip_model):
        self.model = CustomCLIP(cfg, classnames, clip_model)
