import copy
import json
import random
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torchinfo import summary

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler
from dassl.utils import load_pretrained_weights

from clip_coprompt import clip
from clip_coprompt.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .._base_ import BaseCustomCLIP, BaseTrainer, build_optimizer
from .constants import get_dataset_specified_config

_tokenizer = _Tokenizer()

dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}

CUSTOM_TEMPLATES = {
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


def cosine_loss(student_embedding, teacher_embedding):
    return (1 - F.cosine_similarity(student_embedding, teacher_embedding)).mean()


def load_clip_to_cpu(cfg, design_details=None):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if design_details is None:
        design_details = {
            "trainer": "CoPrompt",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": cfg.TRAINER.CoPrompt.N_CTX,
        }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [
            x,
            compound_prompts_deeper_text,
            0,
        ]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, clip_model_distill=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CoPrompt.N_CTX
        ctx_init = cfg.TRAINER.CoPrompt.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert (
            cfg.TRAINER.CoPrompt.PROMPT_DEPTH >= 1
        ), "For CoPrompt, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = (
            cfg.TRAINER.CoPrompt.PROMPT_DEPTH
        )  # max=12, but will create 11 such shared prompts
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print("CoPrompt design: Multi-modal Prompt Learning")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of CoPrompt context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        if dtype == torch.float16:
            self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow CoPrompt
        # compound prompts
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, 512))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        clip_model_ = clip_model_distill
        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model_.float()
        if torch.cuda.is_available():
            clip_model_.cuda()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        if torch.cuda.is_available():
            prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )  # pass here original, as for visual 768 is required


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(BaseCustomCLIP):
    def __init__(self, cfg, classnames, clip_model, clip_model_distill, clip_prompt_weights):
        super().__init__(cfg, classnames, clip_model)
        
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model, clip_model_distill)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = clip_prompt_weights
        self.text_encoder = TextEncoder(clip_model)
        self.distill_criteria = cfg.TRAINER.DISTILL
        self.model_distill = clip_model_distill
        self.lambd = cfg.TRAINER.W
        self.adapter_image = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.2
        
        self.learnable_params['prompt_learner'] = self.prompt_learner
        self.learnable_params['adapter_image'] = self.adapter_image
        self.learnable_params['adapter_text'] = self.adapter_text

    def forward(self, image1, image2=None, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = self.prompt_learner()
        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        )
        image_features = self.image_encoder(
            image1.type(self.dtype), shared_ctx, deep_compound_prompts_vision
        )

        x_a = self.adapter_image(image_features)
        image_features = (
            self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
        )

        x_b = self.adapter_text(text_features)
        text_features = (
            self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            pre_trained_text_features = self.ori_embedding[
                random.randint(0, self.ori_embedding.shape[0] - 1)
            ]
            pre_trained_image_features = self.model_distill.encode_image(image2)
            pre_trained_text_features = (
                pre_trained_text_features
                / pre_trained_text_features.norm(dim=-1, keepdim=True)
            )
            pre_trained_image_features = (
                pre_trained_image_features
                / pre_trained_image_features.norm(dim=-1, keepdim=True)
            )

            loss = F.cross_entropy(logits, label)

            if self.distill_criteria == "cosine":
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                score = cos(text_features, pre_trained_text_features)
                loss_distill_text = 1.0 - torch.mean(score)

                score = cos(image_features, pre_trained_image_features)
                loss_distill_image = 1.0 - torch.mean(score)

                loss_distill = loss_distill_text + loss_distill_image
            else:
                loss_distill = F.mse_loss(
                    text_features, pre_trained_text_features
                ) + F.mse_loss(image_features, pre_trained_image_features)

            return loss + self.lambd * loss_distill

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os

    os.makedirs("cache/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache/{dataset_name}_clip_weights_random.pt")
    return clip_weights


@TRAINER_REGISTRY.register()
class CoPrompt(BaseTrainer):
    def build_model(self):
        if self.cfg.DATASET.SUBSAMPLE_CLASSES in ['base', 'new']:
            self.cfg.merge_from_list(get_dataset_specified_config(self.cfg.DATASET.NAME))
        else:
            self.cfg.merge_from_list(get_dataset_specified_config('ImageNet'))
            
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        print("Loading original CLIP for distillation")
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = load_clip_to_cpu(cfg, design_details=design_details)

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            clip_model_distill.float()

        with open(
            f"gpt_file/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json"
        ) as f:
            gpt3_prompt = json.load(f)

        # Textual features
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_distill, cfg.DATASET.NAME
        )

        print("Building custom CLIP")
        self.model = CustomCLIP(
            cfg, classnames, clip_model, clip_model_distill, clip_weights
        )

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

    def forward_backward(self, batch):
        image1, image2, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PREC
        if prec == "amp":
            with autocast():
                loss = model(image1, image2, label).mean()
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image1, image2, label).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        image1, image2 = input[0], input[1]
        label = batch["label"]
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        label = label.to(self.device)
        return image1, image2, label
    
    def model_summary(self):
        batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        input_size = self.cfg.INPUT.SIZE
        summary(self.model, input_size=[(batch_size, 3, *input_size), 
                                        (batch_size, 3, *input_size), 
                                        (batch_size, )], 
                device=self.device, mode='train', 
                dtypes=[torch.float, torch.float, torch.long])
