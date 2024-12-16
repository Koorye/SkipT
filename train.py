import sys
sys.path.append('.')

import argparse
import os.path as osp
import shutil
import traceback
import torch

from dassl.utils import set_random_seed, collect_env_info
from dassl.engine import build_trainer

from utils.logger import setup_logger
from utils.misc import remove_if_exist, check_log_valid

# register datasets and trainers
import datasets
import trainers

from trainers._base_.config import get_cfg_default, clean_cfg


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    # dataset configs
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # optim configs
    cfg.OPTIM.STAGED_LR = False
    cfg.OPTIM.NEW_LAYERS = ['image_encoder']
    cfg.OPTIM.LR = 1e-5
    cfg.OPTIM.MAX_EPOCH = 10
    
    # training configs
    cfg.TRAIN.CHECKPOINT_FREQ = -1
    cfg.TRAINER.PREC = 'fp16'

    # Skip configs
    cfg.TRAINER.SKIP = CN()
    # delay settings
    cfg.TRAINER.SKIP.START_LAYER = 6
    # loss settings
    cfg.TRAINER.SKIP.TOP_RATIO = 0.5
    cfg.TRAINER.SKIP.MAX_TOP = 50
    cfg.TRAINER.SKIP.LAMBDA = 0.3
    cfg.TRAINER.SKIP.DROP_RATIO = 0.0
    # data settings
    cfg.TRAINER.SKIP.QUICK_BATCH_SIZE = 128

    # CoOp configs
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.COOP.ALPHA = 1.0 # for KgCoOp but NOT USE
    cfg.TRAINER.COOP.W = 2.0 # for KgCoOp

    # CoCoOp configs
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = "a photo of a"  # initialization words

    # KgCoOp settings
    cfg.TRAINER.KGCOOP = CN()
    cfg.TRAINER.KGCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.KGCOOP.CSC = False  # class-specific context
    cfg.TRAINER.KGCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.KGCOOP.W = 1.5

    # Finetune CLIP configs
    cfg.TRAINER.FINETUNE = CN()
    cfg.TRAINER.FINETUNE.LEARNABLE_TEXT = True
    cfg.TRAINER.FINETUNE.LEARNABLE_IMAGE = True
    cfg.TRAINER.FINETUNE.LEARNABLE_PREFIX_MODULES = ''
    cfg.TRAINER.FINETUNE.START_LAYER = 0

    # MaPLe configs
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)

    # PromptSRC configs
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 12  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 12  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1

    # DePT configs
    cfg.TRAINER.DEPT = CN()
    cfg.TRAINER.DEPT.SIM_WEIGHT = 0.3
    cfg.TRAINER.DEPT.CLS_WEIGHT = 0.7

    # CoPrompt configs
    cfg.TRAINER.CoPrompt = CN()
    cfg.TRAINER.CoPrompt.N_CTX = 2  # number of context vectors
    cfg.TRAINER.CoPrompt.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.CoPrompt.PROMPT_DEPTH = 9
    cfg.TRAINER.W = 8.0
    cfg.TRAINER.DISTILL = "cosine"

    # ProGrad configs
    cfg.TRAINER.PROGRAD = CN()
    cfg.TRAINER.PROGRAD.N_CTX = 4  # number of context vectors
    cfg.TRAINER.PROGRAD.CSC = False  # class-specific context
    cfg.TRAINER.PROGRAD.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROGRAD.CLASS_TOKEN_POSITION = "end"
    cfg.LOSS = CN()
    cfg.LOSS.GM = False
    cfg.LOSS.NAME = ""
    cfg.LOSS.ALPHA = 0.
    cfg.LOSS.T = 1.
    cfg.LOSS.LAMBDA = 1.

def setup_cfg(args):
    cfg = get_cfg_default()
    clean_cfg(cfg, 'COOP')
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)

    exception_path = osp.join(cfg.OUTPUT_DIR, 'exceptions.txt')
    remove_if_exist(exception_path)

    log_path = osp.join(cfg.OUTPUT_DIR, 'log.txt')
    
    if osp.exists(log_path):
        if check_log_valid(log_path):
            print('This task was already finished:', cfg.OUTPUT_DIR)
            return
        else:
            print('This task was not finished successfully', cfg.OUTPUT_DIR)
            shutil.rmtree(cfg.OUTPUT_DIR)
    
    print('Running task:', cfg.OUTPUT_DIR)
    
    setup_logger(cfg.OUTPUT_DIR)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    try:
        trainer = build_trainer(cfg)

        if args.eval_only:
            trainer.load_model(args.model_dir, epoch=args.load_epoch)
            trainer.test()
            return

        if not args.no_train:
            trainer.train()
    except:
        e = traceback.format_exc()
        with open(exception_path, 'w') as f:
            f.write(e)
        raise Exception('Training task does not run successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
