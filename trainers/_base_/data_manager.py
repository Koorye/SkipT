"""
Modified from Dassl
"""
import copy
import random
import torch
from torch.utils.data import Sampler
from collections import defaultdict
from dassl.data.data_manager import DatasetWrapper
from dassl.data.samplers import RandomSampler, SequentialSampler, RandomDomainSampler, SeqDomainSampler, RandomClassSampler


class RandomBatchSampler(Sampler):
    """ Make a batch of samples has the same label. """
    
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        self.class_dict = defaultdict(list)
        for idx, item in enumerate(data_source):
            self.class_dict[item.label].append(idx)

        self.classes = list(self.classes.keys())
        self.classes.sort()
        
        self.num_classes = len(self.classes)
    
    def __iter__(self):
        classes = copy.deepcopy(self.classes)
        class_dict = copy.deepcopy(self.class_dict)
        final_idxs = []

        while True:
            selected_class = random.choice(classes)
            idxs = class_dict[selected_class]
            selected_idxs = random.sample(idxs, self.batch_size)
            final_idxs.extend(selected_idxs)
            
            for idx in selected_idxs:
                class_dict[selected_class].remove(idx)
                
            if len(class_dict[selected_class]) < self.batch_size:
                classes.remove(selected_class)
            
            if len(classes) == 0:
                break

        return iter(final_idxs)
    
    def __len__(self):
        return len(list(self.__iter__()))


def build_sampler(
    sampler_type,
    cfg=None,
    data_source=None,
    batch_size=32,
    n_domain=0,
    n_ins=16
):
    if sampler_type == "RandomSampler":
        return RandomSampler(data_source)

    elif sampler_type == "SequentialSampler":
        return SequentialSampler(data_source)

    elif sampler_type == "RandomDomainSampler":
        return RandomDomainSampler(data_source, batch_size, n_domain)

    elif sampler_type == "SeqDomainSampler":
        return SeqDomainSampler(data_source, batch_size)

    elif sampler_type == "RandomClassSampler":
        return RandomClassSampler(data_source, batch_size, n_ins)

    elif sampler_type == "RandomBatchSampler":
        return RandomBatchSampler(data_source, batch_size)

    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader
