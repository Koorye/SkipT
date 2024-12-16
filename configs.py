import copy
from deepmerge import Merger


base = dict(
    # dataset configs
    data = dict(
        root='your/data/root/here',
        datasets_base_to_new=['imagenet', 'sun397', 'stanford_cars', 'oxford_flowers', 'food101', 'ucf101', 
                              'caltech101', 'fgvc_aircraft', 'dtd', 'oxford_pets', 'eurosat'],
        datasets_cross_dataset=['caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'food101',
                                'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101',
                                'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r'],
        datasets_all=['imagenet', 'sun397', 'stanford_cars', 'oxford_flowers', 'food101', 'ucf101', 
                      'caltech101', 'fgvc_aircraft', 'dtd', 'oxford_pets', 'eurosat'],
    ),

    # mail configs
    mail = dict(
        username='your@mail.com',
        password='password here',
        host='your.host.com',
        to='your@mail.com',
    ),

    # training configs
    train = dict(
        mode='b2n',
        seeds=[1, 2, 3],
        load_from='',
        loadep=-1,
        shots=16,
        opts=[],
    ),

    # grid search configs
    grid_search = dict(
        plot='line',
        mode='sequential',
        params=[]
    ),

    # output configs
    output = dict(
        root='outputs',
        result='results/acc',
        cost='results/cost',
        remove_dirs=[],
    ),
)

#####################################################

# Base-to-New Generalization
skip_tuning = dict(
    train = dict(
        trainer='SkipTuning',
        cfg='vit_b16_bs4',
    ),
)

# Cross Dataset Transfer & Domain Generalization
skip_tuning_xd = dict(
    train = dict(
        mode='xd',
        trainer='SkipTuning',
        cfg='vit_b16_bs4_cross_datasets',
    ),
)

# Few-shot Learning
skip_tuning_all = dict(
    train = dict(
        mode='all',
        trainer='SkipTuning',
        cfg='vit_b16_bs4_few_shot',
    ),

    grid_search = dict(
        params=[
            dict(
                name='DATASET.NUM_SHOTS',
                alias='shot',
                values=[1, 2, 4, 8, 16],
            ),
        ],
    )
)

#####################################################
# Ablation study

skip_tuning_layer = dict(
    train = dict(
        trainer='SkipTuning',
        cfg='vit_b16_bs4',
    ),
    
    grid_search = dict(
        params=[
            dict(
                name='TRAINER.SKIP.START_LAYER',
                alias='layer',
                values=[2, 4, 6, 8, 10],
            )
        ]
    )
)

skip_tuning_top = dict(
    train = dict(
        trainer='SkipTuning',
        cfg='vit_b16_bs4',
    ),
    
    grid_search = dict(
        params=[
            dict(
                name='TRAINER.SKIP.TOP_RATIO',
                alias='top',
                values=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
            )
        ]
    )
)

skip_tuning_lambda = dict(
    train = dict(
        trainer='SkipTuning',
        cfg='vit_b16_bs4',
    ),
    
    grid_search = dict(
        params=[
            dict(
                name='TRAINER.SKIP.LAMBDA',
                alias='lambda',
                values=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            )
        ]
    )
)

#####################################################

pipeline = [
    dict(
        gpu_ids=[0],
        tasks=[
            'skip_tuning',
            'skip_tuning_xd',
            'skip_tuning_all',
        ]
    )
]

#####################################################


def get_pipeline():
    global base, pipeline

    pipeline = copy.deepcopy(pipeline)
    merger = Merger([(list, ['override']), (dict, ['merge']), (set, ['override'])],
                    ['override'], ['override'])

    for pipe in pipeline:
        tasks = []

        for task in pipe['tasks']:
            base_cfg = copy.deepcopy(base)
            cfg = copy.deepcopy(eval(task))
            cfg = merger.merge(base_cfg, cfg)
            cfg['gpu_ids'] = pipe['gpu_ids']
            cfg['name'] = task
            tasks.append(copy.deepcopy(cfg))

        pipe['tasks'] = tasks

    return pipeline
