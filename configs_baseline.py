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

clip_adapter_origin = dict(
    train = dict(
        trainer='OriginCLIPAdapter',
        cfg='vit_b16_ep100',
    ),
)

clip_adapter = dict(
    train = dict(
        trainer='CLIPAdapter',
        cfg='vit_b16_ep100',
    ),
)

clip_adapter_bs4 = dict(
    train = dict(
        trainer='CLIPAdapter',
        cfg='vit_b16_ep10_bs4',
    ),
)

coop = dict(
    train = dict(
        trainer='CoOp',
        cfg='vit_b16_ep100',
    ),
)

cocoop = dict(
    train = dict(
        trainer='CoCoOp',
        cfg='vit_b16_c4_ep10_batch1_ctxv1',
    ),
)

prograd = dict(
    train = dict(
        trainer='ProGrad',
        cfg='vit_b16_ep100_ctx',
    ),
)

kgcoop = dict(
    train = dict(
        trainer='KgCoOp',
        cfg='vit_b16_ep100_ctxv1',
    ),
)

maple = dict(
    train = dict(
        trainer='MaPLe',
        cfg='vit_b16_c2_ep5_batch4_2ctx',
    ),
)

tcp = dict(
    train = dict(
        trainer='TCP',
        cfg='vit_b16_ep100_ctxv1',
    ),
)

promptsrc = dict(
    train = dict(
        trainer='PromptSRC',
        cfg='vit_b16_c2_ep20_batch4_4+4ctx',
    ),
)

kgdept = dict(
    train = dict(
        trainer='KgDePT',              
        cfg='vit_b16_ep10_ctxv1_bs4_lr35', 
    ),
)

coprompt = dict(
    train = dict(
        trainer='CoPrompt',
        cfg='coprompt',
    ),
)

ftclip = dict(
    train = dict(
        trainer='FinetuneCLIP',
        cfg='vit_b16_ep10_bs4',
    ),
)

#####################################################

coop_xd = dict(
    train = dict(
        mode='xd',
        trainer='CoOp',
        cfg='vit_b16_ep100',
    ),
)

cocoop_xd = dict(
    train = dict(
        mode='xd',
        trainer='CoCoOp',
        cfg='vit_b16_c4_ep10_batch1_ctxv1',
    ),
)

prograd_xd = dict(
    train = dict(
        mode='xd',
        trainer='ProGrad',
        cfg='vit_b16_ep100_ctx',
    ),
)

kgcoop_xd = dict(
    train = dict(
        mode='xd',
        trainer='KgCoOp',
        cfg='vit_b16_ep100_ctxv1',
    ),
)

kgdept_xd = dict(
    train = dict(
        mode='xd',
        trainer='KgDePT',              
        cfg='vit_b16_ep10_ctxv1_bs4_lr35_cross_datasets', 
    ),
)

maple_xd = dict(
    train = dict(
        mode='xd',
        trainer='MaPLe',
        cfg='vit_b16_c2_ep5_batch4_2ctx_cross_datasets',
    ),
)

tcp_xd = dict(
    train = dict(
        mode='xd',
        trainer='TCP',
        cfg='vit_b16_ep100_ctxv1',
    )
)

promptsrc_xd = dict(
    train = dict(
        mode='xd',
        trainer='PromptSRC',
        cfg='vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets',
    ),
)

coprompt_xd = dict(
    train = dict(
        mode='xd',
        trainer='CoPrompt',
        cfg='coprompt',
    ),
)

ftclip_xd = dict(
    train = dict(
        mode='xd',
        trainer='FinetuneCLIP',
        cfg='vit_b16_ep4_bs4_cross_datasets',
    ),
)

#####################################################
# Few-shot Learning

coop_all = dict(
    train = dict(
        mode='all',
        trainer='CoOp',
        cfg='vit_b16_ep100',
    ),

    grid_search = dict(
        params=[
            dict(
                name='DATASET.NUM_SHOTS',
                alias='shot',
                values=[1, 2, 4, 8, 16],
            )
        ]
    )
)

cocoop_all = dict(
    train = dict(
        mode='all',
        trainer='CoCoOp',
        cfg='vit_b16_c4_ep10_batch1_ctxv1',
    ),

    grid_search = dict(
        params=[
            dict(
                name='DATASET.NUM_SHOTS',
                alias='shot',
                values=[1, 2, 4, 8, 16],
            )
        ]
    )
)

maple_all = dict(
    train = dict(
        mode='all',
        trainer='MaPLe',
        cfg='vit_b16_c2_ep5_batch4_2ctx_few_shot',
    ),

    grid_search = dict(
        params=[
            dict(
                name='DATASET.NUM_SHOTS',
                alias='shot',
                values=[1, 2, 4, 8, 16],
            )
        ]
    )
)

promptsrc_all = dict(
    train = dict(
        mode='all',
        trainer='PromptSRC',
        cfg='vit_b16_c2_ep50_batch4_4+4ctx_few_shot',
    ),

    grid_search = dict(
        params=[
            dict(
                name='DATASET.NUM_SHOTS',
                alias='shot',
                values=[1, 2, 4, 8, 16],
            )
        ]
    )
)

#####################################################

pipeline = [
    dict(
        gpu_ids=[0],
        tasks=[
            'coop',
            'cocoop',
            'prograd',
            'kgcoop',
            'maple',
            'tcp',
            'promptsrc',
            'kgdept',
            'coprompt',
            'ftclip',
        ],
    ),
    dict(
        gpu_ids=[0],
        tasks=[
            'coop_xd',
            'cocoop_xd',
            'prograd_xd',
            'kgcoop_xd',
            'kgdept_xd',
            'maple_xd',
            'tcp_xd',
            'promptsrc_xd',
            'coprompt_xd',
            'ftclip_xd',
        ],
    ),
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
