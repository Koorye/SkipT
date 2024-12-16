import argparse
import os
import os.path as osp
import pandas as pd
import re


_ORDERS_ALL = ['imagenet', 'caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 
               'food101', 'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101']

_ORDERS_BASE_TO_NEW = ['imagenet', 'caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 
                       'food101', 'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101']

_ORDERS_CROSS_DATASET = ['imagenet', 'caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 
                         'food101', 'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101', 
                         'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r']


def get_parent(path):
    path = path.replace('\\', '/')
    if path.endswith('/'):
        path = path[:-1]
    if '/' not in path:
        return '.'
    return '/'.join(path.split('/')[:-1])


class ResultParser(object):
    def __init__(self, mode, dir_, result_path, cost_path):
        self.mode = mode
        self.dir_ = dir_
        self.result_path = result_path
        self.cost_path = cost_path
    
    def parse_and_save(self):
        if self.mode == 'all':
            acc_df, cost_df = self.read_all()
        elif self.mode == 'b2n':
            acc_df, cost_df = self.read_base_to_new()
        elif self.mode == 'xd':
            acc_df, cost_df = self.read_cross_dataset()
        else:
            raise NotImplementedError
        self.save(acc_df, cost_df)

    def load_property(self, dir_):
        """ get property (trainer, datasets, num_shots, cfg, seeds) from directory """
        trainer = [subdir for subdir in os.listdir(dir_) if osp.isdir(osp.join(dir_, subdir))][0]
        
        dir_ = osp.join(dir_, trainer)
        datasets = os.listdir(dir_)

        if self.mode == 'b2n':
            datasets = [dataset for dataset in _ORDERS_BASE_TO_NEW if dataset in datasets]
        elif self.mode == 'all':
            datasets = [dataset for dataset in _ORDERS_ALL if dataset in datasets]
        elif self.mode == 'xd':
            datasets = [dataset for dataset in _ORDERS_CROSS_DATASET if dataset in datasets]
        else:
            raise NotImplementedError
        
        dir_ = osp.join(dir_, datasets[0])
        num_shots = int(os.listdir(dir_)[0][5:])
        
        dir_ = osp.join(dir_, f'shots{num_shots}')
        cfg = os.listdir(dir_)[0]
        
        dir_ = osp.join(dir_, cfg)
        seeds = list(sorted([int(name[4:]) for name in os.listdir(dir_)]))
        return {
            'trainer': trainer,
            'datasets': datasets,
            'num_shots': num_shots,
            'cfg': cfg,
            'seeds': seeds
        }
    
    def read_all(self):
        prop = self.load_property(self.dir_)
        acc_dfs, cost_dfs = [], []
        
        for dataset in prop['datasets']:
            for seed in prop['seeds']:
                log_path = self._get_log_path(self.dir_, prop, dataset, seed)
                acc, training_time, test_time, memory, parameter = self._read_result(log_path)
                acc_dfs.append(pd.DataFrame({
                    'dataset': [dataset],
                    'seed': [seed],
                    'acc': [acc],
                }))
                cost_dfs.append(pd.DataFrame({
                    'dataset': [dataset],
                    'seed': [seed],
                    'training_time': [training_time],
                    'test_time': [test_time],
                    'memory': [memory],
                    'parameter': [parameter],
                }))
                
        acc_df = pd.concat(acc_dfs, ignore_index=True)
        cost_df = pd.concat(cost_dfs, ignore_index=True)
        acc_df = self._pivot(acc_df, ['acc'], prop)
        cost_df = self._pivot(cost_df, ['training_time', 'test_time', 'memory', 'parameter'], prop)
        acc_df = self._stat(acc_df, 'average', prop)
        cost_df = self._stat(cost_df, 'all', prop)
        return acc_df, cost_df
        
    def read_base_to_new(self):
        base_dir, new_dir = osp.join(self.dir_, 'train_base'), osp.join(self.dir_, 'test_new')
        prop = self.load_property(base_dir)
        acc_dfs, cost_dfs = [], []
        
        for dataset in prop['datasets']:
            for seed in prop['seeds']:
                base_log_path = self._get_log_path(base_dir, prop, dataset, seed)
                new_log_path = self._get_log_path(new_dir, prop, dataset, seed)
                acc_base, training_time_base, test_time_base, memory, parameter = self._read_result(base_log_path)
                acc_new, _, test_time_new, _, _ = self._read_result(new_log_path)
                H = 2 / (1 / acc_base + 1 / acc_new)
                acc_dfs.append(pd.DataFrame({
                    'dataset': [dataset],
                    'seed': [seed],
                    'base_acc': [acc_base],
                    'new_acc': [acc_new],
                    'H': [H],
                }))
                cost_dfs.append(pd.DataFrame({
                    'dataset': [dataset],
                    'seed': [seed],
                    'training_time': [training_time_base],
                    'base_test_time': [test_time_base],
                    'new_test_time': [test_time_new], 
                    'memory': [memory],
                    'parameter': [parameter],
                }))
        
        acc_df = pd.concat(acc_dfs, ignore_index=True)
        cost_df = pd.concat(cost_dfs, ignore_index=True)
        acc_df = self._pivot(acc_df, ['base_acc', 'new_acc', 'H'], prop)
        cost_df = self._pivot(cost_df, ['training_time', 'base_test_time', 
                                        'new_test_time', 'memory', 'parameter'], prop)

        acc_df = self._stat(acc_df, 'average', prop)
        for seed in prop['seeds']:
            acc_df[f'H_seed{seed}'] = 2 / (1 / acc_df[f'base_acc_seed{seed}'] + 1 / acc_df[f'new_acc_seed{seed}'])
        acc_df['H'] = 2 / (1 / acc_df['base_acc'] + 1 / acc_df['new_acc'])
        
        cost_df = self._stat(cost_df, 'all', prop)
        return acc_df, cost_df
    
    def read_cross_dataset(self):
        prop = self.load_property(self.dir_)
        acc_dfs, cost_dfs = [], []
        
        for dataset in prop['datasets']:
            for seed in prop['seeds']:
                log_path = self._get_log_path(self.dir_, prop, dataset, seed)
                acc, training_time, test_time, memory, parameter = self._read_result(log_path)
                acc_dfs.append(pd.DataFrame({
                    'dataset': [dataset],
                    'seed': [seed],
                    'acc': [acc],
                }))
                cost_dfs.append(pd.DataFrame({
                    'dataset': [dataset],
                    'seed': [seed],
                    'training_time': [training_time],
                    'test_time': [test_time],
                    'memory': [memory],
                    'parameter': [parameter],
                }))
                
        acc_df = pd.concat(acc_dfs, ignore_index=True)
        cost_df = pd.concat(cost_dfs, ignore_index=True)
        acc_df = self._pivot(acc_df, ['acc'], prop)
        cost_df = self._pivot(cost_df, ['training_time', 'test_time', 'memory', 'parameter'], prop)

        dg_datasets = [dataset for dataset in prop['datasets']
                      if 'imagenet' in dataset and dataset != 'imagenet']
        xd_datasets = [dataset for dataset in prop['datasets']
                      if dataset not in dg_datasets and dataset != 'imagenet']
        acc_df.loc[len(acc_df)] = ['average_xd'] + acc_df.loc[acc_df['dataset'].isin(xd_datasets)].drop('dataset', axis=1).mean().tolist()
        acc_df.loc[len(acc_df)] = ['average_dg'] + acc_df.loc[acc_df['dataset'].isin(dg_datasets)].drop('dataset', axis=1).mean().tolist()
        
        acc_df = self._stat(acc_df, None, prop)
        cost_df = self._stat(cost_df, 'all', prop)
        return acc_df, cost_df
        
    def save(self, acc_df, cost_df):
        result_dir = get_parent(self.result_path)
        os.makedirs(result_dir, exist_ok=True)
        acc_df.round(2).to_csv(self.result_path, index=None)

        cost_dir = get_parent(self.cost_path)
        os.makedirs(cost_dir, exist_ok=True)
        cost_df.round(2).to_csv(self.cost_path, index=None)

    def _get_log_path(self, dir_, prop, dataset, seed):
        num_shots = prop.get('num_shots')
        return osp.join(dir_, prop['trainer'], dataset, f'shots{num_shots}', prop['cfg'], f'seed{seed}', 'log.txt')

    def _read_result(self, path):
        with open(path, encoding='utf-8') as f:
            content = ''.join(f.readlines())
        
        try:
            acc = float(re.findall(r'accuracy\: (\d+\.\d*)\%', content)[-1])
        except:
            print(f'Accuracy is not found in file {path}.')
            acc = 0
        
        try:
            # Training time: 31327.27s
            training_time = float(re.findall(r'Training time\: (\d+\.\d*)s', content)[-1])
        except:
            training_time = 0
        
        try:
            # Test time: 70.30s, Speed: 355.63 img/s, Memory: 333.77 MB
            test_time = float(re.findall(r'Test time\: (\d+\.\d*)s', content)[-1])
        except:
            test_time = 0

        try:
            # Estimated Total Size (MB): 8630.85
            memory = float(re.findall(r'Estimated Total Size \(MB\)\: (\d+\.\d*)', content)[-1])
        except:
            memory = 0
        
        try:
            # Trainable params: 8,192
            parameter = int(re.findall(r'Trainable params\: (\d+)', content)[-1])
        except:
            parameter = 0
        
        return acc, training_time, test_time, memory, parameter

    def _pivot(self, df, values, prop):
        dfs = []
        for idx, value in enumerate(values):
            df_ = df.pivot_table(index='dataset', columns='seed', values=value).reset_index()
            df_.columns = ['dataset'] + [f'{value}_seed{seed}' for seed in prop['seeds']]
            if idx > 0:
                df_ = df_.drop('dataset', axis=1)
            dfs.append(df_)
        df = pd.concat(dfs, axis=1)
        return df.set_index('dataset').loc[prop['datasets']].reset_index()
    
    def _stat(self, df, mode, prop):
        if mode == 'total':
            df.loc[len(df)] = ['total'] + df.iloc[:, 1:].sum().tolist()
        elif mode == 'average':
            df.loc[len(df)] = ['average'] + df.iloc[:, 1:].mean().tolist()
        elif mode == 'all':
            df.loc[len(df)] = ['average'] + df.iloc[:, 1:].mean().tolist()
            df.loc[len(df)] = ['total'] + df.iloc[:-1, 1:].sum().tolist()
        
        # base_seed1, base_seed2, base_seed3, new_seed1, new_seed2, new_seed3
        # -> base, base, base, new, new, new -> base, new
        columns = ['_'.join(col.split('_')[:-1]) for col in df.columns.tolist() if 'seed' in col]
        # unique and keep order
        columns = list(dict.fromkeys(columns))
        
        for col in columns:
            df[col] = df[[f'{col}_seed{seed}' for seed in prop['seeds']]].mean(axis=1)
        
        return df
        

def main(args):
    parser = ResultParser(args.mode, args.dir, args.result_path, args.cost_path)
    parser.parse_and_save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='mode, b2n or xd')
    parser.add_argument('--dir', type=str, help='directory which need to stats')
    parser.add_argument('--result-path', type=str, help='directory to save statistics')
    parser.add_argument('--cost-path', type=str, help='directory to save statistics')
    args = parser.parse_args()
    main(args)
