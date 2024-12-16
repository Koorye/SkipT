import argparse
import matplotlib.pyplot as plt
import os
import os.path as osp
import pandas as pd
import re
import seaborn as sns
from plotnine import *
from tqdm import tqdm


_INTEGER_PATTERN = r'^-?\d+$'
_FLOAT_PATTERN = r'^\-?\d+\.\d+$'
_SCIENCE_PATTERN = r'^-?\d+(\.\d+)?e-?\d+$'
_SIZE_RATIO = 5


class Ploter(object):
    def __init__(self, dir_, pattern, save_path, mode):
        self.dir_ = dir_
        self.pattern = pattern
        self.save_path = save_path
        self.mode = mode
    
    def plot(self):
        df, variables = self.load_dfs()
        
        if self.mode == 'line' or self.mode == 'col':
            self.plot_1d(df, variables)
        elif self.mode == 'heat':
            self.plot_2d(df, variables)

    def load_dfs(self):
        print(os.listdir(self.dir_))
        filenames = [filename for filename in os.listdir(self.dir_) 
                     if re.match(self.pattern, filename)]
        paths = [osp.join(self.dir_, filename) for filename in filenames]
        dfs = [self.load_df(path) for path in paths]
        dfs_new = []
        
        for df, filename in zip(dfs, filenames):
            trainer, cfg, var_and_values = self.parse_filename(filename)

            df['trainer'] = trainer
            df['cfg'] = cfg
            for var, value in var_and_values.items():
                df[var] = value

            dfs_new.append(df)
        
        return pd.concat(dfs_new, ignore_index=True), list(var_and_values.keys())
    
    def plot_1d(self, df, variables):
        if len(variables) == 1:
            var = variables[0]
        else:
            var = '&'.join(variables)
            df[var] = df.apply(lambda row: '&'.join([str(row[var]) for var in variables]), axis=1)
        
        mode, save_path = self.mode, self.save_path
        if mode == 'line':
            (
                ggplot(df, aes(var, 'acc', color='acc_type', label='acc', group='acc_type'))
                + geom_line()
                + geom_point()
                + facet_wrap('dataset', scales='free_y')
                + theme_seaborn()
                + theme(axis_text_x=element_text(rotation=45))
            ).save(save_path, height=8, width=10)

        elif mode == 'col':
            (
                ggplot(df, aes(var, 'acc', fill='acc_type', label='acc', group='acc_type'))
                + geom_col(position='dodge')
                + facet_wrap('dataset', scales='free_y')
                + theme_seaborn()
                + theme(axis_text_x=element_text(rotation=45))
            ).save(save_path, height=8, width=10)
            
    def plot_2d(self, df, variables):
        assert len(variables) == 2, 'length of variables should be 2!'
        var1, var2 = variables
        datasets = df['dataset'].unique().tolist()
        acc_types = df['acc_type'].unique().tolist()

        plt.figure(figsize=(len(acc_types) * _SIZE_RATIO, len(datasets) * _SIZE_RATIO))

        for r, dataset in enumerate(tqdm(datasets)):
            df_per_dataset = df[df['dataset'] == dataset]
            
            for c, acc_type in enumerate(acc_types):
                df_per_acc_type = df_per_dataset[df_per_dataset['acc_type'] == acc_type][[var1, var2, 'acc']]
                df_2d = df_per_acc_type.set_index([var1, var2]).unstack()

                xticks = [idx[1] for idx in df_2d.columns]
                yticks = df_2d.index.tolist()

                plt.subplot(len(datasets), len(acc_types), r * len(acc_types) + c + 1)
                sns.heatmap(df_2d, annot=True, fmt='.2f', cbar=False,
                            xticklabels=xticks, yticklabels=yticks)
                plt.xlabel(var2)
                plt.ylabel(var1)
                plt.title(dataset + ' ' + acc_type)
        
        plt.savefig(self.save_path)
        plt.close()
        
    def load_df(self, path):
        df = pd.read_csv(path, index_col=None)
        
        if 'H' in df.columns:
            # load base-to-new dataframe
            df = df[['dataset', 'base_acc', 'new_acc', 'H']]
            return pd.melt(df, id_vars=['dataset'], value_vars=['base_acc', 'new_acc', 'H'],
                        var_name='acc_type', value_name='acc')
        else:
            # load cross dataset dataframe
            df = df[['dataset', 'acc']]
            df['acc_type'] = 'all'
            return df

    def parse_filename(self, filename):
        outs = dict()
        filename = '.'.join(filename.split('.')[:-1])
        s = filename.split('-')

        if len(s) == 3:
            _, trainer, cfg = s
            return trainer, cfg, outs

        if len(s) == 4:
            _, trainer, cfg, var_and_values = s
        else:
            trainer = s[1]
            cfg = s[2]
            var_and_values = '-'.join(s[3:])
        
        for var_and_value in var_and_values.split('_'):
            var, value = var_and_value.split('=')

            if re.match(_INTEGER_PATTERN, value):
                value = int(value)
            elif re.match(_FLOAT_PATTERN, value):
                value = float(value)
            elif re.match(_SCIENCE_PATTERN, value):
                value = float(value)
            
            outs[var] = value

        return trainer, cfg, outs


def main(args):
    ploter = Ploter(args.dir, args.pattern, args.save_path, args.mode)
    ploter.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--pattern', type=str)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--mode', type=str, default='line')
    args = parser.parse_args()
    main(args)
