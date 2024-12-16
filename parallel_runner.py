import datetime
import json
import multiprocessing
import numpy as np
import os
import os.path as osp
import shutil
import time

from utils.gpu_allocater_v2 import GPUAllocater
from utils.logger import setup_logger
from utils.mail import MailClient
from utils.misc import find_files, as_yagmail_inline
from utils.plot_v2 import Ploter
from utils.result_parser_v2 import ResultParser
from utils.templates import get_command

# from configs_baseline import get_pipeline
from configs import get_pipeline


class ParallelRunner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.data_cfg = cfg['data']
        self.train_cfg = cfg['train']
        self.grid_search_cfg = cfg['grid_search']
        self.output_cfg = cfg['output']
        self.mail_cfg = cfg['mail']
        
        self.allocater = GPUAllocater(cfg['gpu_ids'])
        self.mail = MailClient(self.mail_cfg)

        self.output_cfg['root'] = osp.join(self.output_cfg['root'], cfg['name'])
        self.output_cfg['result'] = osp.join(self.output_cfg['result'], cfg['name'])
        self.output_cfg['cost'] = osp.join(self.output_cfg['cost'], cfg['name'])

    def run(self):
        grid_search_cfg = self.grid_search_cfg
        output_cfg = self.output_cfg

        # init logger
        setup_logger(osp.join(output_cfg['root'], 'log.txt'), write_to_console=True)
        # self.wait_gpu()
        
        # remove & recreate exist directories
        remove_dirs = [output_cfg[name] for name in output_cfg['remove_dirs']]
        for dir_ in remove_dirs:
            if osp.exists(dir_):
                print(f'Removing directory {dir_}...')
                shutil.rmtree(dir_)
                
            os.makedirs(dir_)
        
        print('Config:')
        print(json.dumps(self.cfg, indent=2))
        print('')

        start_time = datetime.datetime.now()

        try:
            # main function
            if len(grid_search_cfg['params']) > 0:
                result_paths = self.run_grid_search() 
            else:
                result_paths = [self.run_single()]
        except:
            # if exception occurs, send mail
            end_time = datetime.datetime.now()
            contents = [f'<b>Training tasks FAILED!</b> Time cost: {end_time - start_time}\n\n', 
                        '<b>Exception is following above:</b>\n']

            exception_path = find_files(output_cfg['root'], 'exceptions.txt')[0]
            with open(exception_path) as f:
                contents += f.readlines()
            
            print('Training tasks FAILED! Mail will be sent >>> {}'.format(self.mail_cfg['to']))
            self.mail.send('Training Tasks FAILED!', contents)
            return -1

        # send mail while task is finished
        end_time = datetime.datetime.now()
        contents = [f'<b>Training tasks FINISHED!</b> Time cost: {end_time - start_time}\n\n', 
                    '<b>Results are following above:</b>\n']
        
        for path in result_paths:
            if path.endswith('.csv'):
                contents += [f'\n<b>{path}</b>\n']
                with open(path) as f:
                    contents += f.readlines()
            else:
                contents.append(as_yagmail_inline(path))

        print('Training tasks FINISHED! Mail will be sent >>> {}'.format(self.mail_cfg['to']))
        self.mail.send('Training Tasks FINISHED!', contents)
        return 0

    def run_grid_search(self):
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg
        grid_search_cfg = self.grid_search_cfg
        
        root = output_cfg['root']
        result_dir = output_cfg['result']

        # generate grid search opts
        # suffix: [alias1=value1_alias2=value2..., ...]
        # opts_list: [[name1, value1, name2, value2, ...], ...]
        suffixs, opts_list, aliases = self.get_grid_search_opts()
        
        print('Grid search opts:')
        for opts in opts_list:
            print(opts)
        print()
        
        result_paths = []
        
        # run per single task with grid search opts
        for idx, (suffix, opts) in enumerate(zip(suffixs, opts_list)):
            print(f'[{idx + 1} / {len(suffixs)}] Running task {opts}\n')
            output_cfg['root'] = osp.join(root, suffix)
            result_paths.append(self.run_single(opts, suffix))
        
        # plot results
        prefix = '{}-{}-{}'.format(train_cfg['mode'], train_cfg['trainer'], train_cfg['cfg'])
        suffix = '_'.join(aliases)
        save_path = osp.join(result_dir, '{}-{}-{}-{}.jpg'.format(train_cfg['mode'], train_cfg['trainer'],
                                                                  train_cfg['cfg'], suffix))
        mode = grid_search_cfg['plot']
        ploter = Ploter(result_dir, prefix, aliases, save_path, mode)
        ploter.plot()
        result_paths.append(save_path)
        
        return result_paths

    def run_single(self, opts=[], suffix=None):
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg
        
        extra_opts = train_cfg['opts']
        opts += extra_opts
        
        # save results
        if suffix is None:
            filename = '{}-{}-{}.csv'.format(train_cfg['mode'], train_cfg['trainer'], train_cfg['cfg'])
        else:
            filename = '{}-{}-{}-{}.csv'.format(train_cfg['mode'], train_cfg['trainer'], train_cfg['cfg'], suffix)
        result_path = osp.join(output_cfg['result'], filename)
        cost_path = osp.join(output_cfg['cost'], filename)

        if osp.exists(result_path):
            print(f'Results already exist >>> {result_path}\n')
            return result_path
        
        # generate commands
        if train_cfg['mode'] == 'all':
            commands = self.get_all_commands(opts)
        elif train_cfg['mode'] == 'b2n':
            commands = self.get_base_to_new_commands(opts)
        elif train_cfg['mode'] == 'xd':
            commands = self.get_cross_dataset_commands(opts)
        else:
            raise NotImplementedError
        
        if train_cfg['load_from'] != '':
            src = osp.join(train_cfg['load_from'], train_cfg['trainer'])
            dst = osp.join(output_cfg['root'], train_cfg['trainer'])
            shutil.copytree(src, dst)
        
        # add commands and run
        self.allocater.reset()
        self.allocater.add_commands(commands)
        self.allocater.run()

        print(f'Results will be save >>> {result_path}')
        os.makedirs(output_cfg['result'], exist_ok=True)
        parser = ResultParser(train_cfg['mode'], output_cfg['root'], result_path, cost_path)
        parser.parse_and_save()
        
        return result_path

    def get_all_commands(self, opts=[]):
        data_cfg = self.data_cfg
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg
        commands = []

        # training on all datasets
        for dataset in data_cfg['datasets_all']:
            for seed in train_cfg['seeds']:
                cmd = get_command(data_cfg['root'], seed, train_cfg['trainer'], dataset, 
                                  train_cfg['cfg'], output_cfg['root'], train_cfg['shots'], dataset, 
                                  train_cfg['loadep'], opts, mode='all', train=True)
                commands.append(cmd)

        return commands
    
    def get_base_to_new_commands(self, opts=[]):
        data_cfg = self.data_cfg
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg
        commands = []

        # training on all datasets
        for dataset in data_cfg['datasets_base_to_new']:
            for seed in train_cfg['seeds']:
                cmd = get_command(data_cfg['root'], seed, train_cfg['trainer'], dataset, 
                                train_cfg['cfg'], output_cfg['root'], train_cfg['shots'], dataset,
                                train_cfg['loadep'], opts, mode='b2n', train=True)
                commands.append(cmd)
        commands.append('wait')

        # testing on all datasets
        for dataset in data_cfg['datasets_base_to_new']:
            for seed in train_cfg['seeds']:
                cmd = get_command(data_cfg['root'], seed, train_cfg['trainer'], dataset, 
                                  train_cfg['cfg'], output_cfg['root'], train_cfg['shots'], dataset,
                                  train_cfg['loadep'], opts, mode='b2n', train=False)
                commands.append(cmd)
                
        return commands
    
    def get_cross_dataset_commands(self, opts):
        data_cfg = self.data_cfg
        train_cfg = self.train_cfg
        output_cfg = self.output_cfg
        commands = []
        
        # training on all datasets
        load_dataset = 'imagenet'
        for seed in train_cfg['seeds']:
            cmd = get_command(data_cfg['root'], seed, train_cfg['trainer'], load_dataset, 
                            train_cfg['cfg'], output_cfg['root'], train_cfg['shots'], load_dataset,
                            train_cfg['loadep'], opts, mode='xd', train=True)
            commands.append(cmd)
        commands.append('wait')

        # testing on all datasets
        for dataset in data_cfg['datasets_cross_dataset']:
            for seed in train_cfg['seeds']:
                cmd = get_command(data_cfg['root'], seed, train_cfg['trainer'], dataset, 
                                  train_cfg['cfg'], output_cfg['root'], train_cfg['shots'], load_dataset,
                                  train_cfg['loadep'], opts, mode='xd', train=False)
                commands.append(cmd)
                
        return commands
    
    def get_grid_search_opts(self):
        grid_search_cfg = self.grid_search_cfg
        mode = grid_search_cfg['mode']
        params = grid_search_cfg['params']

        names = [param['name'] for param in params]
        aliases = [param['alias'] for param in params]
        values_list = [param['values'] for param in params]
        
        # grid to sequential
        if mode == 'grid' and len(names) > 1:
            values_list = [list(arr.flatten()) for arr in np.meshgrid(*values_list)]
        
        # build opts
        suffixs, grid_search_opts_list = [], []
        for i in range(len(values_list[0])):
            values = [values[i] for values in values_list]

            # suffix is like "alias1=value1_alias2=value2..."
            suffix, opts = [], []
            for name, alias, value in zip(names, aliases, values):
                suffix.append(f'{alias}={value}')
                opts += [name, value]
            
            suffix = '_'.join(suffix)
            suffixs.append(suffix)
            grid_search_opts_list.append(opts)
            
        return suffixs, grid_search_opts_list, aliases
    
    def wait_gpu(self):
        def get_free_memory(gpu_id):
            cmd = f'nvidia-smi --query-gpu=memory.free --format=csv,nounits -i {gpu_id}'
            free_memory = os.popen(cmd).readlines()[1].strip()
            return int(free_memory)

        while True:
            is_all_free = True
            for gpu in self.cfg['gpu_ids']:
                free_memory = get_free_memory(gpu)
                if free_memory < 10000:
                    print(f'GPU {gpu} is busy, waiting...')
                    is_all_free = False
            if is_all_free:
                break
            time.sleep(60)
            
        print('All GPUs are free, continue...')
        return True


class RunnersProcess(multiprocessing.Process):
    def __init__(self, name, cfgs):
        super().__init__()
        self.name = name
        self.runners = [ParallelRunner(cfg) for cfg in cfgs]

    def run(self):
        for idx, runner in enumerate(self.runners):
            print(f'{self.name} >>> Running task {idx + 1} / {len(self.runners)} {runner.cfg["name"]}') 
            result = runner.run()

            if result == -1:
                print(f'{self.name} >>> Task {idx + 1} {runner.cfg["name"]} FAILED!')
                break


def main():
    pipeline = get_pipeline()

    # run tasks in parallel
    # use process to run each pipe, to avoid system confict
    processes = []
    for idx, pipe in enumerate(pipeline):
        processes.append(RunnersProcess(f'pipe{idx}', pipe['tasks']))

    for p in processes:
        p.start()
    
    for p in processes:
        p.join()

    print('All tasks finished!')


if __name__ == '__main__':
    main()
