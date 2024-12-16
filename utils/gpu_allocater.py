import datetime
import os
from threading import Thread


class RunCommandThread(Thread):
    def __init__(self, command):
        Thread.__init__(self)
        self.command = command

    def run(self):
        self.result = os.system(self.command)

    def get_result(self):
        return self.result


class GPUAllocater(object):
    def __init__(self, gpu_ids):
        self.gpu_ids = gpu_ids

        self.num_gpus = len(gpu_ids)
        self.commands = []
        self._current_command_idx = 0
    
    def add_commands(self, commands):
        self.commands += commands
    
    def add_command(self, command):
        self.commands.append(command)
        
    def clear(self):
        self.commands = []
        self._current_command_idx = 0
    
    def run(self):
        print('Summary of all commands:')
        for command in self.commands:
            command_ = command.replace('\\', '').replace('\n', ' ')
            print(command_)
        print('=' * 40)
        
        num_commands = len(self.commands)
        print(f'Number of commands: {num_commands}\n')
        
        while len(self.commands) > 0:
            next_command = self.get_command()
            print(f'[{self._current_command_idx} / {num_commands}] Running commands:')
            self.run_once(next_command)
    
    def get_command(self):
        if self.commands[0] == 'wait':
            self.commands = self.commands[1:]
            self._current_command_idx += 1

        next_commands, reserve_commands = self.commands[:self.num_gpus], self.commands[self.num_gpus:]
        
        if 'wait' not in next_commands:
            self.commands = reserve_commands
            self._current_command_idx += len(next_commands)
            return next_commands
        else:
            idx = next_commands.index('wait')
            reserve_commands = next_commands[idx + 1:] + reserve_commands
            next_commands = next_commands[:idx]

            self._current_command_idx += len(next_commands)
            self.commands = reserve_commands
            return next_commands

    def run_once(self, commands):
        tasks = []

        print('=' * 40)
        for idx, command in enumerate(commands):
            gpu_id = self.gpu_ids[idx]
            command = f'CUDA_VISIBLE_DEVICES={gpu_id} ' + command
            
            print(command)
            if idx != len(commands) - 1:
                print('\n')
            
            t = RunCommandThread(command)
            tasks.append(t)

        print('=' * 40)
        print('Starting commands...')

        start_time = datetime.datetime.now()
        for t in tasks:
            t.start()
           
        for t in tasks:
            t.join()
        
        # raise exception when one of tasks does not run successfully
        results = [t.get_result() for t in tasks]
        for res in results:
            if res != 0:
                raise Exception('Commands cannot run properly!')

        end_time = datetime.datetime.now()
        print(f'All tasks FINISHED! Time cost: {end_time - start_time}\n')
