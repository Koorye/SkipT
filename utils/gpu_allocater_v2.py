import ctypes
import inspect
import os
import time
from threading import Thread


def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class RunCommandThread(Thread):
    def __init__(self, command, gpus_free, gpu_ids, results, gpu_id):
        Thread.__init__(self)
        self.command = command
        self.gpus_free = gpus_free
        self.gpu_ids = gpu_ids
        self.results = results
        self.gpu_id = gpu_id
        
    def run(self):
        self.gpus_free[self.gpu_ids.index(self.gpu_id)] = False
        command = f'CUDA_VISIBLE_DEVICES={self.gpu_id} {self.command}'
        result = os.system(command)
        self.gpus_free[self.gpu_ids.index(self.gpu_id)] = True
        self.results.append(result)


class GPUAllocater(object):
    def __init__(self, gpu_ids):
        self.gpu_ids = gpu_ids
        self.gpus_free = [True] * len(gpu_ids)
        self.commands = []
        self.threads = []
        self.results = []
    
    def add_commands(self, commands):
        self.commands += commands
    
    def add_command(self, command):
        self.commands.append(command)
        
    def reset(self):
        self.commands = []
        self.threads = []
        self.results = []
    
    def run(self):
        commands_summary = '\n'.join([c.replace('\\', '').replace('\n', ' ') for c in self.commands])
        print(f'All commands:\n{commands_summary}')
        print(f'Number of commands: {len(self.commands)}\n')
        
        while len(self.commands) > 0:
            self.check_results()
            gpu_ids = self.find_free_gpus()
            
            if len(gpu_ids) == 0:
                time.sleep(1)
                continue

            command = self.commands.pop(0)

            if command == 'wait':
                if not all(self.gpus_free):
                    self.commands.insert(0, 'wait')
                    time.sleep(1)
                    continue
                else:
                    command = self.commands.pop(0)
            
            gpu_id = gpu_ids[0]
            print(f'Allocating to GPU {gpu_id}:\n{command}\n')
            thread = RunCommandThread(command, self.gpus_free, self.gpu_ids, self.results, gpu_id)
            thread.start()
            self.threads.append(thread)
            time.sleep(1)
            
        while True:
            self.check_results()
            if all(self.gpus_free):
                break
            time.sleep(1)

    def find_free_gpus(self):
        free_idxs = [i for i, free in enumerate(self.gpus_free) if free]
        return [self.gpu_ids[i] for i in free_idxs]
    
    def check_results(self):
        for res in self.results:
            if res != 0:
                for thread in self.threads:
                    stop_thread(thread)
                raise Exception('Error in running commands')
