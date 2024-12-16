import os


_START_PATTERN = 'Initialize tensorboard'
_END_PATTERN = 'Estimated Total Size (MB)'


def find_pattern(lines, pattern):
    for idx, line in enumerate(lines):
        if pattern in line:
            return idx
    raise ValueError(f'Pattern "{pattern}" not found in lines')


def get_parent(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def concat_log(log_path1, log_path2, save_path):
    with open(log_path1, 'r') as f:
        log1 = f.readlines()
    with open(log_path2, 'r') as f:
        log2 = f.readlines()
    
    try:
        start_idx = find_pattern(log1, _START_PATTERN) + 1
        end_idx = find_pattern(log1, _END_PATTERN) + 1
        log1 = log1[start_idx:end_idx + 1]
        insert_idx = find_pattern(log2, _START_PATTERN) + 1
        log2 = log2[:insert_idx] + log1 + log2[insert_idx:]
    except:
        print(f'Error in {log_path1}')
    
    os.makedirs(get_parent(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.writelines(log2)


def concat_logs(log_root1, log_root2, save_root):
    for root, _, filenames in os.walk(log_root1):
        for filename in filenames:
            if filename.endswith('.txt'):
                log_path1 = os.path.join(root, filename)
                log_path2 = log_path1.replace(log_root1, log_root2)
                save_path = log_path1.replace(log_root1, save_root)
                concat_log(log_path1, log_path2, save_path)


if __name__ == '__main__':
    concat_logs('outputs', 'outputs_no_memory', 'outputs_memory')
