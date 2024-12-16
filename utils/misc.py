import os
import os.path as osp
import yagmail


def find_files(root, filename):
    paths = []
    for root, _, filenames in os.walk(root):
        if filename in filenames:
            paths.append(osp.join(root, filename))
    return paths


def get_parent(path):
    path = path.replace('\\', '/')
    if path.endswith('/'):
        path = path[:-1]
    return '/'.join(path.split('/')[:-1])


def as_yagmail_inline(path):
    return yagmail.inline(path)


def remove_if_exist(path):
    if osp.exists(path):
        os.remove(path)


def check_log_valid(path):
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('* accuracy:'):
            return True

    return False