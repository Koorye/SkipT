import argparse
import os
import os.path as osp


def clear_logs(root):
    for root, _, filenames in os.walk(root):
        for filename in filenames:
            if filename == 'checkpoint' or filename.startswith('model.pth.tar'):
                path = osp.join(root, filename)
                print(f'Deleting file {path}')
                os.remove(path)

def main(args):
    root = args.root
    clear_logs(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    args = parser.parse_args()
    main(args)
