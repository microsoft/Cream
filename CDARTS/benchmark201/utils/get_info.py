import os
import argparse

parser = argparse.ArgumentParser(description='supernet training')
parser.add_argument('path', type=str, default='train',
                    help='mode')

args = parser.parse_args()

def main():
    file_path = args.path
    info = {}
    cnt = 0
    dataset_idx = 0
    dataset = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
    acc = [['train', 'valid'], ['train', 'test'], ['train', 'valid', 'test'], ['train', 'valid', 'test']]
    with open(file_path, 'r') as f:
        for line in f:
            line = line.split(' ')
            if 'datasets' in line:
                cnt = cnt + 1
                info[cnt] = {}
                dataset_idx = 0
            if line[0] in dataset:
                top1 = []
                info[cnt][line[0]] = {}
                for item in line:
                    if '%' in item:
                        item = item.split("%")[0]
                        top1.append(float(item))
                if len(top1) > 0:
                    for value, name in zip(top1, acc[dataset_idx]):
                        info[cnt][line[0]][name] = value

                    dataset_idx = dataset_idx + 1

    for key in info.keys():
        print(key, info[key])

if __name__ == '__main__':
    main()